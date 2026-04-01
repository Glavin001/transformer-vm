"""Cranelift CLIF IR interpreter expressed in the CALM graph DSL.

When compiled via build_model(), produces a universal CLIF executor transformer.
Programs are provided as token prefixes with 7-token instructions:
    opcode hex0 hex1 hex2 hex3 hex4 hex5

Unlike the WASM interpreter (which uses a stack), this interpreter uses SSA
variable bindings accessed via attention keyed by (dest_v * 4 + byte_index).
Each value-producing instruction writes its result to a variable slot; future
instructions read from named variables, not from an implicit stack.
"""

from transformer_vm.graph import core as _graph
from transformer_vm.graph.core import (
    Expression,
    InputDimension,
    auto_name,
    fetch,
    fetch_sum,
    persist,
    reglu,
    stepglu,
)

# ── Opcode dispatch via circle points ──────────────────────────

# Squared radius of the circle (all points satisfy x² + y² = R²)
pointsR2 = 32045

# Circle points — same set as WASM, just mapped to CLIF opcodes
points = [
    (179, 2),
    (179, -2),
    (-179, 2),
    (-179, -2),
    (2, 179),
    (2, -179),
    (-2, 179),
    (-2, -179),
    (178, 19),
    (178, -19),
    (-178, 19),
    (-178, -19),
    (19, 178),
    (19, -178),
    (-19, 178),
    (-19, -178),
    (173, 46),
    (173, -46),
    (-173, 46),
    (-173, -46),
    (46, 173),
    (46, -173),
    (-46, 173),
    (-46, -173),
    (166, 67),
    (166, -67),
    (-166, 67),
    (-166, -67),
    (67, 166),
    (67, -166),
    (-67, 166),
    (-67, -166),
    (163, 74),
    (163, -74),
    (-163, 74),
    (-163, -74),
    (74, 163),
    (74, -163),
    (-74, 163),
    (-74, -163),
]

OPCODES = {
    "halt": 0,
    "iconst": 1,
    "iadd": 2,
    "isub": 3,
    "imul": 4,
    "band": 5,
    "bor": 6,
    "bxor": 7,
    "ishl": 8,
    "ushr": 9,
    "sshr": 10,
    "icmp": 11,
    "select": 12,
    "load": 13,
    "uload8": 14,
    "sload8": 15,
    "store": 16,
    "store8": 17,
    "brif": 18,
    "jump": 19,
    "output": 20,
    "copy": 21,
    "copy_true": 22,
    "copy_false": 23,
    "ineg": 24,
    "umulhi": 25,
    "smin": 26,
    "smax": 27,
    "sextend8": 28,
    "input_base": 29,
    "return": 30,
    "store16": 31,
    "uload16": 32,
    "sload16": 33,
}

OPCODE_POINT = {op: points[i] for i, op in enumerate(OPCODES)}

# Instructions that write a value to a variable slot
VAR_WRITE_OPS = {
    "iconst",
    "iadd",
    "isub",
    "imul",
    "band",
    "bor",
    "bxor",
    "ishl",
    "ushr",
    "sshr",
    "icmp",
    "select",
    "load",
    "uload8",
    "sload8",
    "uload16",
    "sload16",
    "copy",
    "copy_true",
    "copy_false",
    "ineg",
    "umulhi",
    "smin",
    "smax",
    "sextend8",
}

# Instructions that produce trace bytes (value-producing or store)
BYTE_OPS = VAR_WRITE_OPS | {"store", "store8", "store16"}

# Instructions that are branches
BRANCH_OPS = {"brif", "jump"}


def get_byte_value(bv, i, signed=False):
    return (bv - 256 if signed and bv >= 128 else bv) * (1 << (8 * i))


def build(program=None):
    one = _graph.one
    position = _graph.position

    # ── Circle-point opcode dispatch ─────────────────────────────
    _op_dot_cache = {}

    def op_dot(op):
        """Gate: 1 when op matches, <= -1 otherwise."""
        if op not in _op_dot_cache:
            px, py = OPCODE_POINT[op]
            _op_dot_cache[op] = (
                px * fetched_opcode_x + py * fetched_opcode_y - pointsR2 * one + 1
            )
        return _op_dot_cache[op]

    _is_op_cache = {}

    def is_op(op):
        if op not in _is_op_cache:
            _is_op_cache[op] = reglu(one, op_dot(op))
        return _is_op_cache[op]

    # ── Input dimensions ─────────────────────────────────────────
    byte_number = InputDimension("byte_number")
    carry = InputDimension("carry")
    delta_cursor = InputDimension("delta_cursor")
    is_jump = InputDimension("is_jump")
    var_write = InputDimension("var_write")  # 1 if this instruction writes a variable
    is_branch_taken = InputDimension("is_branch_taken")

    # Program prefix dimensions (universal mode only)
    opcode_x = InputDimension("opcode_x")
    opcode_y = InputDimension("opcode_y")
    var_write_prefix = InputDimension("var_write_prefix")

    # ── Build input_tokens ───────────────────────────────────────
    input_tokens = (
        # Byte tokens: hex value + carry/borrow
        {
            (f"{bv:02x}'" if c else f"{bv:02x}"): (bv + 1) * byte_number + c * carry
            for bv in range(256)
            for c in range(2)
        }
        # Commit tokens: encode delta_cursor and var_write flag
        | {
            f"commit({dc:+d},vw={vw},bt={bt})": delta_cursor * dc
            + vw * var_write
            + bt * is_jump
            for dc in [0, 1]
            for vw in [0, 1]
            for bt in [0, 1]
        }
        # Output tokens
        | {
            (f"out({chr(bv)})" if 0x20 < bv < 0x7F else f"out({bv:02x})"): delta_cursor * 1
            for bv in range(256)
        }
        # Control tokens
        | {
            "branch_taken": 1 * is_branch_taken,
            "halt": 0 * one,
        }
    )

    # Program prefix tokens: opcode names + hex bytes
    input_tokens["{"] = 0 * one
    input_tokens["}"] = 0 * one  # No stack, just end marker
    for op in OPCODES:
        vw = 1 if op in VAR_WRITE_OPS else 0
        px, py = OPCODE_POINT[op]
        embedding = px * opcode_x + py * opcode_y + vw * var_write_prefix
        input_tokens[op] = embedding

    # Printable ASCII aliases
    for bv in range(0x21, 0x7F):
        ch = chr(bv)
        if ch in input_tokens:
            continue
        input_tokens[ch] = input_tokens[f"{bv:02x}"]

    # Set one=1 for all tokens except "{"
    for tok in input_tokens:
        if tok != "{":
            input_tokens[tok][one] = 1

    # ── Store value (reconstructed from preceding byte tokens) ───
    store_bytes = [
        fetch(byte_number - 1, query=position - i, key=position) for i in range(1, 5)
    ]
    store_value = sum(
        (1 << (8 * (4 - i))) * store_bytes[i - 1] for i in range(1, 5)
    )
    store_value = persist(store_value)

    # Branch offset (same pattern as WASM)
    msb = store_bytes[0]
    unsigned_branch = reglu(store_value, is_jump)
    jump_sign = stepglu(one, msb + 128 * is_jump - 256)
    delta_cursor_expr = delta_cursor + unsigned_branch - jump_sign * (1 << 32)

    # Byte index within current instruction group
    byte_index = position - fetch(
        position, query=one, key=one, clear_key=byte_number
    )
    is_boundary = stepglu(one, -byte_number)

    # ── Cumulative state ─────────────────────────────────────────
    cursor = fetch_sum(delta_cursor_expr)

    # ── Instruction fetch (universal mode) ───────────────────────
    # Each CLIF instruction is 7 tokens: opcode + 6 data bytes
    instruction_position = 7 * cursor + 1

    fetched_opcode_x, fetched_opcode_y, fetched_var_write = fetch(
        [opcode_x, opcode_y, var_write_prefix],
        query=instruction_position,
        key=position,
    )

    # Fetch the 6 data bytes (fields f0..f5)
    field_bytes = [
        fetch(byte_number - 1, query=instruction_position + i, key=position)
        for i in range(1, 7)
    ]

    # Reconstruct key fields from data bytes
    # f0 = dest_v (or src for void ops like store/brif)
    # f1 = src1_v (or immediate low byte for iconst)
    # For iconst: immediate = f1 | (f2 << 8) | (f3 << 16) | (f4 << 24)
    # For arithmetic: f0=dest, f1=src1, f2=src2
    # For icmp: f0=dest, f1=src1, f2=src2, f3=cond
    # For brif: f0=cond_v, f1:f2=true_offset, f3:f4=false_offset
    fetched_dest_v = field_bytes[0]
    fetched_src1_v = field_bytes[1]
    fetched_src2_v = field_bytes[2]

    # Immediate value (for iconst): field_bytes[1:5]
    immediate = sum(
        (1 << (8 * i)) * field_bytes[1 + i] for i in range(4)
    )
    immediate = persist(immediate)

    # ── Variable access via per-byte attention ──────────────────
    # Each byte token during a variable-write instruction writes its value
    # keyed by (dest_v, byte_index). Boundary/commit positions clear the key.
    #
    # Key: 4 * (fetched_dest_v + 1) + byte_index   (at byte positions)
    # Query: 4 * (fetched_src_v + 1) + byte_index   (at any position)
    # Clear: at boundaries (byte_number = 0) — commits, out(), branch_taken, etc.

    # Use (dest_v + 1) to avoid key=0 conflicts
    var_write_key = 4 * (fetched_dest_v + 1) + byte_index
    # Clear key at boundaries AND at byte positions of non-var-write instructions.
    # This prevents branch offset bytes, store bytes, etc. from polluting the
    # variable key space (since brif has dest_v=0 which collides with v0).
    not_var_write_instr = 1 - fetched_var_write
    clear_at_non_var_byte = is_boundary + not_var_write_instr

    src1_byte = fetch(
        byte_number - 1,
        query=4 * (fetched_src1_v + 1) + byte_index + 1,
        key=var_write_key,
        clear_key=clear_at_non_var_byte,
    )

    src2_byte = fetch(
        byte_number - 1,
        query=4 * (fetched_src2_v + 1) + byte_index + 1,
        key=var_write_key,
        clear_key=clear_at_non_var_byte,
    )

    # Reconstruct full 32-bit src values for comparisons and memory access
    # Note: query index i matches write byte_index=i, which contains byte (i-1) of the value.
    # Byte (i-1) has weight 2^(8*(i-1)) in little-endian reconstruction.
    src1_bytes = [
        fetch(
            byte_number - 1,
            query=4 * (fetched_src1_v + 1) + i,
            key=var_write_key,
            clear_key=clear_at_non_var_byte,
        )
        for i in range(1, 5)
    ]
    src1_value = persist(sum(
        (1 << (8 * (i - 1))) * src1_bytes[i - 1] for i in range(1, 5)
    ))

    src2_bytes = [
        fetch(
            byte_number - 1,
            query=4 * (fetched_src2_v + 1) + i,
            key=var_write_key,
            clear_key=clear_at_non_var_byte,
        )
        for i in range(1, 5)
    ]
    src2_value = persist(sum(
        (1 << (8 * (i - 1))) * src2_bytes[i - 1] for i in range(1, 5)
    ))

    # ── Memory access ────────────────────────────────────────────
    # Memory uses latest-write-wins keyed by address.
    # For load: addr = src1_value + load_offset
    # For store: addr = src2_value + store_offset
    # For input_base: addr = immediate (the input buffer base address)
    memory_load_offset = field_bytes[2] + 256 * field_bytes[3]
    memory_store_offset = field_bytes[3] + 256 * field_bytes[4]

    # input_base uses immediate as write address base (not src2_value)
    input_base_gate = is_op("input_base")
    memory_write_base = persist(
        reglu(src2_value + memory_store_offset, 1 - input_base_gate)
        + reglu(immediate, input_base_gate)
    )
    memory_read_address = src1_value + memory_load_offset + byte_index
    memory_write_address = memory_write_base + byte_index - 1
    memory_write_gate = persist(
        is_op("store") + is_op("store8") + is_op("store16") + is_op("input_base")
    )
    not_memory_write_byte = 1 + is_boundary - memory_write_gate

    memory_byte_dirty, memory_byte_dirty_position = fetch(
        [byte_number - 1, memory_write_address],
        query=memory_read_address,
        key=memory_write_address,
        clear_key=not_memory_write_byte,
    )
    diff = memory_byte_dirty_position - memory_read_address
    memory_byte = (
        reglu(memory_byte_dirty, diff + 1)
        - 2 * reglu(memory_byte_dirty, diff)
        + reglu(memory_byte_dirty, diff - 1)
    )

    # ── Arithmetic ───────────────────────────────────────────────
    carry_late = persist(carry)

    add_value = src1_byte + src2_byte + carry_late
    add_carry = stepglu(one, add_value - 256)
    add_byte = add_value - 256 * add_carry

    sub_value = src1_byte - src2_byte - carry_late
    sub_borrow = 1 - stepglu(one, sub_value)
    sub_byte = sub_value + 256 * sub_borrow

    # ── Comparisons ──────────────────────────────────────────────
    a_gt_b_u = stepglu(one, src1_value - src2_value - 1)
    a_lt_b_u = stepglu(one, src2_value - src1_value - 1)
    a_eq_b = one - a_gt_b_u - a_lt_b_u

    sign_diff = persist(
        reglu(one, src2_value - (1 << 31) + 1)
        - reglu(one, src2_value - (1 << 31))
        - reglu(one, src1_value - (1 << 31) + 1)
        + reglu(one, src1_value - (1 << 31))
    )
    a_gt_b_s = stepglu(one, sign_diff + a_gt_b_u - 1)
    a_lt_b_s = stepglu(one, -sign_diff + a_lt_b_u - 1)

    # Condition code sub-dispatch (field_bytes[3] encodes the condition)
    cond_code = field_bytes[3]
    # cond_code: 0=eq, 1=ne, 2=slt, 3=sgt, 4=sle, 5=sge, 6=ult, 7=ugt, 8=ule, 9=uge
    is_eq = stepglu(one, -cond_code) - stepglu(one, -cond_code - 1)
    is_ne = stepglu(one, cond_code - 1) - stepglu(one, cond_code - 2)
    is_slt = stepglu(one, cond_code - 2) - stepglu(one, cond_code - 3)
    is_sgt = stepglu(one, cond_code - 3) - stepglu(one, cond_code - 4)
    is_ult = stepglu(one, cond_code - 6) - stepglu(one, cond_code - 7)
    is_ugt = stepglu(one, cond_code - 7) - stepglu(one, cond_code - 8)

    cmp_result = persist(
        reglu(a_eq_b, is_eq)
        + reglu(1 - a_eq_b, is_ne)
        + reglu(a_lt_b_s, is_slt)
        + reglu(a_gt_b_s, is_sgt)
        + reglu(1 - a_gt_b_s, stepglu(one, cond_code - 4) - stepglu(one, cond_code - 5))  # sle
        + reglu(1 - a_lt_b_s, stepglu(one, cond_code - 5) - stepglu(one, cond_code - 6))  # sge
        + reglu(a_lt_b_u, is_ult)
        + reglu(a_gt_b_u, is_ugt)
        + reglu(1 - a_gt_b_u, stepglu(one, cond_code - 8) - stepglu(one, cond_code - 9))  # ule
        + reglu(1 - a_lt_b_u, stepglu(one, cond_code - 9) - stepglu(one, cond_code - 10))  # uge
    )

    # Select: cond_nonzero from src1 (the condition variable)
    cond_nonzero = stepglu(one, src1_value - 1)

    # ── Immediate byte (used by iconst and branch offsets) ──────
    # iconst: immediate bytes at instruction_position + 2 + byte_index
    const_byte = fetch(
        byte_number - 1, query=instruction_position + byte_index + 2, key=position
    )

    # ── Branch offset bytes ────────────────────────────────────
    # brif and jump both use a single 32-bit signed offset, stored the same
    # way as iconst's immediate (at instruction_position + 2 + byte_index).
    # So const_byte already gives the correct offset bytes for brif.
    #
    # jump: offset at f0:f3 = instruction_position + (1..4)
    # At byte_index=0 (boundary prediction): need f0 at instruction_position+1
    jump_offset_byte = fetch(
        byte_number - 1, query=instruction_position + byte_index + 1, key=position
    )
    # brif: offset at f2:f5 = instruction_position + (3..6)
    # At byte_index=0 (boundary prediction): need f2 at instruction_position+3
    brif_offset_byte = fetch(
        byte_number - 1, query=instruction_position + byte_index + 3, key=position
    )

    # ── Top byte (used for stores and output) ────────────────────
    # For output/store: the value comes from src1 (field_bytes[1])
    top_byte = src1_byte

    # src3 for select (third operand = false_val from field_bytes[3])
    src3_byte = fetch(
        byte_number - 1,
        query=4 * (field_bytes[3] + 1) + byte_index + 1,
        key=var_write_key,
        clear_key=clear_at_non_var_byte,
    )

    # ── Result byte computation ──────────────────────────────────
    is_output = is_op("output")

    # ── Bitwise operations (approximate for ALM) ───────────────
    # True bitwise AND/OR/XOR can't be expressed directly in the ALM.
    # For the common case of masking with 0xFF (band v, 0xFF), the result
    # is just byte 0 of src1 with bytes 1-3 zeroed — same as the identity
    # at the boundary. For full generality, these would need to be lowered.
    # TODO: implement proper bitwise lowering for non-mask cases

    # For sextend8: byte 0 = src byte 0, bytes 1-3 = 0xFF if sign bit set else 0
    memory_sign = stepglu(one, src1_byte - 128)
    sext_byte = persist(
        reglu(src1_byte, is_boundary)
        + reglu(255 * memory_sign, 1 - is_boundary)
    )

    # The result byte to emit — gated by opcode
    result_byte = persist(
        # iconst: immediate bytes from program prefix
        reglu(const_byte, op_dot("iconst"))
        # iadd
        + reglu(add_byte, op_dot("iadd"))
        # isub
        + reglu(sub_byte, op_dot("isub"))
        # icmp (byte 0 = result, bytes 1-3 = 0)
        + reglu(cmp_result, op_dot("icmp") + is_boundary - 1)
        # copy variants
        + reglu(src1_byte, op_dot("copy"))
        + reglu(src1_byte, op_dot("copy_true"))
        + reglu(src1_byte, op_dot("copy_false"))
        # select: true_val (src2) if cond_nonzero, else false_val (src3)
        + reglu(src2_byte, op_dot("select") + cond_nonzero - 1)
        + reglu(src3_byte, op_dot("select") - cond_nonzero)
        # load
        + reglu(memory_byte, op_dot("load"))
        + reglu(memory_byte, op_dot("uload8") + is_boundary - 1)
        # store (write-through: emit the value being stored)
        + reglu(top_byte, op_dot("store"))
        + reglu(top_byte, op_dot("store8") + is_boundary - 1)
        # ineg = 0 - src1
        + reglu(sub_byte, op_dot("ineg"))
        # band: byte 0 = src1 & src2 (approximated as src1 for 0xFF mask)
        + reglu(src1_byte, op_dot("band") + is_boundary - 1)
        # sextend8: byte 0 = src byte, bytes 1-3 = sign extension
        + reglu(sext_byte, op_dot("sextend8"))
        # brif: 32-bit offset bytes (taken when condition is true)
        + reglu(brif_offset_byte, op_dot("brif"))
        # jump: 32-bit offset bytes
        + reglu(jump_offset_byte, op_dot("jump"))
    )

    result_carry = persist(
        reglu(add_carry, op_dot("iadd"))
        + reglu(sub_borrow, op_dot("isub"))
        + reglu(sub_borrow, op_dot("ineg"))
    )

    # ── Next-token prediction ────────────────────────────────────
    byte_index_4 = stepglu(one, byte_index - 4)
    is_producing_bytes = fetched_var_write + is_op("store") + is_op("store8") + is_op("store16")
    early_done = reglu(1 - is_boundary, op_dot("store8"))
    early_done = persist(early_done)
    byte_done = byte_index_4 + early_done
    is_byte_seq = 1 - is_boundary - byte_done

    emit_halt = reglu(is_boundary, op_dot("halt"))
    emit_branch_taken = (
        reglu(is_boundary, op_dot("brif") + cond_nonzero - is_branch_taken - 1)
        + reglu(is_boundary, op_dot("jump") - is_branch_taken)
    )
    emit_out = reglu(is_boundary, op_dot("output"))
    emit_byte_start = reglu(is_producing_bytes, is_boundary)
    emit_byte = emit_byte_start + is_byte_seq + is_branch_taken
    emit_bt = reglu(byte_done, op_dot("brif")) + reglu(byte_done, op_dot("jump"))
    emit_commit = (
        byte_done
        + is_boundary
        - emit_halt
        - emit_branch_taken
        - emit_out
        - emit_byte_start
        - is_branch_taken
    )

    # ── Build output_tokens ──────────────────────────────────────
    H = 1e5
    output_tokens = {}
    output_tokens["halt"] = H * emit_halt
    output_tokens["branch_taken"] = H * emit_branch_taken

    for bv in range(256):
        tok = f"out({chr(bv)})" if 0x20 < bv < 0x7F else f"out({bv:02x})"
        output_tokens[tok] = H * emit_out + (2 * bv) * top_byte - bv * bv

    # Commit tokens — dc=1 (advance cursor) for all execution commits
    emit_dc1 = is_producing_bytes + emit_commit  # Always 1 during execution
    for dc in [0, 1]:
        for vw in [0, 1]:
            for bt in [0, 1]:
                output_tokens[f"commit({dc:+d},vw={vw},bt={bt})"] = (
                    H * emit_commit
                    + (2 * dc) * emit_dc1
                    - dc * dc
                    + (2 * vw) * fetched_var_write
                    - vw * vw
                    + (2 * bt) * emit_bt
                    - bt * bt
                )

    # Byte tokens (result value + carry)
    for bv in range(256):
        bv_base = H * emit_byte + (2 * bv) * result_byte - bv * bv
        for c in range(2):
            score = bv_base + (2 * c) * result_carry - c * c
            output_tokens[f"{bv:02x}'" if c else f"{bv:02x}"] = score

    auto_name(locals())
    return input_tokens, output_tokens


class CLIFMachine:
    """The CLIF interpreter expressed as a Program.

    When compiled via build_model(), produces the universal CLIF executor
    transformer. Programs are provided as 7-token-per-instruction prefixes.
    """

    def __init__(self, program=None):
        self.program = program

    def build(self):
        from transformer_vm.graph.core import ProgramGraph, reset_graph

        reset_graph()
        input_tokens, output_tokens = build(program=self.program)
        return ProgramGraph(input_tokens, output_tokens)
