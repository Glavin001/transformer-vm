"""Simplify raw Cranelift CLIF to minimal i32 instruction subset.

Transforms wasmtime's CLIF output (which includes vmctx, 64-bit pointers,
type extensions, call_indirect for imports) into a flat list of simple i32
instructions suitable for the CLIF CALM interpreter.

The subsetting pass:
1. Resolves vmctx-based memory access (heap base, stack pointer)
2. Replaces call_indirect for output_byte with 'output' pseudo-op
3. Removes i64 intermediaries (uextend/ireduce/sextend)
4. Flattens multi-function programs by inlining helper functions
5. Converts blocks to linear instruction list with PC offsets
6. Inserts explicit 'copy' instructions for block parameter bindings
7. Renumbers v-variables sequentially
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .parser import CLIFBlock, CLIFFunction, CLIFInstr

logger = logging.getLogger(__name__)

MASK32 = 0xFFFFFFFF

# Condition codes for icmp
COND_CODES = {
    "eq": 0,
    "ne": 1,
    "slt": 2,
    "sgt": 3,
    "sle": 4,
    "sge": 5,
    "ult": 6,
    "ugt": 7,
    "ule": 8,
    "uge": 9,
}


@dataclass
class SimpleInstr:
    """A simplified CLIF instruction in the minimal i32 subset."""

    opcode: str
    dest: int | None = None  # destination v-number (None for void ops)
    src1: int | None = None  # first source v-number or None
    src2: int | None = None  # second source v-number or None
    src3: int | None = None  # third source (for select: false_val)
    imm: int = 0  # immediate value (for iconst, load/store offset, branch offset)
    cond: int = 0  # condition code for icmp
    # For brif: imm = true_offset, cond_field = false_offset (overloaded)
    false_offset: int = 0


@dataclass
class SimpleProg:
    """A complete simplified CLIF program."""

    instrs: list[SimpleInstr] = field(default_factory=list)
    input_base: int = 0  # memory address where input is stored
    max_var: int = 0  # highest v-number used
    data_segments: list = field(default_factory=list)  # [(offset, bytes), ...]


# ── Analysis: identify vmctx patterns ─────────────────────────


def _find_heap_base(func: CLIFFunction) -> int | None:
    """Find the v-number that holds the heap base pointer.

    Pattern: vN = load.i64 ... v0+56 (vmctx+56 in Wasmtime = heap base)
    """
    for block in func.blocks:
        for instr in block.instrs:
            if instr.opcode == "load" and instr.type == "i64" and instr.offset == 56 and 0 in instr.operands:
                return instr.dest
    return None


def _find_stack_ptr(func: CLIFFunction) -> int | None:
    """Find the v-number that holds the stack pointer.

    Pattern: vN = load.i32 ... v0+96 (vmctx+96 in Wasmtime = __stack_pointer global)
    """
    for block in func.blocks:
        for instr in block.instrs:
            if instr.opcode == "load" and instr.type == "i32" and instr.offset == 96 and 0 in instr.operands:
                return instr.dest
    return None


def _find_output_byte_fn(func: CLIFFunction) -> tuple[int | None, int | None]:
    """Find v-numbers for the output_byte function pointer and vmctx ref.

    Pattern: vN = load.i64 ... v0+72 (function table pointer)
             vM = load.i64 ... v0+88 (callee vmctx)
    Returns (func_ptr_v, callee_vmctx_v).
    """
    func_ptr = None
    callee_ctx = None
    for block in func.blocks:
        for instr in block.instrs:
            if instr.opcode == "load" and instr.type == "i64":
                if instr.offset == 72 and 0 in instr.operands:
                    func_ptr = instr.dest
                elif instr.offset == 88 and 0 in instr.operands:
                    callee_ctx = instr.dest
    return func_ptr, callee_ctx


# ── Subsetting: simplify one function ─────────────────────────


def _subset_function(func: CLIFFunction) -> tuple[list[CLIFBlock], dict[int, int], set[int]]:
    """Simplify a single function's CLIF to minimal i32 subset.

    Returns:
        simplified_blocks: Blocks with simplified instructions
        var_map: Maps original v-numbers to simplified v-numbers
        dead_vars: Set of v-numbers to skip (i64 intermediaries, vmctx loads)
    """
    heap_base_v = _find_heap_base(func)
    stack_ptr_v = _find_stack_ptr(func)
    output_fn_v, output_ctx_v = _find_output_byte_fn(func)

    # Track which v-numbers are i64 intermediaries (to be eliminated)
    dead_vars: set[int] = set()
    # Track which v-numbers are just uextend.i64 of an i32 (map to the i32 source)
    extend_map: dict[int, int] = {}  # v_i64 -> v_i32
    # Track vmctx-related loads
    vmctx_vars: set[int] = set()

    # Mark vmctx params (v0, v1) as dead
    if func.params and func.params[0][0] == "i64":
        dead_vars.add(0)  # vmctx
        vmctx_vars.add(0)
    if len(func.params) > 1 and func.params[1][0] == "i64":
        dead_vars.add(1)  # unused i64 param
        vmctx_vars.add(1)

    if heap_base_v is not None:
        dead_vars.add(heap_base_v)
        vmctx_vars.add(heap_base_v)
    if stack_ptr_v is not None:
        vmctx_vars.add(stack_ptr_v)
    if output_fn_v is not None:
        dead_vars.add(output_fn_v)
        vmctx_vars.add(output_fn_v)
    if output_ctx_v is not None:
        dead_vars.add(output_ctx_v)
        vmctx_vars.add(output_ctx_v)

    # Track i64 constant values for address offset computation
    i64_const_vals: dict[int, int] = {}  # v_num -> constant value
    # Track i64 iadd results that should become i32 iadd with constant offset
    i32_const_for_i64: dict[int, tuple] = {}  # dest_v -> (base_i32_v, const_val, const_var)

    # First pass: identify i64 intermediaries and build resolution maps
    for block in func.blocks:
        for instr in block.instrs:
            if instr.opcode == "uextend" and instr.type == "i64":
                # uextend.i64 vN -> maps to vN (the i32 value)
                if instr.operands:
                    extend_map[instr.dest] = instr.operands[0]
                    dead_vars.add(instr.dest)
            elif instr.opcode == "uextend" and instr.type == "i32":
                # uextend.i32 from icmp boolean result -> identity
                if instr.operands:
                    extend_map[instr.dest] = instr.operands[0]
                    dead_vars.add(instr.dest)
            elif instr.opcode == "load" and instr.type == "i64":
                # i64 loads from vmctx are infrastructure
                if instr.operands and instr.operands[0] in vmctx_vars:
                    dead_vars.add(instr.dest)
                    vmctx_vars.add(instr.dest)
            elif instr.opcode == "iconst" and instr.type == "i64":
                # i64 constants are intermediaries for address computation
                dead_vars.add(instr.dest)
                if instr.immediates:
                    i64_const_vals[instr.dest] = instr.immediates[0]
            elif instr.opcode == "iadd":
                # Check if this is an i64 address computation by examining operands
                ops = instr.operands
                if len(ops) == 2:
                    op0_is_i64 = ops[0] in dead_vars or ops[0] in vmctx_vars or ops[0] in extend_map or ops[0] in i64_const_vals
                    op1_is_i64 = ops[1] in dead_vars or ops[1] in vmctx_vars or ops[1] in extend_map or ops[1] in i64_const_vals
                    op0_is_ext = ops[0] in extend_map
                    op1_is_ext = ops[1] in extend_map
                    op0_is_dead = ops[0] in dead_vars or ops[0] in vmctx_vars
                    op1_is_dead = ops[1] in dead_vars or ops[1] in vmctx_vars

                    if op0_is_i64 and op1_is_i64:
                        # Both operands are i64 → address computation.
                        # Check addr+const FIRST (before dead+ext which would drop the const)
                        op0_resolved = extend_map.get(ops[0])
                        op1_resolved = extend_map.get(ops[1])
                        op0_const = i64_const_vals.get(ops[0])
                        op1_const = i64_const_vals.get(ops[1])

                        if op0_resolved is not None and op1_const is not None:
                            # iadd(resolved_addr, i64_const) → keep as i32 iadd
                            const_var = instr.dest + 50000
                            i32_const_for_i64[instr.dest] = (op0_resolved, op1_const, const_var)
                            continue
                        elif op1_resolved is not None and op0_const is not None:
                            const_var = instr.dest + 50000
                            i32_const_for_i64[instr.dest] = (op1_resolved, op0_const, const_var)
                            continue
                        elif op0_is_dead and op1_is_ext:
                            extend_map[instr.dest] = extend_map[ops[1]]
                        elif op1_is_dead and op0_is_ext:
                            extend_map[instr.dest] = extend_map[ops[0]]
                        elif op0_is_dead and op1_is_dead:
                            vmctx_vars.add(instr.dest)
                        elif op0_is_ext and op1_is_ext:
                            # iadd of two extended i32s — keep as real iadd
                            continue
                        dead_vars.add(instr.dest)
                        continue
                # Not an i64 address computation — regular iadd, don't mark dead

    # Resolve value aliases
    alias_map = dict(func.aliases)

    def resolve_var(v: int) -> int:
        """Resolve a v-number through extend_map and alias_map.

        Handles transitive chains: v30 (i64 addr) → v28 (uextend) → v2 (i32).
        """
        seen = set()
        while v in extend_map or v in alias_map:
            if v in seen:
                break
            seen.add(v)
            if v in extend_map:
                v = extend_map[v]
            elif v in alias_map:
                v = alias_map[v]
        return v

    # Second pass: simplify instructions
    simplified_blocks = []
    for block in func.blocks:
        new_instrs = []
        for instr in block.instrs:
            # Handle iadd that should become i32 iadd + iconst
            if instr.dest in i32_const_for_i64:
                base_v, const_val, const_var = i32_const_for_i64[instr.dest]
                # Emit: iconst const_var = const_val
                ci = CLIFInstr(opcode="iconst", dest=const_var, type="i32")
                ci.immediates = [const_val & MASK32]
                new_instrs.append(ci)
                # Emit: iadd dest = base_v + const_var
                ai = CLIFInstr(opcode="iadd", dest=instr.dest, type="i32")
                ai.operands = [base_v, const_var]
                new_instrs.append(ai)
                continue

            simplified = _simplify_instr(
                instr,
                resolve_var,
                dead_vars,
                vmctx_vars,
                heap_base_v,
                stack_ptr_v,
                output_fn_v,
                output_ctx_v,
            )
            if simplified is not None:
                new_instrs.extend(simplified if isinstance(simplified, list) else [simplified])

        new_block = CLIFBlock(
            id=block.id,
            params=[(resolve_var(v), t) for v, t in block.params if t == "i32"],
            instrs=new_instrs,
        )
        simplified_blocks.append(new_block)

    return simplified_blocks, extend_map, dead_vars


_STACK_PTR_ADDR = 4  # Fixed memory address for the stack pointer variable
_IMM_TEMP_VAR = 59999  # Shared temp variable for materialized ALU immediates (renumbered later)


def _simplify_instr(
    instr: CLIFInstr,
    resolve: callable,
    dead: set[int],
    vmctx: set[int],
    heap_base: int | None,
    stack_ptr: int | None,
    output_fn: int | None,
    output_ctx: int | None,
) -> CLIFInstr | list[CLIFInstr] | None:
    """Simplify a single instruction. Returns None to skip, or simplified instruction(s)."""

    # Skip dead variable definitions (i64 intermediaries, vmctx loads)
    if instr.dest is not None and instr.dest in dead:
        return None

    # Stack pointer operations: load/store with 'table' flag at vmctx+96
    # The stack pointer is a WASM global stored in vmctx. After vmctx
    # elimination, we redirect these to a fixed memory address.
    if instr.opcode == "load" and "table" in instr.flags and instr.operands and instr.operands[0] in vmctx:
            # load sp from fixed address: load dest, addr=0, offset=_STACK_PTR_ADDR
            # After flattening: SimpleInstr(load, dest=X, src1=None, imm=_STACK_PTR_ADDR)
            # Reference interpreter: addr = vars_[src1](=0) + offset = 0 + 4 = 4
            r = CLIFInstr(opcode="load_sp", dest=instr.dest, type="i32")
            r.offset = _STACK_PTR_ADDR
            return r

    if instr.opcode == "store" and "table" in instr.flags and instr.operands and instr.operands[-1] in vmctx:
            # store sp to fixed address
            val_v = resolve(instr.operands[0])
            r = CLIFInstr(opcode="store_sp", type="i32")
            r.operands = [val_v]
            r.offset = _STACK_PTR_ADDR
            return r

    op = instr.opcode

    # ── call_indirect → output ──
    if op == "call_indirect":
        # call_indirect sig0, v7(v6, v0, vN) → output vN
        # The last operand is the value being output
        if instr.operands:
            out_val = resolve(instr.operands[-1])
            r = CLIFInstr(opcode="output")
            r.operands = [out_val]
            return r
        return None

    # ── call fnN(...) → call with resolved args ──
    if op == "call":
        # Keep as call for now; inlining happens at a higher level
        r = CLIFInstr(opcode="call", fn_ref=instr.fn_ref)
        # Skip vmctx args (first two: v0, v0)
        r.operands = [resolve(v) for v in instr.operands[2:]]
        return r

    # ── Type conversion instructions ──
    if op == "uextend":
        # Should have been handled in first pass; if still here, it's identity
        if instr.operands:
            r = CLIFInstr(opcode="copy", dest=instr.dest)
            r.operands = [resolve(instr.operands[0])]
            return r
        return None

    if op == "sextend":
        # sextend.i32 from i8: sign-extend byte to i32
        if instr.type == "i32" and instr.operands:
            src = resolve(instr.operands[0])
            # Emit: dest = (src ^ 0x80) - 0x80 (sign extend from 8-bit)
            # Simplification: keep as sextend8 pseudo-op, reference interpreter handles it
            r = CLIFInstr(opcode="sextend8", dest=instr.dest)
            r.operands = [src]
            return r
        return None

    if op == "ireduce":
        # ireduce.i8: truncate to byte (mask with 0xFF)
        if instr.type == "i8" and instr.operands:
            src = resolve(instr.operands[0])
            r = CLIFInstr(opcode="band", dest=instr.dest)
            r.operands = [src]
            r.immediates = [0xFF]
            return r
        return None

    # ── Arithmetic ──
    if op in ("iadd", "isub", "imul", "band", "bor", "bxor", "ishl", "ushr", "sshr"):
        r = CLIFInstr(opcode=op, dest=instr.dest, type="i32")
        r.operands = [resolve(v) for v in instr.operands]
        r.immediates = list(instr.immediates)
        return r

    if op == "ineg":
        # ineg v → isub 0, v
        if instr.operands:
            r = CLIFInstr(opcode="ineg", dest=instr.dest, type="i32")
            r.operands = [resolve(instr.operands[0])]
            return r
        return None

    if op == "bnot":
        # bnot v → lower to: iconst tmp=0xFFFFFFFF, bxor dest=v^tmp
        # But bxor isn't in the ALM either. Express as: dest = -1 - v = ineg(v) - 1
        # Which is: isub 0xFFFFFFFF, v (since ~v = 0xFFFFFFFF - v for unsigned)
        if instr.operands:
            src = resolve(instr.operands[0])
            # Lower to two instructions: iconst tmp = 0xFFFFFFFF, isub dest = tmp - src
            tmp_v = instr.dest + 10000  # temporary variable (will be renumbered)
            r1 = CLIFInstr(opcode="iconst", dest=tmp_v, type="i32")
            r1.immediates = [MASK32]
            r2 = CLIFInstr(opcode="isub", dest=instr.dest, type="i32")
            r2.operands = [tmp_v, src]
            return [r1, r2]
        return None

    if op == "umulhi":
        # Upper half of unsigned multiply — keep as-is
        if len(instr.operands) >= 2:
            r = CLIFInstr(opcode="umulhi", dest=instr.dest, type="i32")
            r.operands = [resolve(v) for v in instr.operands[:2]]
            return r
        return None

    if op == "smin" or op == "smax":
        # smin/smax v1, v2 → keep as-is
        r = CLIFInstr(opcode=op, dest=instr.dest, type="i32")
        r.operands = [resolve(v) for v in instr.operands]
        r.immediates = list(instr.immediates)
        return r

    # ── Constants ──
    if op == "iconst":
        if instr.type == "i64":
            return None  # i64 constants are dead intermediaries
        r = CLIFInstr(opcode="iconst", dest=instr.dest, type="i32")
        r.immediates = [instr.immediates[0] & MASK32] if instr.immediates else [0]
        return r

    # ── Comparison ──
    if op == "icmp":
        r = CLIFInstr(opcode="icmp", dest=instr.dest, type="i32", cond=instr.cond)
        r.operands = [resolve(v) for v in instr.operands]
        r.immediates = list(instr.immediates)
        return r

    # ── Select ──
    if op == "select":
        # select cond, true_val, false_val
        if len(instr.operands) >= 3:
            r = CLIFInstr(opcode="select", dest=instr.dest, type="i32")
            r.operands = [resolve(v) for v in instr.operands[:3]]
            return r
        return None

    # ── Memory ──
    if op in ("load", "uload8", "sload8", "uload", "sload", "uload16", "sload16"):
        r = CLIFInstr(opcode=op, dest=instr.dest, type="i32")
        resolved_ops = [resolve(v) for v in instr.operands]
        r.operands = resolved_ops
        r.offset = instr.offset or 0
        r.flags = list(instr.flags)
        return r

    if op in ("store", "istore8", "store8", "istore16", "store16"):
        # Map store variants to canonical names
        op_name = "store8" if "8" in op else ("store16" if "16" in op else "store")
        r = CLIFInstr(opcode=op_name, type="i32")
        resolved_ops = [resolve(v) for v in instr.operands]
        r.operands = resolved_ops
        r.offset = instr.offset or 0
        r.flags = list(instr.flags)
        return r

    # ── Control flow ──
    if op == "brif":
        r = CLIFInstr(opcode="brif")
        r.operands = [resolve(v) for v in instr.operands]
        r.targets = [
            (bid, [resolve(a) for a in args]) for bid, args in instr.targets
        ]
        return r

    if op == "jump":
        r = CLIFInstr(opcode="jump")
        r.targets = [
            (bid, [resolve(a) for a in args]) for bid, args in instr.targets
        ]
        return r

    if op == "return":
        r = CLIFInstr(opcode="return")
        r.operands = [resolve(v) for v in instr.operands]
        return r

    # ── Stack operations (from Wasmtime's linear memory management) ──
    if op in ("get_stack_pointer", "get_frame_pointer", "get_exception_handler_address"):
        return None  # Infrastructure, not user code

    if op == "stack_load" or op == "stack_store" or op == "stack_addr":
        return None  # Stack slot operations handled via memory

    # Unknown instruction — keep it and warn
    logger.warning("Unknown CLIF instruction: %s (dest=v%s)", op, instr.dest)
    return None


# ── Flatten blocks to linear instruction list ──────────────────


def flatten_blocks(
    blocks: list[CLIFBlock],
    func_name: str = "",
) -> list[SimpleInstr]:
    """Convert blocks with branch targets to a flat instruction list with PC offsets.

    Inserts explicit 'copy' instructions for block parameter bindings before branches.
    Returns a list of SimpleInstr with resolved PC offsets.
    """
    # Phase 1: Compute the size of each block (including copy inserts)
    # We need to know the PC offset of each block to resolve branch targets.

    # First, compute how many copies each branch needs
    block_map = {b.id: b for b in blocks}

    # Build the linear layout: for each block, emit its instructions.
    # For branch/jump instructions, insert copies before the branch.
    # We need two passes: first compute sizes, then emit.

    # Pass 1: compute block sizes
    block_sizes = {}  # block_id -> number of SimpleInstrs
    for block in blocks:
        size = 0
        for instr in block.instrs:
            if instr.opcode == "brif":
                # brif expands to: copies + brif + jump (for false fallthrough)
                for _bid, args in instr.targets:
                    target_block = block_map.get(_bid)
                    if target_block and target_block.params and args:
                        size += min(len(args), len(target_block.params))
                size += 2  # brif + jump
            elif instr.opcode == "jump":
                for _bid, args in instr.targets:
                    target_block = block_map.get(_bid)
                    if target_block and target_block.params and args:
                        for (pv, _pt), av in zip(target_block.params, args, strict=False):
                            if pv != av:  # Match pass-2 identity-copy skip
                                size += 1
                size += 1
            elif instr.opcode in ("iadd", "isub", "imul", "band", "bor", "bxor",
                                   "ishl", "ushr", "sshr", "smin", "smax"):
                ops = instr.operands
                imms = instr.immediates
                s2 = ops[1] if len(ops) > 1 else None
                if s2 is None and imms:
                    size += 2  # iconst tmp + ALU reg-reg
                else:
                    size += 1
            else:
                size += 1
        block_sizes[block.id] = size

    # Compute block start offsets
    block_offsets = {}  # block_id -> PC offset
    offset = 0
    for block in blocks:
        block_offsets[block.id] = offset
        offset += block_sizes[block.id]

    # Pass 2: emit instructions
    result: list[SimpleInstr] = []
    current_pc = 0

    for block in blocks:

        for instr in block.instrs:
            if instr.opcode == "iconst":
                imm = instr.immediates[0] if instr.immediates else 0
                result.append(SimpleInstr(opcode="iconst", dest=instr.dest, imm=imm & MASK32))
                current_pc += 1

            elif instr.opcode in ("iadd", "isub", "imul", "band", "bor", "bxor", "ishl", "ushr", "sshr"):
                ops = instr.operands
                imms = instr.immediates
                s1 = ops[0] if len(ops) > 0 else None
                s2 = ops[1] if len(ops) > 1 else None
                imm = imms[0] if imms else 0
                if s2 is None and imms:
                    # Materialize immediate as iconst + reg-reg ALU
                    # Reuse a single temp var (consumed immediately, no conflict)
                    result.append(SimpleInstr(opcode="iconst", dest=_IMM_TEMP_VAR, imm=imm & MASK32))
                    result.append(SimpleInstr(opcode=instr.opcode, dest=instr.dest, src1=s1, src2=_IMM_TEMP_VAR))
                    current_pc += 2
                else:
                    result.append(SimpleInstr(opcode=instr.opcode, dest=instr.dest, src1=s1, src2=s2))
                    current_pc += 1

            elif instr.opcode == "ineg":
                result.append(SimpleInstr(opcode="ineg", dest=instr.dest, src1=instr.operands[0] if instr.operands else None))
                current_pc += 1

            elif instr.opcode == "umulhi":
                ops = instr.operands
                result.append(SimpleInstr(opcode="umulhi", dest=instr.dest, src1=ops[0] if ops else None, src2=ops[1] if len(ops) > 1 else None))
                current_pc += 1

            elif instr.opcode in ("smin", "smax"):
                ops = instr.operands
                imms = instr.immediates
                s1 = ops[0] if len(ops) > 0 else None
                s2 = ops[1] if len(ops) > 1 else None
                imm = imms[0] if imms else 0
                if s2 is None and imms:
                    # Materialize immediate as iconst + reg-reg
                    result.append(SimpleInstr(opcode="iconst", dest=_IMM_TEMP_VAR, imm=imm & MASK32))
                    result.append(SimpleInstr(opcode=instr.opcode, dest=instr.dest, src1=s1, src2=_IMM_TEMP_VAR))
                    current_pc += 2
                else:
                    result.append(SimpleInstr(opcode=instr.opcode, dest=instr.dest, src1=s1, src2=s2))
                    current_pc += 1

            elif instr.opcode == "icmp":
                ops = instr.operands
                cond_code = COND_CODES.get(instr.cond, 0)
                result.append(SimpleInstr(
                    opcode="icmp",
                    dest=instr.dest,
                    src1=ops[0] if ops else None,
                    src2=ops[1] if len(ops) > 1 else None,
                    cond=cond_code,
                ))
                current_pc += 1

            elif instr.opcode == "select":
                ops = instr.operands
                result.append(SimpleInstr(
                    opcode="select",
                    dest=instr.dest,
                    src1=ops[0] if ops else None,  # cond
                    src2=ops[1] if len(ops) > 1 else None,  # true_val
                    src3=ops[2] if len(ops) > 2 else None,  # false_val
                ))
                current_pc += 1

            elif instr.opcode in ("load", "uload8", "sload8", "uload", "sload", "uload16", "sload16"):
                ops = instr.operands
                addr = ops[-1] if ops else None
                off = instr.offset or 0
                op_name = instr.opcode
                if op_name in ("uload", "sload"):
                    op_name = "load"
                result.append(SimpleInstr(opcode=op_name, dest=instr.dest, src1=addr, imm=off))
                current_pc += 1

            elif instr.opcode in ("store", "store8", "store16"):
                ops = instr.operands
                off = instr.offset or 0
                val = ops[0] if ops else None
                addr = ops[1] if len(ops) > 1 else None
                result.append(SimpleInstr(opcode=instr.opcode, src1=val, src2=addr, imm=off))
                current_pc += 1

            elif instr.opcode == "load_sp":
                # Load from fixed stack pointer address
                result.append(SimpleInstr(opcode="load", dest=instr.dest, src1=None, imm=_STACK_PTR_ADDR))
                current_pc += 1

            elif instr.opcode == "store_sp":
                # Store to fixed stack pointer address
                val = instr.operands[0] if instr.operands else None
                result.append(SimpleInstr(opcode="store", src1=val, src2=None, imm=_STACK_PTR_ADDR))
                current_pc += 1

            elif instr.opcode == "copy":
                result.append(SimpleInstr(opcode="copy", dest=instr.dest, src1=instr.operands[0] if instr.operands else None))
                current_pc += 1

            elif instr.opcode == "sextend8":
                result.append(SimpleInstr(opcode="sextend8", dest=instr.dest, src1=instr.operands[0] if instr.operands else None))
                current_pc += 1

            elif instr.opcode == "output":
                result.append(SimpleInstr(opcode="output", src1=instr.operands[0] if instr.operands else None))
                current_pc += 1

            elif instr.opcode == "return":
                result.append(SimpleInstr(opcode="return"))
                current_pc += 1

            elif instr.opcode == "call":
                # For now, keep call as-is; will be handled during inlining
                result.append(SimpleInstr(opcode="call"))
                current_pc += 1

            elif instr.opcode == "brif":
                # Flatten brif into: copies + conditional branch + unconditional jump
                # This gives each branch a single 32-bit offset (like WASM br_if).
                #
                # Layout:
                #   copy_true  dest, src, cond   (for true branch params)
                #   copy_false dest, src, cond   (for false branch params)
                #   brif cond, +true_offset      (if cond: jump to true target)
                #   jump +false_offset            (else: jump to false target)
                #
                # If the false target is the next instruction after the jump,
                # the jump is just a fallthrough (offset=0) and could be omitted,
                # but we keep it for uniformity.

                cond_v = instr.operands[0] if instr.operands else 0
                true_bid, true_args = instr.targets[0] if instr.targets else (0, [])
                false_bid, false_args = instr.targets[1] if len(instr.targets) > 1 else (0, [])

                true_block = block_map.get(true_bid)
                false_block = block_map.get(false_bid)

                # Emit copies for true branch params (conditional on cond_v)
                if true_block and true_block.params and true_args:
                    for (param_v, _ptype), arg_v in zip(true_block.params, true_args, strict=False):
                        result.append(SimpleInstr(opcode="copy_true", dest=param_v, src1=arg_v, src2=cond_v))
                        current_pc += 1

                # Emit copies for false branch params (conditional on !cond_v)
                if false_block and false_block.params and false_args:
                    for (param_v, _ptype), arg_v in zip(false_block.params, false_args, strict=False):
                        result.append(SimpleInstr(opcode="copy_false", dest=param_v, src1=arg_v, src2=cond_v))
                        current_pc += 1

                # Emit brif with single 32-bit true offset
                # Need +2 because the jump instruction follows the brif
                true_offset = block_offsets.get(true_bid, 0) - (current_pc + 1)
                result.append(SimpleInstr(opcode="brif", src1=cond_v, imm=true_offset))
                current_pc += 1

                # Emit unconditional jump for false branch
                false_offset = block_offsets.get(false_bid, 0) - (current_pc + 1)
                result.append(SimpleInstr(opcode="jump", imm=false_offset))
                current_pc += 1

            elif instr.opcode == "jump":
                target_bid, target_args = instr.targets[0] if instr.targets else (0, [])
                target_block = block_map.get(target_bid)

                # Emit copies for target block params
                if target_block and target_block.params and target_args:
                    for (param_v, _ptype), arg_v in zip(target_block.params, target_args, strict=False):
                        if param_v != arg_v:  # Skip identity copies
                            result.append(SimpleInstr(opcode="copy", dest=param_v, src1=arg_v))
                            current_pc += 1

                # Emit the jump
                target_offset = block_offsets.get(target_bid, 0) - (current_pc + 1)
                result.append(SimpleInstr(opcode="jump", imm=target_offset))
                current_pc += 1

            else:
                logger.warning("Unhandled instruction in flatten: %s", instr.opcode)
                current_pc += 1

    return result


def _lower_complex_ops(instrs: list[SimpleInstr]) -> list[SimpleInstr]:
    """Lower imul, umulhi+ushr (division-by-constant) to loops using basic ops.

    Mirrors the WASM lowering in compilation/lower.py: imul becomes a repeated-
    addition loop, and the umulhi(x, magic)>>shift pattern (Cranelift's
    unsigned-division-by-constant idiom) becomes a repeated-subtraction loop.

    Operates on the flattened instruction list.  Internal loop branches use
    pre-computed PC-relative offsets; surviving original branches are remapped
    through an old→new PC table.
    """
    # Collect iconst values for constant propagation
    const_vals: dict[int, int] = {}
    for instr in instrs:
        if instr.opcode == "iconst" and instr.dest is not None:
            const_vals[instr.dest] = instr.imm

    # Find highest variable number in use
    max_var = max(
        (v for instr in instrs
         for v in (instr.dest, instr.src1, instr.src2, instr.src3)
         if v is not None),
        default=0,
    )

    # ── Pattern detection ────────────────────────────────────────
    skip: set[int] = set()          # PCs already claimed by a pattern
    plan: dict[int, tuple[set[int], list[SimpleInstr]]] = {}

    for i, instr in enumerate(instrs):
        if i in skip:
            continue

        # Pattern: umulhi dest, src, magic_const  ...  ushr q, dest, shift_const
        # → unsigned division loop  q = src / D
        if instr.opcode == "umulhi":
            magic = const_vals.get(instr.src2)
            if magic is None:
                continue
            for j in range(i + 1, min(i + 8, len(instrs))):
                if j in skip:
                    continue
                jj = instrs[j]
                if jj.opcode == "ushr" and jj.src1 == instr.dest:
                    shift = const_vals.get(jj.src2)
                    if shift is not None:
                        divisor = round((1 << (32 + shift)) / magic)
                        if divisor < 1:
                            divisor = 1
                        src = instr.src1
                        dest = jj.dest

                        max_var += 1; v_q = max_var
                        max_var += 1; v_a = max_var
                        max_var += 1; v_div = max_var
                        max_var += 1; v_done = max_var
                        max_var += 1; v_one = max_var

                        # Division loop: q=0; a=src; while a>=div: a-=div; q++
                        # Division loop: q=0; a=src; while a>=div: a-=div; q++; dest=q
                        # brif at +5 exits to +10: offset = 10-(5+1) = 4
                        # jump at +9 loops to +4: offset = 4-(9+1) = -6
                        loop = [
                            SimpleInstr(opcode="iconst", dest=v_q, imm=0),                       # +0
                            SimpleInstr(opcode="copy", dest=v_a, src1=src),                       # +1
                            SimpleInstr(opcode="iconst", dest=v_div, imm=divisor),                # +2
                            SimpleInstr(opcode="iconst", dest=v_one, imm=1),                      # +3
                            SimpleInstr(opcode="icmp", dest=v_done, src1=v_a, src2=v_div, cond=6),  # +4 ult
                            SimpleInstr(opcode="brif", src1=v_done, imm=4),                       # +5 → +10
                            SimpleInstr(opcode="jump", imm=0),                                    # +6 → +7
                            SimpleInstr(opcode="isub", dest=v_a, src1=v_a, src2=v_div),           # +7
                            SimpleInstr(opcode="iadd", dest=v_q, src1=v_q, src2=v_one),           # +8
                            SimpleInstr(opcode="jump", imm=-6),                                   # +9 → +4
                            SimpleInstr(opcode="copy", dest=dest, src1=v_q),                      # +10
                        ]
                        plan[i] = ({i, j}, loop)
                        skip.add(i)
                        skip.add(j)
                        break

        # Pattern: imul dest, src1, src2  → repeated-addition loop
        elif instr.opcode == "imul":
            dest = instr.dest
            src1 = instr.src1
            src2 = instr.src2

            max_var += 1; v_result = max_var
            max_var += 1; v_counter = max_var
            max_var += 1; v_zero = max_var
            max_var += 1; v_one = max_var
            max_var += 1; v_done = max_var

            # Multiply loop: result=0; counter=b; while counter!=0: result+=a; counter--
            # brif at +5 exits to +10: offset = 10-(5+1) = 4
            # jump at +9 loops to +4: offset = 4-(9+1) = -6
            loop = [
                SimpleInstr(opcode="iconst", dest=v_result, imm=0),                            # +0
                SimpleInstr(opcode="copy", dest=v_counter, src1=src2),                         # +1
                SimpleInstr(opcode="iconst", dest=v_zero, imm=0),                              # +2
                SimpleInstr(opcode="iconst", dest=v_one, imm=1),                               # +3
                SimpleInstr(opcode="icmp", dest=v_done, src1=v_counter, src2=v_zero, cond=0),  # +4 eq
                SimpleInstr(opcode="brif", src1=v_done, imm=4),                                # +5 → +10
                SimpleInstr(opcode="jump", imm=0),                                             # +6 → +7
                SimpleInstr(opcode="iadd", dest=v_result, src1=v_result, src2=src1),            # +7
                SimpleInstr(opcode="isub", dest=v_counter, src1=v_counter, src2=v_one),         # +8
                SimpleInstr(opcode="jump", imm=-6),                                            # +9 → +4
                SimpleInstr(opcode="copy", dest=dest, src1=v_result),                          # +10
            ]
            plan[i] = ({i}, loop)
            skip.add(i)

    if not plan:
        return instrs

    # ── Build new instruction list with old→new PC mapping ───────
    all_remove: set[int] = set()
    for pcs, _ in plan.values():
        all_remove.update(pcs)

    new_instrs: list[SimpleInstr] = []
    old_to_new: dict[int, int] = {}
    replacement_pcs: set[int] = set()        # new-PCs that belong to loops

    for old_pc in range(len(instrs)):
        old_to_new[old_pc] = len(new_instrs)
        if old_pc in plan:
            start = len(new_instrs)
            _, loop_instrs = plan[old_pc]
            new_instrs.extend(loop_instrs)
            replacement_pcs.update(range(start, len(new_instrs)))
            continue
        if old_pc in all_remove:
            continue
        new_instrs.append(instrs[old_pc])

    old_to_new[len(instrs)] = len(new_instrs)

    # ── Fix branch offsets for original (non-loop) instructions ──
    # Build reverse map: new_pc → old_pc for surviving original instrs
    new_to_old: dict[int, int] = {}
    for old_pc, new_pc in old_to_new.items():
        if old_pc < len(instrs) and old_pc not in all_remove:
            new_to_old[new_pc] = old_pc

    for new_pc, instr in enumerate(new_instrs):
        if new_pc in replacement_pcs:
            continue
        if instr.opcode in ("brif", "jump"):
            old_pc = new_to_old.get(new_pc)
            if old_pc is not None:
                old_target = old_pc + 1 + instr.imm
                new_target = old_to_new.get(old_target, old_to_new[len(instrs)])
                instr.imm = new_target - (new_pc + 1)

    logger.info("Lowered %d complex ops (imul/umulhi+ushr)", len(plan))
    return new_instrs


def _eliminate_dead_vars(instrs: list[SimpleInstr]) -> list[SimpleInstr]:
    """Remove instructions that write to variables never read by any other instruction.

    Iterates until no more dead writes are found. Only removes side-effect-free instructions.
    Branch offsets are recomputed after removal.
    """
    _SIDE_EFFECT_FREE = {
        "iconst", "iadd", "isub", "imul", "band", "bor", "bxor",
        "ishl", "ushr", "sshr", "icmp", "select", "copy", "copy_true",
        "copy_false", "ineg", "umulhi", "smin", "smax", "sextend8",
        "load", "uload8", "sload8", "uload16", "sload16",
    }

    changed = True
    while changed:
        changed = False
        # Collect all read variables
        read_vars: set[int] = set()
        for instr in instrs:
            for v in (instr.src1, instr.src2, instr.src3):
                if v is not None:
                    read_vars.add(v)

        # Find dead writes
        dead_indices: set[int] = set()
        for i, instr in enumerate(instrs):
            if (instr.dest is not None
                    and instr.dest not in read_vars
                    and instr.opcode in _SIDE_EFFECT_FREE):
                dead_indices.add(i)

        if not dead_indices:
            break
        changed = True

        # Build old-to-new PC mapping for branch fixup
        new_instrs: list[SimpleInstr] = []
        old_to_new: dict[int, int] = {}
        new_to_old: dict[int, int] = {}  # reverse mapping for surviving instrs
        new_pc = 0
        for old_pc, instr in enumerate(instrs):
            old_to_new[old_pc] = new_pc
            if old_pc not in dead_indices:
                new_to_old[new_pc] = old_pc
                new_instrs.append(instr)
                new_pc += 1
        old_to_new[len(instrs)] = new_pc  # sentinel

        # Fix branch offsets
        for new_i, instr in enumerate(new_instrs):
            if instr.opcode in ("brif", "jump"):
                old_pc = new_to_old[new_i]
                old_target = old_pc + 1 + instr.imm
                new_target = old_to_new.get(old_target, new_pc)
                instr.imm = new_target - (new_i + 1)

        instrs = new_instrs

    return instrs


def _renumber_vars(instrs: list[SimpleInstr]) -> tuple[list[SimpleInstr], int]:
    """Renumber v-variables sequentially starting from 0.

    Returns (renumbered instructions, max v-number used).
    """
    # Collect all v-numbers in use
    used = set()
    for instr in instrs:
        for v in (instr.dest, instr.src1, instr.src2, instr.src3):
            if v is not None:
                used.add(v)

    if not used:
        return instrs, 0

    # Build renumbering map
    sorted_vars = sorted(used)
    var_map = {old: new for new, old in enumerate(sorted_vars)}

    def remap(v):
        return var_map[v] if v is not None else None

    result = []
    for instr in instrs:
        new = SimpleInstr(
            opcode=instr.opcode,
            dest=remap(instr.dest),
            src1=remap(instr.src1),
            src2=remap(instr.src2),
            src3=remap(instr.src3),
            imm=instr.imm,
            cond=instr.cond,
            false_offset=instr.false_offset,
        )
        result.append(new)

    return result, len(sorted_vars) - 1


# ── Top-level: subset and flatten a set of functions ──────────


def _inline_calls(
    main_blocks: list[CLIFBlock],
    all_functions: dict[str, list[CLIFBlock]],
    fn_refs: dict[str, tuple[str, str | None]],
) -> list[CLIFBlock]:
    """Inline function calls by splicing callee blocks into the main function.

    Each call is replaced with:
    1. Copy instructions mapping call arguments to callee parameters
    2. A jump to the callee's entry block (remapped id)
    3. The callee's blocks appended to the function (remapped ids + v-numbers)
    4. Callee's return replaced with jump to a continuation block
    5. A continuation block where execution resumes after the call

    Variable offsets are reused per callee function: multiple calls to the
    same function share the same v-number remapping.  This keeps the total
    variable count low enough for the 1-byte encoding (max 256 variables).
    Block IDs remain unique per call site since all blocks coexist in the
    flattened function.
    """
    # Find max v-number and block id across all main blocks
    max_v = 0
    max_block = 0
    for block in main_blocks:
        max_block = max(max_block, block.id)
        for v, _ in block.params:
            max_v = max(max_v, v)
        for instr in block.instrs:
            if instr.dest is not None:
                max_v = max(max_v, instr.dest)
            for v in instr.operands:
                if v is not None:
                    max_v = max(max_v, v)

    output_blocks = []

    # All callee functions share a single v_offset since calls are sequential
    # and never overlap. Use the same variable space for all callees.
    shared_v_offset = max_v + 1
    func_v_offsets: dict[str, int] = {}
    for _fn_name, (func_id, _sig) in fn_refs.items():
        if func_id not in all_functions:
            continue
        func_v_offsets[func_id] = shared_v_offset

    for block in main_blocks:
        # Split block at each call site
        current_instrs = []
        current_block_id = block.id
        current_params = block.params

        for _instr_idx, instr in enumerate(block.instrs):
            if instr.opcode != "call" or instr.fn_ref is None:
                current_instrs.append(instr)
                continue

            fn_name = instr.fn_ref
            func_id, _sig = fn_refs.get(fn_name, (None, None))
            if func_id is None or func_id not in all_functions:
                current_instrs.append(instr)
                continue

            callee_blocks = all_functions[func_id]
            if not callee_blocks:
                current_instrs.append(instr)
                continue

            # Reuse the pre-allocated v_offset for this callee function;
            # allocate a fresh block_offset per call site.
            v_offset = func_v_offsets[func_id]
            block_offset = max_block + 1
            max_callee_block = max(cb.id for cb in callee_blocks)
            cont_block_id = block_offset + max_callee_block + 1

            # Remap callee v-numbers
            def _remap_v(v, off=v_offset):
                return v + off if v is not None else None

            # Copy call arguments to callee entry block parameters
            callee_entry = callee_blocks[0]
            i32_params = [(v, t) for v, t in callee_entry.params if t == "i32"]
            call_args = instr.operands

            for (param_v, _), arg_v in zip(i32_params, call_args, strict=False):
                ci = CLIFInstr(opcode="copy", dest=param_v + v_offset)
                ci.operands = [arg_v]
                current_instrs.append(ci)

            # Jump to callee entry
            ji = CLIFInstr(opcode="jump")
            ji.targets = [(callee_entry.id + block_offset, [])]
            current_instrs.append(ji)

            # Emit the current block up to this point
            output_blocks.append(CLIFBlock(
                id=current_block_id, params=current_params, instrs=current_instrs
            ))

            # Emit remapped callee blocks
            for cb in callee_blocks:
                new_instrs = []
                for ci in cb.instrs:
                    if ci.opcode == "return":
                        ri = CLIFInstr(opcode="jump")
                        ri.targets = [(cont_block_id, [])]
                        new_instrs.append(ri)
                    else:
                        ni = CLIFInstr(
                            opcode=ci.opcode,
                            dest=_remap_v(ci.dest),
                            type=ci.type,
                            operands=[_remap_v(v) for v in ci.operands],
                            immediates=list(ci.immediates),
                            cond=ci.cond,
                            targets=[
                                (bid + block_offset, [_remap_v(a) for a in args])
                                for bid, args in ci.targets
                            ],
                            flags=list(ci.flags),
                            sig_ref=ci.sig_ref,
                            fn_ref=ci.fn_ref,
                            offset=ci.offset,
                        )
                        new_instrs.append(ni)
                output_blocks.append(CLIFBlock(
                    id=cb.id + block_offset,
                    params=[(v + v_offset, t) for v, t in cb.params],
                    instrs=new_instrs,
                ))

            # Only advance max_block (block IDs must be unique per call site)
            for cb in callee_blocks:
                max_block = max(max_block, cb.id + block_offset)
            max_block = cont_block_id

            # Start a new continuation block for remaining instructions
            current_block_id = cont_block_id
            current_params = []
            current_instrs = []

        # Emit the final segment of this block
        output_blocks.append(CLIFBlock(
            id=current_block_id, params=current_params, instrs=current_instrs
        ))

    return output_blocks


def _renumber_block_vars(blocks: list[CLIFBlock]) -> list[CLIFBlock]:
    """Renumber variables in a list of CLIFBlocks to be contiguous starting from 0."""
    # Collect all variable numbers
    used: set[int] = set()
    for block in blocks:
        for v, _ in block.params:
            used.add(v)
        for instr in block.instrs:
            if instr.dest is not None:
                used.add(instr.dest)
            for v in instr.operands:
                if v is not None:
                    used.add(v)
    if not used:
        return blocks

    sorted_vars = sorted(used)
    var_map = {old: new for new, old in enumerate(sorted_vars)}

    def remap(v):
        return var_map[v] if v is not None else None

    result = []
    for block in blocks:
        new_params = [(var_map.get(v, v), t) for v, t in block.params]
        new_instrs = []
        for instr in block.instrs:
            ni = CLIFInstr(
                opcode=instr.opcode,
                dest=remap(instr.dest),
                type=instr.type,
                operands=[remap(v) for v in instr.operands],
                immediates=list(instr.immediates),
                cond=instr.cond,
                targets=[
                    (bid, [remap(a) for a in args])
                    for bid, args in instr.targets
                ],
                flags=list(instr.flags),
                sig_ref=instr.sig_ref,
                fn_ref=instr.fn_ref,
                offset=instr.offset,
            )
            new_instrs.append(ni)
        result.append(CLIFBlock(id=block.id, params=new_params, instrs=new_instrs))
    return result


def subset_and_flatten(
    functions: list[CLIFFunction],
    input_base: int = 0,
    stack_pointer_init: int = 0,
    data_segments: list | None = None,
) -> SimpleProg:
    """Subset and flatten a CLIF program (possibly multiple functions) into SimpleProg.

    Inlines helper functions (sscanf, printf) into the main compute function.
    """
    # Find the compute function (first function = u0:0)
    compute_func = None
    func_map = {}  # func_id -> CLIFFunction
    for func in functions:
        func_map[func.func_id] = func
        if func.func_id == "u0:0" or func.func_id == "%compute":
            compute_func = func

    if compute_func is None and functions:
        compute_func = functions[0]

    if compute_func is None:
        return SimpleProg()

    # Subset ALL functions and renumber callee variables to minimize v_offset space
    all_subsetted = {}
    for func_id, func in func_map.items():
        blocks, _, _ = _subset_function(func)
        # Renumber callee functions' variables to be contiguous starting from 0
        if func_id != (compute_func.func_id if compute_func else None):
            blocks = _renumber_block_vars(blocks)
        all_subsetted[func_id] = blocks

    # Inline calls in the compute function
    compute_blocks = all_subsetted.get(compute_func.func_id, [])
    if compute_func.fn_refs:
        compute_blocks = _inline_calls(
            compute_blocks, all_subsetted, compute_func.fn_refs
        )

    # Find the i32 parameter v-number (the input pointer)
    input_param_v = None
    for i, (ptype, _pname) in enumerate(compute_func.params):
        if ptype == "i32":
            input_param_v = i
            break

    # Flatten to linear instruction list
    flat_instrs = flatten_blocks(compute_blocks, func_name=compute_func.name)

    # Lower complex ops (imul, umulhi+ushr) to loops before further processing
    flat_instrs = _lower_complex_ops(flat_instrs)

    # Prepend initialization for function parameters that are used but not defined
    # The i32 parameter (input pointer) needs to be initialized with input_base
    if input_param_v is not None and input_base > 0:
        # Check if this v-number is referenced by any instruction
        used_vars = set()
        for instr in flat_instrs:
            for v in (instr.src1, instr.src2, instr.src3):
                if v is not None:
                    used_vars.add(v)
        if input_param_v in used_vars:
            # Insert iconst at the start to initialize the param
            init_instr = SimpleInstr(
                opcode="iconst", dest=input_param_v, imm=input_base & MASK32
            )
            flat_instrs.insert(0, init_instr)
            # Fix all branch offsets to account for the inserted instruction
            for instr in flat_instrs[1:]:
                if instr.opcode in ("brif", "jump"):
                    if instr.imm < 0:
                        instr.imm -= 1  # Backward branches go one further
                    if instr.opcode == "brif" and instr.false_offset < 0:
                        instr.false_offset -= 1

    # Initialize stack pointer memory if the program uses it
    has_sp_ops = any(
        (instr.src1 is None and instr.imm == _STACK_PTR_ADDR)
        or (instr.src2 is None and instr.imm == _STACK_PTR_ADDR)
        for instr in flat_instrs
    )
    if stack_pointer_init > 0 and has_sp_ops:
        # Find free v-numbers
        all_vars = set()
        for instr in flat_instrs:
            for v in (instr.dest, instr.src1, instr.src2, instr.src3):
                if v is not None:
                    all_vars.add(v)
        sp_val_v = max(all_vars) + 1 if all_vars else 0
        sp_addr_v = sp_val_v + 1

        # Patch load_sp/store_sp instructions to use sp_addr_v as the address
        for instr in flat_instrs:
            if instr.opcode == "load" and instr.src1 is None and instr.imm == _STACK_PTR_ADDR:
                instr.src1 = sp_addr_v
                instr.imm = 0
            elif instr.opcode == "store" and instr.src2 is None and instr.imm == _STACK_PTR_ADDR:
                instr.src2 = sp_addr_v
                instr.imm = 0

        # Prepend SP initialization (offsets are relative, no fixup needed)
        sp_init = [
            SimpleInstr(opcode="iconst", dest=sp_val_v, imm=stack_pointer_init & MASK32),
            SimpleInstr(opcode="iconst", dest=sp_addr_v, imm=_STACK_PTR_ADDR),
            SimpleInstr(opcode="store", src1=sp_val_v, src2=sp_addr_v),
        ]
        flat_instrs = sp_init + flat_instrs

    # Add halt at the end if the function doesn't end with return
    if not flat_instrs or flat_instrs[-1].opcode != "return":
        flat_instrs.append(SimpleInstr(opcode="halt"))

    # Replace final 'return' with 'halt' (top-level function)
    for i, instr in enumerate(flat_instrs):
        if instr.opcode == "return":
            flat_instrs[i] = SimpleInstr(opcode="halt")

    # Eliminate dead variables to reduce variable count
    flat_instrs = _eliminate_dead_vars(flat_instrs)

    # Renumber variables
    flat_instrs, max_var = _renumber_vars(flat_instrs)

    return SimpleProg(
        instrs=flat_instrs, input_base=input_base, max_var=max_var,
        data_segments=data_segments or [],
    )


def dump_simple_prog(prog: SimpleProg) -> str:
    """Format a SimpleProg as human-readable text."""
    lines = []
    for i, instr in enumerate(prog.instrs):
        parts = [f"{i:4d}: {instr.opcode:12s}"]
        if instr.dest is not None:
            parts.append(f"v{instr.dest} =")
        if instr.src1 is not None:
            parts.append(f"v{instr.src1}")
        if instr.src2 is not None:
            parts.append(f"v{instr.src2}")
        if instr.src3 is not None:
            parts.append(f"v{instr.src3}")
        if instr.imm != 0 or instr.opcode == "iconst":
            parts.append(f"imm={instr.imm}")
        if instr.cond != 0 or instr.opcode == "icmp":
            cond_name = {v: k for k, v in COND_CODES.items()}.get(instr.cond, "?")
            parts.append(f"cond={cond_name}")
        if instr.false_offset != 0:
            parts.append(f"false_off={instr.false_offset}")
        lines.append(" ".join(parts))
    return "\n".join(lines)
