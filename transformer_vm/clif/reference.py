#!/usr/bin/env python3
"""Generate reference token traces by executing simplified CLIF programs directly.

This is the CLIF equivalent of wasm/reference.py. It executes the simplified
CLIF instruction set and generates execution traces that the graph evaluator
and transformer model should reproduce exactly.

Usage:
    python -m transformer_vm.clif.reference                      # all from manifest
    python -m transformer_vm.clif.reference data/hello_clif.txt  # single program
"""

from __future__ import annotations

import argparse
import logging
import os
import re

from transformer_vm.clif.subset import COND_CODES, MASK32

logger = logging.getLogger(__name__)

# Reverse condition code map
COND_NAMES = {v: k for k, v in COND_CODES.items()}


def _to_signed(v):
    return v - (1 << 32) if v >= (1 << 31) else v


def _add_carries(a, b):
    carries = []
    carry = 0
    for i in range(4):
        s = ((a >> (8 * i)) & 0xFF) + ((b >> (8 * i)) & 0xFF) + carry
        carry = 1 if s >= 256 else 0
        carries.append(carry)
    return carries


def _sub_borrows(a, b):
    borrows = []
    borrow = 0
    for i in range(4):
        s = ((a >> (8 * i)) & 0xFF) - ((b >> (8 * i)) & 0xFF) - borrow
        borrow = 1 if s < 0 else 0
        borrows.append(borrow)
    return borrows


def _byte_tokens(value, num_bytes, carries=None):
    tokens = []
    for i in range(num_bytes):
        bv = (value >> (8 * i)) & 0xFF
        c = carries[i] if carries else 0
        tokens.append(f"{bv:02x}'" if c else f"{bv:02x}")
    return tokens


def _out_token(bv):
    bv &= 0xFF
    if 0x20 < bv < 0x7F:
        return f"out({chr(bv)})"
    return f"out({bv:02x})"


def _commit(delta_cursor=1, var_write=0, bt=0):
    """Generate a commit token for CLIF execution traces."""
    return f"commit({delta_cursor:+d},vw={var_write},bt={bt})"


# ── Program loading ──────────────────────────────────────────


def load_clif_program(path):
    """Parse a CLIF .txt program file into instruction list and input string."""
    with open(path) as f:
        tokens = f.read().split()

    assert tokens[0] == "{", f"Expected '{{' at start of {path}"
    try:
        end = len(tokens) - 1 - tokens[::-1].index("}")
    except ValueError:
        raise ValueError(f"No closing '}}' found in {path}") from None

    body = tokens[1:end]
    program = []
    i = 0
    while i < len(body):
        op = body[i]
        data = [int(body[i + 1 + j], 16) for j in range(6)]
        program.append((op, data))
        i += 7

    # Extract input string
    input_tokens = tokens[end + 1:]
    if input_tokens and input_tokens[-1].startswith("commit("):
        input_tokens = input_tokens[:-1]
    chars = []
    for tok in input_tokens:
        if len(tok) == 1:
            chars.append(tok)
        elif len(tok) == 2:
            b = int(tok, 16)
            if b == 0:
                break
            chars.append(chr(b))
        else:
            break

    return program, "".join(chars)


def _decode_instr(op, data):
    """Decode a (opcode, 6-byte data) tuple into structured fields."""
    d = {}
    d["opcode"] = op

    if op == "iconst":
        d["dest"] = data[0]
        d["imm"] = data[1] | (data[2] << 8) | (data[3] << 16) | (data[4] << 24)
    elif op in ("iadd", "isub", "imul", "band", "bor", "bxor", "ishl", "ushr", "sshr"):
        d["dest"] = data[0]
        d["src1"] = data[1]
        if data[2] < 128 and data[3] == 0 and data[4] == 0 and data[5] == 0:
            # Could be src2 (var ref) or immediate — check context
            # If src2 looks like a var ref (small number), treat as var
            d["src2"] = data[2]
        else:
            d["imm"] = data[2] | (data[3] << 8) | (data[4] << 16) | (data[5] << 24)
    elif op == "icmp":
        d["dest"] = data[0]
        d["src1"] = data[1]
        d["src2"] = data[2]
        d["cond"] = data[3]
    elif op == "select":
        d["dest"] = data[0]
        d["cond"] = data[1]
        d["src_true"] = data[2]
        d["src_false"] = data[3]
    elif op in ("load", "uload8", "sload8", "uload16", "sload16"):
        d["dest"] = data[0]
        d["addr"] = data[1]
        d["offset"] = data[2] | (data[3] << 8)
    elif op in ("store", "store8", "store16"):
        # f0=0, f1=val, f2=addr, f3:f4=offset
        d["val"] = data[1]
        d["addr"] = data[2]
        d["offset"] = data[3] | (data[4] << 8)
    elif op == "brif":
        # f0=0, f1=cond, f2:f3=true_offset, f4:f5=false_offset
        d["cond_var"] = data[1]
        t = data[2] | (data[3] << 8)
        if t >= 0x8000:
            t -= 0x10000
        d["true_offset"] = t
        f = data[4] | (data[5] << 8)
        if f >= 0x8000:
            f -= 0x10000
        d["false_offset"] = f
    elif op == "jump":
        t = data[0] | (data[1] << 8)
        if t >= 0x8000:
            t -= 0x10000
        d["offset"] = t
    elif op in ("copy", "copy_true", "copy_false"):
        d["dest"] = data[0]
        d["src"] = data[1]
        d["cond_var"] = data[2]
    elif op == "output":
        # f0=0, f1=src
        d["src"] = data[1]
    elif op in ("ineg", "sextend8"):
        d["dest"] = data[0]
        d["src"] = data[1]
    elif op == "umulhi":
        d["dest"] = data[0]
        d["src1"] = data[1]
        d["src2"] = data[2]
    elif op in ("smin", "smax"):
        d["dest"] = data[0]
        d["src1"] = data[1]
        if data[3] == 0 and data[4] == 0 and data[5] == 0:
            d["src2"] = data[2]
        else:
            d["imm"] = data[2] | (data[3] << 8) | (data[4] << 16) | (data[5] << 24)
    elif op == "input_base":
        d["imm"] = data[0] | (data[1] << 8) | (data[2] << 16) | (data[3] << 24)

    return d


# ── CLIF interpreter ─────────────────────────────────────────


def run(program, input_str="", max_tokens=1_000_000, trace=False):
    """Execute a compiled CLIF program.

    Returns (instr_count, token_count, output_str) or, if trace=True,
    (instr_count, token_count, output_str, trace_tokens).
    """
    mem = bytearray(10 * 1024 * 1024)
    vars_ = {}  # v-number -> i32 value

    # Decode all instructions
    decoded = [_decode_instr(op, data) for op, data in program]

    # Find input_base
    input_base = None
    pc_start = 0
    if decoded and decoded[0]["opcode"] == "input_base":
        input_base = decoded[0].get("imm", 0)
        pc_start = 1

    if input_base is not None and input_str:
        for i, ch in enumerate(input_str.encode("utf-8") + b"\x00"):
            mem[input_base + i] = ch

    pc = pc_start
    instr_count = 0
    token_count = 0
    output = []
    trace_tokens = [] if trace else None

    # Emit input tokens
    if input_base is not None and input_str:
        input_bytes = input_str.encode("utf-8") + b"\x00"
        token_count += len(input_bytes) + 1
        if trace:
            for ch_byte in input_bytes:
                if 0x20 < ch_byte < 0x7F:
                    trace_tokens.append(chr(ch_byte))
                else:
                    trace_tokens.append(f"{ch_byte:02x}")
            trace_tokens.append(_commit(0, 0, 0))

    while pc < len(decoded) and token_count < max_tokens:
        d = decoded[pc]
        op = d["opcode"]
        instr_count += 1

        if op == "halt":
            token_count += 1
            if trace:
                trace_tokens.append("halt")
            break

        elif op == "iconst":
            result = d["imm"] & MASK32
            vars_[d["dest"]] = result
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(1, 1, 0))
            pc += 1

        elif op in ("iadd", "isub"):
            a = vars_.get(d["src1"], 0) & MASK32
            if "src2" in d:
                b = vars_.get(d["src2"], 0) & MASK32
            else:
                b = d.get("imm", 0) & MASK32

            if op == "iadd":
                result = (a + b) & MASK32
                carries = _add_carries(a, b)
            else:
                result = (a - b) & MASK32
                carries = _sub_borrows(a, b)

            vars_[d["dest"]] = result
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4, carries))
                trace_tokens.append(_commit(1, 1, 0))
            pc += 1

        elif op == "imul":
            a = vars_.get(d["src1"], 0) & MASK32
            b = vars_.get(d.get("src2", d["src1"]), 0) & MASK32 if "src2" in d else (d.get("imm", 0) & MASK32)
            result = (a * b) & MASK32
            vars_[d["dest"]] = result
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(1, 1, 0))
            pc += 1

        elif op in ("band", "bor", "bxor"):
            a = vars_.get(d["src1"], 0) & MASK32
            if "src2" in d:
                b = vars_.get(d["src2"], 0) & MASK32
            else:
                b = d.get("imm", 0) & MASK32
            if op == "band":
                result = a & b
            elif op == "bor":
                result = a | b
            else:
                result = a ^ b
            vars_[d["dest"]] = result & MASK32
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result & MASK32, 4))
                trace_tokens.append(_commit(1, 1, 0))
            pc += 1

        elif op in ("ishl", "ushr", "sshr"):
            a = vars_.get(d["src1"], 0) & MASK32
            b = vars_.get(d.get("src2", d["src1"]), 0) & MASK32 if "src2" in d else (d.get("imm", 0) & MASK32)
            shift = b & 31
            if op == "ishl":
                result = (a << shift) & MASK32
            elif op == "ushr":
                result = a >> shift
            else:
                result = (_to_signed(a) >> shift) & MASK32
            vars_[d["dest"]] = result
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(1, 1, 0))
            pc += 1

        elif op == "ineg":
            a = vars_.get(d["src"], 0) & MASK32
            result = (-_to_signed(a)) & MASK32
            vars_[d["dest"]] = result
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(1, 1, 0))
            pc += 1

        elif op == "umulhi":
            a = vars_.get(d["src1"], 0) & MASK32
            b = vars_.get(d["src2"], 0) & MASK32
            result = ((a * b) >> 32) & MASK32
            vars_[d["dest"]] = result
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(1, 1, 0))
            pc += 1

        elif op in ("smin", "smax"):
            a = _to_signed(vars_.get(d["src1"], 0) & MASK32)
            if "src2" in d:
                b = _to_signed(vars_.get(d["src2"], 0) & MASK32)
            else:
                b = _to_signed(d.get("imm", 0) & MASK32)
            result = (min(a, b) if op == "smin" else max(a, b)) & MASK32
            vars_[d["dest"]] = result
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(1, 1, 0))
            pc += 1

        elif op == "sextend8":
            a = vars_.get(d["src"], 0) & 0xFF
            result = (a - 256 if a >= 128 else a) & MASK32
            vars_[d["dest"]] = result
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(1, 1, 0))
            pc += 1

        elif op == "icmp":
            a = vars_.get(d["src1"], 0) & MASK32
            b = vars_.get(d["src2"], 0) & MASK32
            cond = d.get("cond", 0)
            cond_name = COND_NAMES.get(cond, "eq")

            if cond_name == "eq":
                result = 1 if a == b else 0
            elif cond_name == "ne":
                result = 1 if a != b else 0
            elif cond_name == "slt":
                result = 1 if _to_signed(a) < _to_signed(b) else 0
            elif cond_name == "sgt":
                result = 1 if _to_signed(a) > _to_signed(b) else 0
            elif cond_name == "sle":
                result = 1 if _to_signed(a) <= _to_signed(b) else 0
            elif cond_name == "sge":
                result = 1 if _to_signed(a) >= _to_signed(b) else 0
            elif cond_name == "ult":
                result = 1 if a < b else 0
            elif cond_name == "ugt":
                result = 1 if a > b else 0
            elif cond_name == "ule":
                result = 1 if a <= b else 0
            elif cond_name == "uge":
                result = 1 if a >= b else 0
            else:
                result = 0

            vars_[d["dest"]] = result
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(1, 1, 0))
            pc += 1

        elif op == "select":
            c = vars_.get(d["cond"], 0)
            t = vars_.get(d["src_true"], 0) & MASK32
            f = vars_.get(d["src_false"], 0) & MASK32
            result = t if c != 0 else f
            vars_[d["dest"]] = result
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(1, 1, 0))
            pc += 1

        elif op in ("load", "uload8", "sload8", "uload16", "sload16"):
            addr = (vars_.get(d["addr"], 0) + d.get("offset", 0)) & MASK32
            if op == "load":
                result = mem[addr] | (mem[addr + 1] << 8) | (mem[addr + 2] << 16) | (mem[addr + 3] << 24)
            elif op == "uload8":
                result = mem[addr]
            elif op == "sload8":
                v = mem[addr]
                result = (v - 256) & MASK32 if v >= 128 else v
            elif op == "uload16":
                result = mem[addr] | (mem[addr + 1] << 8)
            elif op == "sload16":
                v = mem[addr] | (mem[addr + 1] << 8)
                result = (v - 65536) & MASK32 if v >= 32768 else v
            vars_[d["dest"]] = result & MASK32
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result & MASK32, 4))
                trace_tokens.append(_commit(1, 1, 0))
            pc += 1

        elif op in ("store", "store8", "store16"):
            val = vars_.get(d["val"], 0) & MASK32
            addr = (vars_.get(d["addr"], 0) + d.get("offset", 0)) & MASK32
            if op == "store":
                mem[addr] = val & 0xFF
                mem[addr + 1] = (val >> 8) & 0xFF
                mem[addr + 2] = (val >> 16) & 0xFF
                mem[addr + 3] = (val >> 24) & 0xFF
                token_count += 5
                if trace:
                    trace_tokens.extend(_byte_tokens(val, 4))
                    trace_tokens.append(_commit(1, 0, 0))
            elif op == "store16":
                mem[addr] = val & 0xFF
                mem[addr + 1] = (val >> 8) & 0xFF
                token_count += 3
                if trace:
                    trace_tokens.extend(_byte_tokens(val, 2))
                    trace_tokens.append(_commit(1, 0, 0))
            else:
                mem[addr] = val & 0xFF
                token_count += 2
                if trace:
                    trace_tokens.extend(_byte_tokens(val, 1))
                    trace_tokens.append(_commit(1, 0, 0))
            pc += 1

        elif op == "copy":
            result = vars_.get(d["src"], 0) & MASK32
            vars_[d["dest"]] = result
            token_count += 5
            if trace:
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(1, 1, 0))
            pc += 1

        elif op == "copy_true":
            cond_val = vars_.get(d.get("cond_var", 0), 0)
            if cond_val != 0:
                result = vars_.get(d["src"], 0) & MASK32
                vars_[d["dest"]] = result
            token_count += 5
            if trace:
                result = vars_.get(d["dest"], 0) & MASK32
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(1, 1, 0))
            pc += 1

        elif op == "copy_false":
            cond_val = vars_.get(d.get("cond_var", 0), 0)
            if cond_val == 0:
                result = vars_.get(d["src"], 0) & MASK32
                vars_[d["dest"]] = result
            token_count += 5
            if trace:
                result = vars_.get(d["dest"], 0) & MASK32
                trace_tokens.extend(_byte_tokens(result, 4))
                trace_tokens.append(_commit(1, 1, 0))
            pc += 1

        elif op == "brif":
            cond = vars_.get(d["cond_var"], 0)
            if cond != 0:
                offset = d["true_offset"]
            else:
                offset = d["false_offset"]
            token_count += 6
            if trace:
                trace_tokens.append("branch_taken")
                # Emit the actual offset as 4 bytes
                off_u32 = offset & MASK32
                trace_tokens.extend(_byte_tokens(off_u32, 4))
                trace_tokens.append(_commit(1, 0, 1))
            pc = pc + 1 + offset

        elif op == "jump":
            offset = d["offset"]
            token_count += 6
            if trace:
                trace_tokens.append("branch_taken")
                off_u32 = offset & MASK32
                trace_tokens.extend(_byte_tokens(off_u32, 4))
                trace_tokens.append(_commit(1, 0, 1))
            pc = pc + 1 + offset

        elif op == "output":
            val = vars_.get(d["src"], 0) & 0xFF
            output.append(chr(val))
            token_count += 1
            if trace:
                trace_tokens.append(_out_token(val))
            pc += 1

        elif op == "call":
            # Calls not yet supported in reference interpreter
            logger.warning("Unsupported call at pc=%d, skipping", pc)
            pc += 1

        elif op == "input_base":
            # Already handled during setup
            pc += 1

        else:
            raise RuntimeError(f"Unknown op: {op} at pc={pc}")

    result = (instr_count, token_count, "".join(output))
    if trace:
        return result + (trace_tokens,)
    return result


# ── Trace formatting ─────────────────────────────────────────


def format_trace(program_path, trace_tokens):
    """Format program prefix + trace tokens into reference file format."""
    with open(program_path) as f:
        raw = f.read().split()
    assert raw[0] == "{"
    end = len(raw) - 1 - raw[::-1].index("}")

    lines = ["{"]
    for i in range(1, end):
        group_idx = (i - 1) % 7  # 7 tokens per instruction
        if group_idx == 0:
            current = [raw[i]]
        else:
            current.append(raw[i])
        if group_idx == 6:
            lines.append(" ".join(current))
    lines.append("}")

    current = []
    for tok in trace_tokens:
        is_terminal = (
            tok.startswith("commit(")
            or tok.startswith("out(")
            or tok == "halt"
            or tok == "branch_taken"
        )
        current.append(tok)
        if is_terminal:
            lines.append(" ".join(current))
            current = []

    if current:
        lines.append(" ".join(current))
    return "\n".join(lines) + "\n"


# ── Reference file generation ────────────────────────────────


def generate_ref(prog_path, ref_path=None, max_tokens=100_000_000):
    """Run a CLIF program and write the reference trace."""
    if ref_path is None:
        ref_path = prog_path.replace(".txt", "_ref.txt")
    program, input_str = load_clif_program(prog_path)
    _instrs, token_count, output, trace_tokens = run(
        program, input_str, max_tokens=max_tokens, trace=True
    )
    formatted = format_trace(prog_path, trace_tokens)
    with open(ref_path, "w") as f:
        f.write(formatted)
    logger.info("%s: %d tokens, output=%r", ref_path, token_count, output)


def generate_all(regen=False):
    """Generate _ref.txt for all CLIF program .txt files in data/."""
    import glob as globmod

    from transformer_vm._paths import DATA_DIR

    prog_files = sorted(globmod.glob(os.path.join(DATA_DIR, "*_clif.txt")))
    prog_files = [f for f in prog_files if not (f.endswith("_ref.txt") or f.endswith("_spec.txt"))]

    for prog_path in prog_files:
        ref_path = prog_path.replace("_clif.txt", "_clif_ref.txt")
        if not regen and os.path.exists(ref_path):
            logger.info("Skipping %s: already exists", ref_path)
            continue
        logger.info("Generating %s ...", ref_path)
        generate_ref(prog_path, ref_path)

    logger.info("Done.")


# ── CLI ──────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Generate reference token traces from CLIF execution."
    )
    parser.add_argument("files", nargs="*", help="CLIF program .txt files (default: all from data/)")
    parser.add_argument(
        "--regen", action="store_true", help="Regenerate even if _ref.txt already exists"
    )
    args = parser.parse_args()

    if not args.files:
        generate_all(regen=args.regen)
        return

    for prog_path in args.files:
        generate_ref(prog_path)


if __name__ == "__main__":
    main()
