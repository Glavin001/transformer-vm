#!/usr/bin/env python3
"""Compile C programs to CLIF token-prefix format via WASM → Cranelift.

Pipeline: C → WASM binary (clang) → CLIF (wasmtime --emit-clif) → subset → tokenize

Usage:
    python -m transformer_vm.compilation.compile_clif examples/hello.c --args World
    python -m transformer_vm.compilation.compile_clif examples/collatz.c --args 7

Outputs:
    <name>_clif.txt     — full token prefix { program } input (universal model input)
    <name>_clif_spec.txt — start + input tokens (specialized model input)
"""

from __future__ import annotations

import argparse
import glob as globmod
import logging
import os
import shutil
import subprocess
import tempfile

from transformer_vm._paths import DATA_DIR, EXAMPLES_DIR, MANIFEST
from transformer_vm.clif.parser import parse_clif_file
from transformer_vm.clif.subset import COND_CODES, SimpleInstr, SimpleProg, subset_and_flatten

logger = logging.getLogger(__name__)

MASK32 = 0xFFFFFFFF

# CLIF opcode names for the token vocabulary
CLIF_OPCODES = [
    "halt",
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
    "store",
    "store8",
    "store16",
    "uload16",
    "sload16",
    "brif",
    "jump",
    "return",
    "call",
    "output",
    "copy",
    "copy_true",
    "copy_false",
    "ineg",
    "umulhi",
    "smin",
    "smax",
    "sextend8",
    "input_base",
    "data_init",
]


def _find_wasmtime() -> str:
    """Find the wasmtime binary."""
    # Check common locations
    for path in [
        os.path.expanduser("~/.wasmtime/bin/wasmtime"),
        shutil.which("wasmtime"),
    ]:
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "wasmtime not found. Install via: curl https://wasmtime.dev/install.sh -sSf | bash"
    )


def _get_input_base(wasm_path: str) -> int:
    """Get the input_base address from the WASM module (__heap_base export)."""
    from transformer_vm.compilation.decoder import decode

    with open(wasm_path, "rb") as f:
        mod = decode(f.read())
    for exp in mod.exports:
        if exp.name == "__heap_base" and exp.kind == 3:
            return (mod.globals[exp.index]["init"] + 15) & ~15
    return 0


def _get_data_segments(wasm_path: str) -> list[tuple[int, bytes]]:
    """Get initialized data segments from the WASM module."""
    from transformer_vm.compilation.decoder import decode

    with open(wasm_path, "rb") as f:
        mod = decode(f.read())
    return [(seg.offset, seg.data) for seg in mod.data_segments]


def _get_stack_pointer_init(wasm_path: str) -> int:
    """Get the initial stack pointer value from the WASM module."""
    from transformer_vm.compilation.decoder import decode

    with open(wasm_path, "rb") as f:
        mod = decode(f.read())
    # Global 0 is typically __stack_pointer
    if mod.globals:
        return mod.globals[0]["init"]
    return 0


def compile_c_to_clif(c_path: str, args: str = "") -> SimpleProg:
    """Compile a C program to simplified CLIF via WASM.

    Returns a SimpleProg with the flattened instruction list.
    """
    from transformer_vm.compilation.compile_wasm import compile_c_to_wasm

    # Step 1: C → WASM
    wasm_path = compile_c_to_wasm(c_path)
    logger.info("Compiled %s → %s", c_path, wasm_path)

    # Step 2: WASM → CLIF via wasmtime
    wasmtime = _find_wasmtime()
    clif_dir = tempfile.mkdtemp(prefix="clif_")
    try:
        result = subprocess.run(
            [wasmtime, "compile", "--emit-clif", clif_dir, wasm_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"wasmtime compile failed: {result.stderr}")

        # Parse all function CLIF files
        clif_files = sorted(globmod.glob(os.path.join(clif_dir, "wasm*function*.clif")))
        if not clif_files:
            raise RuntimeError(f"No CLIF files generated in {clif_dir}")

        functions = [parse_clif_file(f) for f in clif_files]
        logger.info("Parsed %d CLIF functions from %s", len(functions), wasm_path)

        # Get input_base, stack pointer, and data segments from WASM module
        input_base = _get_input_base(wasm_path)
        stack_pointer_init = _get_stack_pointer_init(wasm_path)
        data_segments = _get_data_segments(wasm_path)
        logger.info(
            "Input base: %d (0x%x), Stack pointer: %d (0x%x), Data segments: %d",
            input_base, input_base, stack_pointer_init, stack_pointer_init, len(data_segments),
        )

        # Step 3: Subset and flatten
        prog = subset_and_flatten(
            functions, input_base=input_base,
            stack_pointer_init=stack_pointer_init,
            data_segments=data_segments,
        )
        logger.info("Simplified to %d instructions, %d vars", len(prog.instrs), prog.max_var + 1)

        return prog
    finally:
        shutil.rmtree(clif_dir, ignore_errors=True)


def _encode_instr(instr: SimpleInstr) -> list[int]:
    """Encode a SimpleInstr as 6 data bytes [f0..f5].

    Encoding per opcode type:
      iconst:  [dest, imm_b0, imm_b1, imm_b2, imm_b3, 0]
      iadd/isub/...:  [dest, src1, src2, 0, 0, 0]  (or [dest, src1, 0, imm_lo, imm_hi, 0] if immediate)
      icmp:    [dest, src1, src2, cond, 0, 0]
      select:  [dest, src1(cond), src2(true), src3(false), 0, 0]
      load:    [dest, src1(addr), off_lo, off_hi, 0, 0]
      store:   [src1(val), src2(addr), off_lo, off_hi, 0, 0]
      brif:    [src1(cond), true_off_lo, true_off_hi, false_off_lo, false_off_hi, 0]
      jump:    [off_lo, off_hi, 0, 0, 0, 0]
      copy/copy_true/copy_false: [dest, src1, src2(cond for _true/_false), 0, 0, 0]
      output:  [src1, 0, 0, 0, 0, 0]
      halt/return: [0, 0, 0, 0, 0, 0]
    """

    def v(x):
        return (x or 0) & 0xFF

    def lo(x):
        return x & 0xFF

    def hi(x):
        return (x >> 8) & 0xFF

    def signed_16(x):
        """Convert signed offset to unsigned 16-bit."""
        return x & 0xFFFF

    op = instr.opcode

    if op == "iconst":
        imm = instr.imm & MASK32
        return [v(instr.dest), lo(imm), hi(imm), (imm >> 16) & 0xFF, (imm >> 24) & 0xFF, 0]

    if op in ("iadd", "isub", "imul", "band", "bor", "bxor", "ishl", "ushr", "sshr"):
        if instr.src2 is not None:
            return [v(instr.dest), v(instr.src1), v(instr.src2), 0, 0, 0]
        # Immediate form
        imm = instr.imm & MASK32
        return [v(instr.dest), v(instr.src1), lo(imm), hi(imm), (imm >> 16) & 0xFF, (imm >> 24) & 0xFF]

    if op == "icmp":
        return [v(instr.dest), v(instr.src1), v(instr.src2), instr.cond & 0xFF, 0, 0]

    if op == "select":
        return [v(instr.dest), v(instr.src1), v(instr.src2), v(instr.src3), 0, 0]

    if op in ("load", "uload8", "sload8", "uload16", "sload16"):
        off = instr.imm & 0xFFFF
        return [v(instr.dest), v(instr.src1), lo(off), hi(off), 0, 0]

    if op in ("store", "store8", "store16"):
        # f0=0(no dest), f1=val(src1), f2=addr(src2), f3:f4=offset
        off = instr.imm & 0xFFFF
        return [0, v(instr.src1), v(instr.src2), lo(off), hi(off), 0]

    if op == "brif":
        # f0=0, f1=cond(src1), f2:f5=32-bit signed offset (same as WASM br_if)
        imm = instr.imm & MASK32
        return [0, v(instr.src1), lo(imm), hi(imm), (imm >> 16) & 0xFF, (imm >> 24) & 0xFF]

    if op == "jump":
        # f0:f3=32-bit signed offset, f4:f5=0
        imm = instr.imm & MASK32
        return [lo(imm), hi(imm), (imm >> 16) & 0xFF, (imm >> 24) & 0xFF, 0, 0]

    if op in ("copy", "copy_true", "copy_false"):
        # f0=dest, f1=src, f2=cond(for _true/_false)
        return [v(instr.dest), v(instr.src1), v(instr.src2), 0, 0, 0]

    if op == "output":
        # f0=0(no dest), f1=src
        return [0, v(instr.src1), 0, 0, 0, 0]

    if op in ("ineg", "sextend8"):
        return [v(instr.dest), v(instr.src1), 0, 0, 0, 0]

    if op in ("umulhi", "smin", "smax"):
        if instr.src2 is not None:
            return [v(instr.dest), v(instr.src1), v(instr.src2), 0, 0, 0]
        imm = instr.imm & MASK32
        return [v(instr.dest), v(instr.src1), lo(imm), hi(imm), (imm >> 16) & 0xFF, (imm >> 24) & 0xFF]

    if op == "input_base":
        # f0=0(no dest), f1:f4=immediate (same layout as iconst for immediate field)
        imm = instr.imm & MASK32
        return [0, lo(imm), hi(imm), (imm >> 16) & 0xFF, (imm >> 24) & 0xFF, 0]

    if op in ("halt", "return", "call"):
        return [0, 0, 0, 0, 0, 0]

    logger.warning("Unknown opcode for encoding: %s", op)
    return [0, 0, 0, 0, 0, 0]


def tokenize_program(prog: SimpleProg, input_str: str = "") -> str:
    """Convert a SimpleProg to the token prefix format.

    Format: { opcode hex0 hex1 hex2 hex3 hex4 hex5 ... } input_tokens commit(...)

    Each instruction is 7 tokens: opcode name + 6 hex bytes.
    """
    lines = ["{"]

    # Emit data segment initialization as store8 instructions
    if prog.data_segments:
        for offset, data in prog.data_segments:
            for i, byte in enumerate(data):
                if byte != 0:
                    addr = offset + i
                    addr32 = addr & MASK32
                    lines.append(
                        f"data_init {addr32 & 0xFF:02x} {(addr32 >> 8) & 0xFF:02x} "
                        f"{(addr32 >> 16) & 0xFF:02x} {(addr32 >> 24) & 0xFF:02x} "
                        f"{byte:02x} 00"
                    )

    # Emit input_base pseudo-instruction
    if prog.input_base:
        ib = SimpleInstr(opcode="input_base", imm=prog.input_base)
        data = _encode_instr(ib)
        tokens = ["input_base"] + [f"{b:02x}" for b in data]
        lines.append(" ".join(tokens))

    # Emit program instructions
    for instr in prog.instrs:
        data = _encode_instr(instr)
        tokens = [instr.opcode] + [f"{b:02x}" for b in data]
        lines.append(" ".join(tokens))

    lines.append("}")

    # Emit input tokens (same format as WASM)
    if input_str:
        input_bytes = input_str.encode("utf-8") + b"\x00"
        for b in input_bytes:
            if 0x20 < b < 0x7F:
                lines.append(chr(b))
            else:
                lines.append(f"{b:02x}")
        lines.append("commit(+1,vw=0,bt=0)")

    return "\n".join(lines) + "\n"


def compile_and_save(
    c_path: str,
    args: str = "",
    output_dir: str | None = None,
    name: str | None = None,
) -> str:
    """Compile a C program to CLIF and save the token prefix file.

    Returns the path to the output .txt file.
    """
    if name is None:
        name = os.path.splitext(os.path.basename(c_path))[0]
    if output_dir is None:
        output_dir = DATA_DIR

    os.makedirs(output_dir, exist_ok=True)

    prog = compile_c_to_clif(c_path, args=args)
    token_str = tokenize_program(prog, input_str=args)

    out_path = os.path.join(output_dir, f"{name}_clif.txt")
    with open(out_path, "w") as f:
        f.write(token_str)
    logger.info("Wrote %s", out_path)

    # Also write spec input (for specialized model)
    spec_tokens = ["start"]
    if args:
        for b in args.encode("utf-8") + b"\x00":
            if 0x20 < b < 0x7F:
                spec_tokens.append(chr(b))
            else:
                spec_tokens.append(f"{b:02x}")
        spec_tokens.append("commit(+0,sts=0,bt=0)")

    spec_path = os.path.join(output_dir, f"{name}_clif_spec.txt")
    with open(spec_path, "w") as f:
        f.write(" ".join(spec_tokens) + "\n")
    logger.info("Wrote %s", spec_path)

    return out_path


def ensure_clif_data():
    """Compile all programs from manifest to CLIF format if not already done."""
    import yaml

    with open(MANIFEST) as f:
        manifest = yaml.safe_load(f)

    for entry in manifest.get("programs", []):
        name = entry["name"]
        args = entry.get("args", "")
        out_path = os.path.join(DATA_DIR, f"{name}_clif.txt")

        if os.path.exists(out_path):
            continue

        c_path = os.path.join(EXAMPLES_DIR, f"{name}.c")
        if not os.path.exists(c_path):
            logger.warning("Skipping %s: %s not found", name, c_path)
            continue

        try:
            compile_and_save(c_path, args=args, name=name)
        except Exception as e:
            logger.warning("Failed to compile %s to CLIF: %s", name, e)


# ── CLI ──────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Compile C programs to CLIF token-prefix format.")
    parser.add_argument("source", help="C source file (.c) or WASM binary (.wasm)")
    parser.add_argument("--args", type=str, default="", help="Input arguments for the program")
    parser.add_argument("-o", "--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--name", type=str, default=None, help="Program name (default: filename)")
    args = parser.parse_args()

    compile_and_save(args.source, args=args.args, output_dir=args.output_dir, name=args.name)


if __name__ == "__main__":
    main()
