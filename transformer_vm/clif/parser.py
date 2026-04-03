"""Parse Cranelift IR (CLIF) text format from wasmtime --emit-clif output.

Parses a single .clif file into a CLIFFunction containing blocks and instructions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class CLIFInstr:
    """A single CLIF instruction."""

    opcode: str  # "iadd", "iconst", "brif", "store", "jump", "return", etc.
    dest: int | None = None  # destination v-number (None for void ops)
    type: str = "i32"  # result type (i32, i64, i8, etc.)
    operands: list = field(default_factory=list)  # source v-numbers
    immediates: list = field(default_factory=list)  # integer constants
    cond: str | None = None  # comparison condition (eq, ne, slt, etc.)
    targets: list = field(default_factory=list)  # branch targets [(block_id, [args])]
    flags: list = field(default_factory=list)  # memory flags (notrap, aligned, etc.)
    sig_ref: str | None = None  # signature reference for call_indirect
    fn_ref: str | None = None  # function reference for call
    offset: int | None = None  # memory offset (v+offset form)


@dataclass
class CLIFBlock:
    """A basic block in CLIF."""

    id: int
    params: list = field(default_factory=list)  # [(v_num, type_str), ...]
    instrs: list[CLIFInstr] = field(default_factory=list)


@dataclass
class CLIFFunction:
    """A parsed CLIF function."""

    name: str
    func_id: str  # e.g. "u0:0"
    params: list = field(default_factory=list)  # [(type_str, name_or_none), ...]
    returns: list[str] = field(default_factory=list)  # return types
    calling_conv: str = "tail"
    blocks: list[CLIFBlock] = field(default_factory=list)
    # Preamble declarations
    global_values: dict = field(default_factory=dict)  # gvN -> definition
    signatures: dict = field(default_factory=dict)  # sigN -> param/return types
    fn_refs: dict = field(default_factory=dict)  # fnN -> (func_id, sig_ref)
    stack_slots: dict = field(default_factory=dict)  # ssN -> size
    aliases: dict = field(default_factory=dict)  # vN -> vM (value aliases)


# ── Regex patterns ─────────────────────────────────────────────

_RE_FUNC_HEADER = re.compile(
    r"function\s+(%?\w[\w:.]*)\((.*?)\)"
    r"(?:\s*->\s*([\w,\s]+))?"
    r"(?:\s+(\w+))?\s*\{"
)
_RE_BLOCK = re.compile(r"block(\d+)(?:\((.*?)\))?:")
_RE_VNUM = re.compile(r"v(\d+)")
_RE_ASSIGN = re.compile(r"v(\d+)\s*=\s*(.*)")
_RE_ALIAS = re.compile(r"v(\d+)\s*->\s*v(\d+)")
_RE_OFFSET = re.compile(r"v(\d+)\+(\d+)")
_RE_HEX_INT = re.compile(r"0x[0-9a-fA-F_]+")
_RE_SOURCE_ANNOT = re.compile(r"^@[0-9a-fA-F]+\s+")
_RE_BLOCK_TARGET = re.compile(r"block(\d+)(?:\(([^)]*)\))?")


def _strip_comment(line: str) -> str:
    """Remove trailing ; comment from a line."""
    # Careful not to strip inside strings (there are none in CLIF)
    idx = line.find(";")
    if idx >= 0:
        return line[:idx].rstrip()
    return line.rstrip()


def _parse_int(s: str) -> int:
    """Parse an integer literal (decimal, hex, or negative)."""
    s = s.strip().replace("_", "")
    if s.startswith("0x") or s.startswith("-0x"):
        return int(s, 16)
    return int(s)


def _parse_type_suffix(opcode_str: str) -> tuple[str, str]:
    """Split 'iadd.i32' into ('iadd', 'i32')."""
    if "." in opcode_str:
        parts = opcode_str.split(".", 1)
        return parts[0], parts[1]
    return opcode_str, ""


# ── Memory/instruction flags ──────────────────────────────────

_MEM_FLAGS = frozenset(
    [
        "notrap",
        "aligned",
        "readonly",
        "can_move",
        "checked",
        "little",
        "heap",
        "table",
    ]
)

_ICMP_CONDS = frozenset(
    ["eq", "ne", "slt", "sgt", "sle", "sge", "ult", "ugt", "ule", "uge"]
)


def _has_open_paren(tok: str) -> bool:
    """Check if token has ( without matching )."""
    return "(" in tok and ")" not in tok


def _retokenize(tokens: list[str]) -> list[str]:
    """Re-tokenize to merge split parenthesized args back together.

    e.g. ['block4(v46', 'v61)'] -> ['block4(v46,v61)']
         ['fn0(v0', 'v0', 'v2)'] -> ['fn0(v0,v0,v2)']
    """
    result = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if _has_open_paren(tok):
            merged = tok
            i += 1
            while i < len(tokens):
                merged += "," + tokens[i]
                if ")" in tokens[i]:
                    break
                i += 1
            result.append(merged)
        else:
            result.append(tok)
        i += 1
    return result


def _parse_operands(tokens: list[str], instr: CLIFInstr):
    """Parse the operand tokens for an instruction."""
    tokens = _retokenize(tokens)
    i = 0
    while i < len(tokens):
        tok = tokens[i]

        # Skip memory flags
        if tok in _MEM_FLAGS:
            instr.flags.append(tok)
            i += 1
            continue

        # Comparison condition code (only for icmp, not brif which uses v-num condition)
        if tok in _ICMP_CONDS and instr.opcode == "icmp":
            instr.cond = tok
            i += 1
            continue

        # Signature reference (sig0, sig1, ...)
        if tok.startswith("sig") and tok[3:].isdigit():
            instr.sig_ref = tok
            i += 1
            continue

        # Function reference (fn0, fn1, ...)
        if tok.startswith("fn") and tok[2:].isdigit():
            instr.fn_ref = tok
            i += 1
            continue

        # Block target with possible args: block3 or block3(v1,v2)
        m = _RE_BLOCK_TARGET.match(tok)
        if m and tok.startswith("block"):
            block_id = int(m.group(1))
            args = []
            if m.group(2):
                for a in m.group(2).split(","):
                    a = a.strip()
                    vm = _RE_VNUM.match(a)
                    if vm:
                        args.append(int(vm.group(1)))
            instr.targets.append((block_id, args))
            i += 1
            continue

        # v-number with offset: v0+72
        m = _RE_OFFSET.match(tok)
        if m:
            instr.operands.append(int(m.group(1)))
            instr.offset = int(m.group(2))
            i += 1
            continue

        # Call with args in parens: v7(v6,v0,v4) or fn0(v0,v0,v2,v7,v93)
        if "(" in tok and ")" in tok:
            parts = tok.split("(", 1)
            prefix = parts[0]
            arg_str = parts[1].rstrip(")")
            # Check if prefix is a v-number or fn-ref
            vm = _RE_VNUM.match(prefix)
            if vm:
                instr.operands.append(int(vm.group(1)))
            elif prefix.startswith("fn") and prefix[2:].isdigit():
                instr.fn_ref = prefix
            # Parse args
            if arg_str:
                for a in arg_str.split(","):
                    a = a.strip()
                    avm = _RE_VNUM.match(a)
                    if avm:
                        instr.operands.append(int(avm.group(1)))
            i += 1
            continue

        # Plain v-number: v42
        m = _RE_VNUM.match(tok)
        if m:
            instr.operands.append(int(m.group(1)))
            i += 1
            continue

        # Hex integer literal
        if _RE_HEX_INT.match(tok):
            instr.immediates.append(_parse_int(tok))
            i += 1
            continue

        # Decimal integer literal (including negative)
        try:
            val = int(tok)
            instr.immediates.append(val)
            i += 1
            continue
        except ValueError:
            pass

        # Skip unknown tokens
        i += 1


def _parse_instruction_line(line: str) -> CLIFInstr | tuple[int, int] | None:
    """Parse a single instruction line. Returns CLIFInstr or (alias_from, alias_to) or None."""
    # Strip source annotation (@xxxx prefix)
    line = _RE_SOURCE_ANNOT.sub("", line)
    line = line.strip()
    if not line:
        return None

    line = _strip_comment(line)
    if not line:
        return None

    # Check for value alias: v204 -> v266
    m = _RE_ALIAS.match(line)
    if m:
        return (int(m.group(1)), int(m.group(2)))

    # Check for assignment: vN = opcode ...
    m = _RE_ASSIGN.match(line)
    if m:
        dest_v = int(m.group(1))
        rest = m.group(2).strip()
        # Split into tokens
        tokens = rest.replace(",", " ").split()
        if not tokens:
            return None

        opcode_raw = tokens[0]
        opcode, type_suffix = _parse_type_suffix(opcode_raw)
        instr = CLIFInstr(opcode=opcode, dest=dest_v, type=type_suffix or "i32")
        _parse_operands(tokens[1:], instr)
        return instr

    # Void instruction: opcode ...
    tokens = line.replace(",", " ").split()
    if not tokens:
        return None

    opcode_raw = tokens[0]
    opcode, type_suffix = _parse_type_suffix(opcode_raw)

    # Skip block headers (handled separately)
    if opcode.startswith("block"):
        return None

    instr = CLIFInstr(opcode=opcode, type=type_suffix or "")
    _parse_operands(tokens[1:], instr)
    return instr


# ── Top-level parser ──────────────────────────────────────────


def parse_clif(text: str) -> CLIFFunction:
    """Parse CLIF text into a CLIFFunction."""
    lines = text.split("\n")

    func = CLIFFunction(name="", func_id="")
    current_block = None

    for line_raw in lines:
        line = line_raw.strip()

        # Skip comments and empty lines
        if not line or line.startswith(";;"):
            continue

        # Function header
        m = _RE_FUNC_HEADER.search(line)
        if m:
            func.func_id = m.group(1)
            func.name = m.group(1)
            # Parse params
            if m.group(2):
                for p in m.group(2).split(","):
                    p = p.strip()
                    if not p:
                        continue
                    parts = p.split()
                    func.params.append((parts[0], parts[1] if len(parts) > 1 else None))
            if m.group(3):
                func.returns = [t.strip() for t in m.group(3).split(",")]
            if m.group(4):
                func.calling_conv = m.group(4)
            continue

        # Closing brace
        if line == "}":
            continue

        # Preamble: gvN = ...
        if line.startswith("gv"):
            parts = line.split("=", 1)
            if len(parts) == 2:
                func.global_values[parts[0].strip()] = parts[1].strip()
            continue

        # Preamble: sigN = ...
        if line.startswith("sig"):
            parts = line.split("=", 1)
            if len(parts) == 2:
                func.signatures[parts[0].strip()] = parts[1].strip()
            continue

        # Preamble: fnN = ...
        if line.startswith("fn"):
            parts = line.split("=", 1)
            if len(parts) == 2:
                rhs = parts[1].strip()
                # "colocated u0:1 sig0"
                rhs_parts = rhs.split()
                func_id = rhs_parts[1] if len(rhs_parts) > 1 else rhs_parts[0]
                sig_ref = rhs_parts[2] if len(rhs_parts) > 2 else None
                func.fn_refs[parts[0].strip()] = (func_id, sig_ref)
            continue

        # Preamble: ssN = explicit_slot N
        if line.startswith("ss"):
            parts = line.split("=", 1)
            if len(parts) == 2:
                size_match = re.search(r"\d+", parts[1])
                if size_match:
                    func.stack_slots[parts[0].strip()] = int(size_match.group())
            continue

        # Preamble: stack_limit = ...
        if line.startswith("stack_limit"):
            continue

        # Block header (may appear with lots of leading whitespace)
        stripped = _RE_SOURCE_ANNOT.sub("", line_raw).strip()
        m = _RE_BLOCK.match(stripped)
        if m:
            block_id = int(m.group(1))
            params = []
            if m.group(2):
                for p in m.group(2).split(","):
                    p = p.strip()
                    parts = p.split(":")
                    vm = _RE_VNUM.match(parts[0].strip())
                    if vm:
                        ptype = parts[1].strip() if len(parts) > 1 else "i32"
                        params.append((int(vm.group(1)), ptype))
            current_block = CLIFBlock(id=block_id, params=params)
            func.blocks.append(current_block)
            continue

        # Instruction line (inside a block)
        if current_block is not None:
            result = _parse_instruction_line(line_raw)
            if result is None:
                continue
            if isinstance(result, tuple):
                # Value alias: vN -> vM
                func.aliases[result[0]] = result[1]
            elif isinstance(result, CLIFInstr):
                current_block.instrs.append(result)

    return func


def parse_clif_file(path: str) -> CLIFFunction:
    """Parse a .clif file."""
    with open(path) as f:
        return parse_clif(f.read())
