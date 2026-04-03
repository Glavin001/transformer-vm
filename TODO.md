# CLIF Interpreter TODO

## Current State (50 tests, 12 programs)

### What Works

**CLIF graph evaluator (analytical transformer weights): 11/12 programs exact match**

| Program | Output | Features | Status |
|---------|--------|----------|--------|
| hello | `Hello World!\n` | string output, memory read, loop | ✓ |
| countdown | `9876543210\n` | arithmetic loop (no input) | ✓ |
| loop3 | `ABC\n` | hardcoded while(i<3) | ✓ |
| loop_n | `XXX\n` | runtime loop bound | ✓ regression: copy_true cond bug |
| minmax | `EO\n` | smin/smax, conditional branches | ✓ |
| store_load | `XY\n` | store8→sload8 roundtrip | ✓ regression: i64 offset bug |
| copy_buf | `AB\n` | store8+uload8 loop | ✓ |
| reverse | `dcba\n` | computed addresses + memory loops | ✓ |
| addition | `19134\n` | multi-loop digit arithmetic | ✓ |
| sum_bytes | `\x83\x00\n` | 32-bit store + load | ✓ |
| max_char | `h\n` | smax conditional update | ✓ |
| multiply | `21\n` | **needs imul/umulhi/ushr** | ref ✓, graph ✗ |

**Reference interpreter + cross-IR (WASM==CLIF): all 12 programs pass.**

### Interpreter Complexity

```
             WASM     CLIF    delta
Total dims    186      216      +30
Attn heads     21       26       +5
```

## Immediate TODOs

### 1. Wire `imul` into CALM graph result_byte

The `multiply` program needs `imul` (integer multiply). The ALM can express
multiply as `a*b = reglu(a,b) - reglu(a,-b)` (already in `core.py` as
`_make_multiply`). Need to add it to the CLIF interpreter's `result_byte`.

Also needs: `umulhi` (upper 32 bits of 64-bit multiply) and `ushr` (unsigned
right shift). These are used by Cranelift for division-by-constant optimization.
Options: (a) wire them natively, or (b) lower them in the subsetting pass.

### 2. Wire remaining bitwise/shift opcodes

Uncovered opcodes (not in any test program yet):
- `bor` (bitwise OR)
- `bxor` (bitwise XOR)
- `ishl` (left shift)
- `ushr` (unsigned right shift)
- `sshr` (signed right shift)
- `store16`, `uload16`, `sload16` (16-bit memory ops)

These can't be expressed directly in the ALM (no bitwise primitives).
Options:
- Lower to add/sub sequences in the subsetting pass (like WASM's `lower.py`)
- Support natively for specific constant patterns (e.g., shift by constant)

### 3. Fix Cranelift 5+ char loop unrolling

`reverse("abcde")` (5 chars) fails because Cranelift unrolls the loop at 5+
iterations, generating code patterns the graph evaluator can't handle.
`reverse("abcd")` (4 chars) works. Investigate the specific unrolling pattern.

### 4. Multi-function support (collatz, fibonacci)

Collatz and fibonacci call `sscanf` and `printf` helper functions.
Current status: function inlining infrastructure exists but the inlined
callees' memory operations produce wrong values (stack pointer chain issue).

Options:
- Fix the inlining variable mapping (medium effort)
- Add `call`/`return` opcodes to the CLIF CALM interpreter (large effort,
  mirrors WASM's call/return with `delta_call_depth` cumsum)

### 5. WASM empty-input bug

`countdown` and `loop3` (programs with `args=""`) fail in the WASM graph
evaluator. Pre-existing issue on main branch — the WASM evaluator produces
`commit(+0,sts=0,bt=0)` where a byte is expected. Not a CLIF issue.

## Architecture Notes

### Key bugs found during development (keep as regression tests)

1. **copy_true condition variable** (`loop_n` regression test):
   `copy_true v2=v11, cond=v5` used `src1_value` (v11=source) for the
   condition instead of `src2_value` (v5=actual condition). Fixed by adding
   `cond_nonzero_src2`.

2. **copy_true value emission** (`loop_n` regression test):
   When condition was false, emitted source value instead of destination's
   current value. Fixed by adding `dest_byte` lookup (+2 attention heads).

3. **i64 address+constant offset** (`store_load` regression test):
   `iadd(heap_base + uextend(ptr), iconst 11)` resolved to just `ptr`,
   dropping the +11 offset. Fixed by reordering the i64 iadd analysis to
   check addr+const patterns before dead+ext patterns.

4. **bnot lowering**: Cranelift generates `bnot` for bitwise NOT. Lowered
   to `isub(0xFFFFFFFF, v)` in the subsetting pass.

### Token format

7 tokens per instruction: `opcode f0 f1 f2 f3 f4 f5`
- f0 = dest_v (or 0 for void ops)
- f1 = src1_v
- f2 = src2_v (or offset bytes)
- f3-f5 = aux (condition code, offset continuation, etc.)

### Variable access (two-step lookup)

1. At commit with var_write=1: write `(dest_v → store_value, position-4)`
2. Query src_v → get (32-bit value, position) in one shot
3. Use position to fetch individual bytes for emission

`dest_v_at_commit = fetch(byte_number-1, query=7*cursor-5, key=position)`
resolves the dest variable from the PREVIOUS instruction (since cursor has
been incremented by the commit's delta_cursor=1).

### Files

| File | Lines | Purpose |
|------|-------|---------|
| `clif/interpreter.py` | ~600 | CALM graph interpreter (26 heads, 216 dims) |
| `clif/parser.py` | ~410 | Parse wasmtime CLIF text output |
| `clif/subset.py` | ~900 | Simplify raw CLIF + function inlining |
| `clif/reference.py` | ~650 | Direct interpreter + trace generation |
| `compilation/compile_clif.py` | ~370 | C→WASM→CLIF→tokens pipeline |
| `tests/test_clif.py` | ~230 | 50 tests across 12 programs |

### Branch

`claude/add-cranelift-interpreter-8XDOl`

PR: https://github.com/Glavin001/transformer-vm/pull/1
