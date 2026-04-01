"""Tests for the CLIF IR interpreter pipeline.

Verifies: CLIF parser, subsetting pass, compilation pipeline,
reference interpreter, and (when implemented) CALM graph interpreter.
"""

import os

import pytest


@pytest.fixture(scope="session")
def clif_data(data_dir):
    """Ensure CLIF program files exist (at least hello)."""
    import logging

    from transformer_vm.compilation.compile_clif import compile_and_save
    from transformer_vm.clif.reference import generate_ref

    # Compile hello (required for tests)
    hello_clif = os.path.join(data_dir, "hello_clif.txt")
    if not os.path.exists(hello_clif):
        from transformer_vm._paths import EXAMPLES_DIR

        compile_and_save(
            os.path.join(EXAMPLES_DIR, "hello.c"), args="World", name="hello"
        )

    # Generate reference trace
    hello_ref = os.path.join(data_dir, "hello_clif_ref.txt")
    if not os.path.exists(hello_ref):
        generate_ref(hello_clif, hello_ref)

    return data_dir


# ── Machine build tests ───────────────────────────────────────


def test_clif_machine_builds():
    """CLIFMachine.build() produces a valid ProgramGraph."""
    from transformer_vm.clif.interpreter import CLIFMachine

    pg = CLIFMachine().build()
    assert pg.input_tokens is not None
    assert pg.output_tokens is not None
    assert len(pg.all_dims) > 0
    assert len(pg.all_lookups) > 0
    # Should have CLIF opcode tokens
    assert "iconst" in pg.input_tokens
    assert "iadd" in pg.input_tokens
    assert "halt" in pg.input_tokens


# ── Parser tests ──────────────────────────────────────────────


def test_clif_parser_hello(clif_data):
    """CLIF parser reads hello_clif.txt correctly."""
    from transformer_vm.clif.reference import load_clif_program

    prog, input_str = load_clif_program(os.path.join(clif_data, "hello_clif.txt"))
    assert len(prog) > 0
    assert input_str == "World"
    # First instruction after input_base should be iconst (for the param init or 'H')
    opcodes = [op for op, _data in prog]
    assert "iconst" in opcodes
    assert "output" in opcodes
    assert "halt" in opcodes


# ── Reference interpreter tests ───────────────────────────────


def test_clif_reference_hello(clif_data):
    """CLIF reference interpreter produces correct output for hello."""
    from transformer_vm.clif.reference import load_clif_program, run

    prog, input_str = load_clif_program(os.path.join(clif_data, "hello_clif.txt"))
    _instrs, _tokens, output = run(prog, input_str)
    assert output == "Hello World!\n"


def test_clif_reference_hello_trace(clif_data):
    """CLIF reference interpreter generates valid trace tokens."""
    from transformer_vm.clif.reference import load_clif_program, run

    prog, input_str = load_clif_program(os.path.join(clif_data, "hello_clif.txt"))
    _instrs, _tokens, output, trace = run(prog, input_str, trace=True)
    assert output == "Hello World!\n"
    assert trace[-1] == "halt"
    # Check trace has out() tokens for each output character
    out_tokens = [t for t in trace if t.startswith("out(")]
    assert len(out_tokens) == len(output)


# ── Cross-IR comparison tests ─────────────────────────────────


@pytest.mark.parametrize("program,args,expected_prefix", [
    ("hello", "World", "Hello World!"),
])
def test_cross_ir_output_hello(clif_data, program, args, expected_prefix):
    """WASM and CLIF reference interpreters produce same output."""
    from transformer_vm.clif.reference import load_clif_program
    from transformer_vm.clif.reference import run as clif_run
    from transformer_vm.wasm.reference import load_program
    from transformer_vm.wasm.reference import run as wasm_run

    wasm_prog, wasm_input = load_program(os.path.join(clif_data, f"{program}.txt"))
    _, _, wasm_output = wasm_run(wasm_prog, wasm_input)

    clif_prog, clif_input = load_clif_program(os.path.join(clif_data, f"{program}_clif.txt"))
    _, _, clif_output = clif_run(clif_prog, clif_input)

    assert wasm_output == clif_output
    assert wasm_output.startswith(expected_prefix)
