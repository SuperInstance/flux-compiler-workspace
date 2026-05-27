"""Comprehensive test suite for flux_compiler."""

import pytest
from flux_compiler.lexer import FluxLexer, Token, TokenKind, LexerError
from flux_compiler.parser import (
    FluxParser, ParseError,
    RangeExpr, DomainExpr, DomainMaskExpr, ExactExpr, ComparisonExpr,
    AndExpr, OrExpr, NotExpr, GroupedExpr,
    TemporalExpr, SecurityExpr, MetadataStmt, Program,
)
from flux_compiler.ir import IrBuilder, IrInstruction, IrModule, IrOpCode, HaltReason, BasicBlock
from flux_compiler.optimizer import PeepholeOptimizer
from flux_compiler.codegen import CodeGenerator, Target, CodegenError


# ═══════════════════════════════════════════════════════════
# Lexer Tests
# ═══════════════════════════════════════════════════════════


class TestFluxLexer:

    def test_empty_input(self):
        tokens = FluxLexer("").tokenize()
        assert len(tokens) == 1
        assert tokens[0].kind == TokenKind.EOF

    def test_whitespace_only(self):
        tokens = FluxLexer("  \t\n  ").tokenize()
        assert tokens[0].kind == TokenKind.EOF

    def test_comment(self):
        tokens = FluxLexer("# this is a comment\nx IN 0..100").tokenize()
        values = [t.value for t in tokens]
        # Comments are skipped entirely, so no comment text in output
        assert "# this is a comment" not in values
        assert TokenKind.IDENT in [t.kind for t in tokens]

    def test_identifier(self):
        tokens = FluxLexer("my_field").tokenize()
        assert tokens[0] == Token(kind=TokenKind.IDENT, value="my_field")

    def test_dotted_field(self):
        tokens = FluxLexer("a.b.c").tokenize()
        values = [t.value for t in tokens if t.kind != TokenKind.EOF]
        assert values == ["a", ".", "b", ".", "c"]

    def test_integer(self):
        tokens = FluxLexer("42").tokenize()
        assert tokens[0] == Token(kind=TokenKind.INT, value="42")

    def test_negative_integer(self):
        tokens = FluxLexer("-5").tokenize()
        assert tokens[0] == Token(kind=TokenKind.INT, value="-5")

    def test_float(self):
        tokens = FluxLexer("3.14").tokenize()
        assert tokens[0] == Token(kind=TokenKind.FLOAT, value="3.14")

    def test_string(self):
        tokens = FluxLexer('"hello world"').tokenize()
        assert tokens[0] == Token(kind=TokenKind.STRING, value="hello world")

    def test_string_escape(self):
        tokens = FluxLexer(r'"say \"hi\""').tokenize()
        assert tokens[0] == Token(kind=TokenKind.STRING, value='say "hi"')

    def test_keywords(self):
        source = "IN IS AND OR NOT FOR WITH SECURITY TRUSTED CLEARANCE TOLERANCE DURATION"
        tokens = FluxLexer(source).tokenize()
        keyword_kinds = [TokenKind.IN, TokenKind.IS, TokenKind.AND, TokenKind.OR,
                         TokenKind.NOT, TokenKind.FOR, TokenKind.WITH, TokenKind.SECURITY,
                         TokenKind.TRUSTED, TokenKind.CLEARANCE, TokenKind.TOLERANCE,
                         TokenKind.DURATION]
        for tok, expected in zip(tokens, keyword_kinds):
            assert tok.kind == expected

    def test_operators(self):
        source = ".. == != >= <= > < @ ,"
        tokens = FluxLexer(source).tokenize()
        expected = [
            TokenKind.DOTDOT, TokenKind.EQ, TokenKind.NEQ,
            TokenKind.GTE, TokenKind.LTE, TokenKind.GT, TokenKind.LT,
            TokenKind.AT, TokenKind.COMMA,
        ]
        for tok, exp in zip(tokens, expected):
            assert tok.kind == exp

    def test_parens_and_brackets(self):
        tokens = FluxLexer("()[]").tokenize()
        expected = [TokenKind.LPAREN, TokenKind.RPAREN, TokenKind.LBRACKET, TokenKind.RBRACKET]
        for tok, exp in zip(tokens, expected):
            assert tok.kind == exp

    def test_unexpected_char(self):
        with pytest.raises(LexerError):
            FluxLexer("^").tokenize()

    def test_unterminated_string(self):
        with pytest.raises(LexerError):
            FluxLexer('"no end').tokenize()

    def test_complex_source(self):
        source = 'x IN 0..100 AND y == 42'
        tokens = FluxLexer(source).tokenize()
        kinds = [t.kind for t in tokens]
        assert kinds == [
            TokenKind.IDENT, TokenKind.IN, TokenKind.INT, TokenKind.DOTDOT,
            TokenKind.INT, TokenKind.AND, TokenKind.IDENT, TokenKind.EQ,
            TokenKind.INT, TokenKind.EOF,
        ]

    def test_case_insensitive_keywords(self):
        tokens = FluxLexer("in and or not").tokenize()
        assert tokens[0].kind == TokenKind.IN
        assert tokens[1].kind == TokenKind.AND
        assert tokens[2].kind == TokenKind.OR
        assert tokens[3].kind == TokenKind.NOT


# ═══════════════════════════════════════════════════════════
# Parser Tests
# ═══════════════════════════════════════════════════════════


def _parse(source: str) -> Program:
    tokens = FluxLexer(source).tokenize()
    return FluxParser(tokens).parse()


class TestFluxParser:

    def test_range_constraint(self):
        prog = _parse("x IN 0..100")
        assert len(prog.statements) == 1
        stmt = prog.statements[0]
        assert isinstance(stmt, RangeExpr)
        assert stmt.name == "x"
        assert stmt.lo == 0
        assert stmt.hi == 100

    def test_exact_string_constraint(self):
        prog = _parse('status IS "active"')
        stmt = prog.statements[0]
        assert isinstance(stmt, ExactExpr)
        assert stmt.name == "status"
        assert stmt.value == "active"

    def test_exact_int_constraint(self):
        prog = _parse("x IS 42")
        stmt = prog.statements[0]
        assert isinstance(stmt, ExactExpr)
        assert stmt.value == 42

    def test_comparison_eq(self):
        prog = _parse("x == 42")
        stmt = prog.statements[0]
        assert isinstance(stmt, ComparisonExpr)
        assert stmt.op == "=="
        assert stmt.value == 42

    def test_comparison_gte(self):
        prog = _parse("age >= 18")
        stmt = prog.statements[0]
        assert isinstance(stmt, ComparisonExpr)
        assert stmt.op == ">="
        assert stmt.value == 18

    def test_comparison_lt(self):
        prog = _parse("temp < 100")
        stmt = prog.statements[0]
        assert isinstance(stmt, ComparisonExpr)
        assert stmt.op == "<"

    def test_domain_list(self):
        prog = _parse('color IN ["red", "green", "blue"]')
        stmt = prog.statements[0]
        assert isinstance(stmt, DomainExpr)
        assert stmt.name == "color"
        assert stmt.values == ["red", "green", "blue"]

    def test_empty_domain(self):
        prog = _parse('x IN []')
        stmt = prog.statements[0]
        assert isinstance(stmt, DomainExpr)
        assert stmt.values == []

    def test_and_combinator(self):
        prog = _parse("x IN 0..100 AND y == 42")
        stmt = prog.statements[0]
        assert isinstance(stmt, AndExpr)
        assert isinstance(stmt.left, RangeExpr)
        assert isinstance(stmt.right, ComparisonExpr)

    def test_or_combinator(self):
        prog = _parse("x == 1 OR y == 2")
        stmt = prog.statements[0]
        assert isinstance(stmt, OrExpr)

    def test_not_constraint(self):
        prog = _parse("NOT x == 0")
        stmt = prog.statements[0]
        assert isinstance(stmt, NotExpr)
        assert isinstance(stmt.expr, ComparisonExpr)

    def test_grouped(self):
        prog = _parse("(x IN 0..10 OR x IN 90..100) AND y == 5")
        stmt = prog.statements[0]
        assert isinstance(stmt, AndExpr)
        assert isinstance(stmt.left, GroupedExpr)
        group = stmt.left
        assert isinstance(group.expr, OrExpr)

    def test_not_not_grouped(self):
        prog = _parse("NOT (x == 1 AND y == 2)")
        stmt = prog.statements[0]
        assert isinstance(stmt, NotExpr)
        assert isinstance(stmt.expr, GroupedExpr)

    def test_dotted_field_name(self):
        prog = _parse("user.age IN 18..65")
        stmt = prog.statements[0]
        assert isinstance(stmt, RangeExpr)
        assert stmt.name == "user.age"

    def test_multiple_statements(self):
        prog = _parse("x IN 0..10\ny == 5")
        assert len(prog.statements) == 2

    def test_metadata(self):
        prog = _parse('@author "test"')
        stmt = prog.statements[0]
        assert isinstance(stmt, MetadataStmt)
        assert stmt.key == "author"
        assert stmt.value == "test"

    def test_parse_error_unexpected(self):
        with pytest.raises(ParseError):
            _parse("IN 0..10")

    def test_parse_error_missing_range(self):
        with pytest.raises(ParseError):
            _parse("x IN")

    def test_complex_expression(self):
        source = '(x IN 0..100 AND y == 42) OR NOT z == 0'
        prog = _parse(source)
        stmt = prog.statements[0]
        assert isinstance(stmt, OrExpr)
        assert isinstance(stmt.left, GroupedExpr)
        assert isinstance(stmt.right, NotExpr)


# ═══════════════════════════════════════════════════════════
# IR Tests
# ═══════════════════════════════════════════════════════════


class TestIR:

    def test_instruction_factory_methods(self):
        inst = IrInstruction.check_range(0, 1, 100)
        assert inst.opcode == IrOpCode.CHECK_RANGE
        assert inst.operands == (0, 1, 100)

        inst = IrInstruction.check_exact(1, 42)
        assert inst.opcode == IrOpCode.CHECK_EXACT

        inst = IrInstruction.and_()
        assert inst.opcode == IrOpCode.AND

        inst = IrInstruction.halt()
        assert inst.opcode == IrOpCode.HALT

    def test_ir_module_basic(self):
        module = IrModule(name="test")
        block = module.add_block("entry")
        block.emit(IrInstruction.check_range(0, 0, 10))
        block.emit(IrInstruction.halt())
        assert len(module.blocks) == 1
        assert len(module.all_instructions) == 2

    def test_ir_builder_range(self):
        prog = _parse("x IN 0..100")
        builder = IrBuilder("test")
        module = builder.build(prog)
        assert "x" in module.slot_map
        insts = module.all_instructions
        assert any(i.opcode == IrOpCode.CHECK_RANGE for i in insts)

    def test_ir_builder_and(self):
        prog = _parse("x IN 0..10 AND y == 5")
        builder = IrBuilder("test")
        module = builder.build(prog)
        insts = module.all_instructions
        opcodes = [i.opcode for i in insts]
        assert IrOpCode.CHECK_RANGE in opcodes
        assert IrOpCode.CHECK_COMPARISON in opcodes
        assert IrOpCode.AND in opcodes

    def test_ir_builder_or(self):
        prog = _parse("x == 1 OR y == 2")
        builder = IrBuilder("test")
        module = builder.build(prog)
        opcodes = [i.opcode for i in module.all_instructions]
        assert IrOpCode.OR in opcodes

    def test_ir_builder_not(self):
        prog = _parse("NOT x == 0")
        builder = IrBuilder("test")
        module = builder.build(prog)
        opcodes = [i.opcode for i in module.all_instructions]
        assert IrOpCode.NOT in opcodes

    def test_ir_builder_halt_appended(self):
        prog = _parse("x IN 0..10")
        builder = IrBuilder("test")
        module = builder.build(prog)
        last = module.all_instructions[-1]
        assert last.opcode == IrOpCode.HALT

    def test_slot_mapping(self):
        prog = _parse("x IN 0..10 AND y == 5 AND z == 3")
        builder = IrBuilder("test")
        module = builder.build(prog)
        assert len(module.slot_map) == 3
        assert "x" in module.slot_map
        assert "y" in module.slot_map
        assert "z" in module.slot_map

    def test_ir_builder_domain(self):
        prog = _parse('color IN ["red", "green"]')
        builder = IrBuilder("test")
        module = builder.build(prog)
        opcodes = [i.opcode for i in module.all_instructions]
        assert IrOpCode.CHECK_DOMAIN in opcodes


# ═══════════════════════════════════════════════════════════
# Optimizer Tests
# ═══════════════════════════════════════════════════════════


class TestOptimizer:

    def _make_module(self, insts: list) -> IrModule:
        module = IrModule(name="opt_test")
        block = module.add_block("entry")
        for inst in insts:
            block.emit(inst)
        return module

    def test_removes_nop(self):
        module = self._make_module([
            IrInstruction.nop(),
            IrInstruction.check_exact(0, 1),
            IrInstruction.halt(),
        ])
        opt = PeepholeOptimizer()
        stats = opt.optimize(module)
        assert stats.nops_removed >= 1
        assert all(i.opcode != IrOpCode.NOP for i in module.all_instructions)

    def test_removes_dead_after_halt(self):
        module = self._make_module([
            IrInstruction.halt(),
            IrInstruction.check_range(0, 0, 10),
            IrInstruction.check_exact(0, 5),
        ])
        opt = PeepholeOptimizer()
        stats = opt.optimize(module)
        assert stats.dead_after_halt >= 2
        insts = module.blocks[0].instructions
        assert len(insts) == 1
        assert insts[0].opcode == IrOpCode.HALT

    def test_fuses_ranges(self):
        module = self._make_module([
            IrInstruction.check_range(0, 0, 50),
            IrInstruction.check_range(0, 25, 100),
            IrInstruction.halt(),
        ])
        opt = PeepholeOptimizer()
        stats = opt.optimize(module)
        assert stats.fusions >= 1
        range_insts = [i for i in module.all_instructions if i.opcode == IrOpCode.CHECK_RANGE]
        assert len(range_insts) == 1
        assert range_insts[0].operands == (0, 0, 100)  # union of [0,50] and [25,100]

    def test_double_not_removal(self):
        module = self._make_module([
            IrInstruction.check_exact(0, 1),
            IrInstruction.not_(),
            IrInstruction.not_(),
            IrInstruction.halt(),
        ])
        opt = PeepholeOptimizer()
        stats = opt.optimize(module)
        assert stats.strength_reductions >= 1
        not_count = sum(1 for i in module.all_instructions if i.opcode == IrOpCode.NOT)
        assert not_count == 0

    def test_no_change_clean_ir(self):
        module = self._make_module([
            IrInstruction.check_range(0, 0, 10),
            IrInstruction.halt(),
        ])
        opt = PeepholeOptimizer()
        stats = opt.optimize(module)
        assert stats.nops_removed == 0
        assert stats.fusions == 0


# ═══════════════════════════════════════════════════════════
# Codegen Tests
# ═══════════════════════════════════════════════════════════


class TestCodegen:

    def _build_module(self, source: str) -> IrModule:
        prog = _parse(source)
        return IrBuilder("test_mod").build(prog)

    def test_native_codegen(self):
        module = self._build_module("x IN 0..100")
        gen = CodeGenerator(Target.NATIVE)
        output = gen.generate(module)
        assert output.target == Target.NATIVE
        assert "FLUX compiled: test_mod" in output.code
        assert "CHECK_RANGE" in output.code

    def test_c_codegen(self):
        module = self._build_module("x IN 0..100 AND y == 42")
        gen = CodeGenerator(Target.C)
        output = gen.generate(module)
        assert "#include" in output.code
        assert "flux_check_test_mod" in output.code

    def test_wasm_codegen(self):
        module = self._build_module("x == 5")
        gen = CodeGenerator(Target.WASM)
        output = gen.generate(module)
        assert "(module" in output.code
        assert "(export" in output.code

    def test_python_codegen(self):
        module = self._build_module("x IN 0..10")
        gen = CodeGenerator(Target.PYTHON)
        output = gen.generate(module)
        assert "def flux_check_test_mod(slots):" in output.code
        assert "return True" in output.code

    def test_codegen_output_metadata(self):
        module = self._build_module("x == 1")
        gen = CodeGenerator(Target.NATIVE)
        output = gen.generate(module)
        assert output.metadata["module_name"] == "test_mod"
        assert output.metadata["num_blocks"] == 1


# ═══════════════════════════════════════════════════════════
# End-to-End Pipeline Tests
# ═══════════════════════════════════════════════════════════


class TestEndToEnd:

    def test_full_pipeline_range(self):
        source = "x IN 0..100"
        tokens = FluxLexer(source).tokenize()
        prog = FluxParser(tokens).parse()
        module = IrBuilder("e2e").build(prog)
        opt = PeepholeOptimizer()
        opt.optimize(module)
        gen = CodeGenerator(Target.PYTHON)
        output = gen.generate(module)
        assert "def flux_check_e2e" in output.code

    def test_full_pipeline_complex(self):
        source = "(x IN 0..100 AND y == 42) OR NOT z == 0"
        tokens = FluxLexer(source).tokenize()
        prog = FluxParser(tokens).parse()
        module = IrBuilder("complex").build(prog)
        assert len(module.slot_map) == 3
        opt = PeepholeOptimizer()
        opt.optimize(module)
        gen = CodeGenerator(Target.NATIVE)
        output = gen.generate(module)
        assert "OR" in output.code

    def test_pipeline_with_nop_optimization(self):
        prog = _parse("x IN 0..10")
        module = IrBuilder("opt").build(prog)
        # Inject NOPs
        module.blocks[0].instructions.insert(0, IrInstruction.nop())
        module.blocks[0].instructions.insert(1, IrInstruction.nop())
        opt = PeepholeOptimizer()
        stats = opt.optimize(module)
        assert stats.nops_removed == 2
        assert not any(i.opcode == IrOpCode.NOP for i in module.all_instructions)

    def test_multiple_constraints(self):
        source = "x IN 0..10\ny IN 20..30\nz == 5"
        prog = _parse(source)
        assert len(prog.statements) == 3
        module = IrBuilder("multi").build(prog)
        assert len(module.slot_map) == 3
