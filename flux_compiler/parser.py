"""FluxParser — Builds an AST from tokenized FLUX source.

Grammar (simplified):
    program      ::= statement*
    statement    ::= constraint | metadata_stmt | security_stmt | temporal_stmt
    constraint   ::= field constraint_op
    constraint_op::= range_op | domain_op | exact_op | comparison_op
    range_op     ::= 'IN' number '..' number
    domain_op    ::= 'IN' '[' string (',' string)* ']'  |  'IN' hex_mask
    exact_op     ::= 'IS' string  |  '==' number
    comparison_op::= comp_op number
    grouped      ::= '(' expr (combinator expr)* ')'
    combinator   ::= 'AND' | 'OR'
    negation     ::= 'NOT' expr
    field        ::= IDENT ('.' IDENT)*
    temporal_stmt::= constraint 'FOR' 'duration' comp_op duration_value
    security_stmt::= 'SECURITY' ...
    metadata_stmt::= '@' IDENT STRING
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union

from .lexer import Token, TokenKind


# ── AST Node Types ──────────────────────────────────────────


@dataclass(frozen=True)
class RangeExpr:
    """field IN lo..hi"""
    name: str
    lo: int
    hi: int


@dataclass(frozen=True)
class DomainExpr:
    """field IN [v1, v2, ...]"""
    name: str
    values: List[str]


@dataclass(frozen=True)
class DomainMaskExpr:
    """field IN 0xNN"""
    name: str
    mask: int


@dataclass(frozen=True)
class ExactExpr:
    """field IS "value" or field == value"""
    name: str
    value: Union[str, int]


@dataclass(frozen=True)
class ComparisonExpr:
    """field op number"""
    name: str
    op: str
    value: Union[int, float]


@dataclass(frozen=True)
class AndExpr:
    left: "ConstraintExpr"
    right: "ConstraintExpr"


@dataclass(frozen=True)
class OrExpr:
    left: "ConstraintExpr"
    right: "ConstraintExpr"


@dataclass(frozen=True)
class NotExpr:
    expr: "ConstraintExpr"


@dataclass(frozen=True)
class GroupedExpr:
    expr: "ConstraintExpr"


@dataclass(frozen=True)
class TemporalExpr:
    constraint: "ConstraintExpr"
    duration_value: float
    duration_unit: str
    comparison_op: str
    tolerance_value: Optional[float] = None
    tolerance_unit: Optional[str] = None


@dataclass(frozen=True)
class SecurityExpr:
    check_type: str  # "clearance" or "trusted"
    field_name: Optional[str] = None
    op: Optional[str] = None
    value: Optional[str] = None


@dataclass(frozen=True)
class MetadataStmt:
    key: str
    value: str


ConstraintExpr = Union[
    RangeExpr,
    DomainExpr,
    DomainMaskExpr,
    ExactExpr,
    ComparisonExpr,
    AndExpr,
    OrExpr,
    NotExpr,
    GroupedExpr,
]


Statement = Union[
    ConstraintExpr,
    TemporalExpr,
    SecurityExpr,
    MetadataStmt,
]


@dataclass
class Program:
    statements: List[Statement] = field(default_factory=list)


# ── Parser Error ────────────────────────────────────────────


@dataclass
class ParseError(Exception):
    message: str
    pos: int = 0

    def __str__(self) -> str:
        return f"ParseError at position {self.pos}: {self.message}"


# ── Parser ──────────────────────────────────────────────────


class FluxParser:
    """Recursive-descent parser for the FLUX constraint language."""

    def __init__(self, tokens: List[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def parse(self) -> Program:
        stmts: List[Statement] = []
        while not self._at_end():
            stmt = self._parse_statement()
            if stmt is not None:
                stmts.append(stmt)
        return Program(statements=stmts)

    # ── Statement dispatch ───────────────────────────────

    def _parse_statement(self) -> Optional[Statement]:
        tok = self._peek()
        if tok is None:
            return None
        if tok.kind == TokenKind.AT:
            return self._parse_metadata()
        if tok.kind == TokenKind.SECURITY:
            return self._parse_security()
        if tok.kind == TokenKind.NOT:
            expr = self._parse_constraint_expr()
            # Check for temporal
            if self._check(TokenKind.FOR):
                return self._parse_temporal(expr)
            return expr
        if tok.kind == TokenKind.LPAREN:
            expr = self._parse_constraint_expr()
            if self._check(TokenKind.FOR):
                return self._parse_temporal(expr)
            return expr
        if tok.kind == TokenKind.IDENT:
            expr = self._parse_constraint_expr()
            if self._check(TokenKind.FOR):
                return self._parse_temporal(expr)
            return expr
        raise ParseError(f"unexpected token: {tok.value!r}", tok.pos)

    # ── Constraint expressions ───────────────────────────

    def _parse_constraint_expr(self) -> ConstraintExpr:
        left = self._parse_unary()
        while self._check(TokenKind.AND) or self._check(TokenKind.OR):
            combinator = self._advance()
            right = self._parse_unary()
            if combinator.kind == TokenKind.AND:
                left = AndExpr(left=left, right=right)
            else:
                left = OrExpr(left=left, right=right)
        return left

    def _parse_unary(self) -> ConstraintExpr:
        if self._check(TokenKind.NOT):
            self._advance()
            expr = self._parse_unary()
            return NotExpr(expr=expr)
        if self._check(TokenKind.LPAREN):
            return self._parse_grouped()
        return self._parse_primary_constraint()

    def _parse_grouped(self) -> ConstraintExpr:
        self._expect(TokenKind.LPAREN)
        expr = self._parse_constraint_expr()
        self._expect(TokenKind.RPAREN)
        return GroupedExpr(expr=expr)

    def _parse_primary_constraint(self) -> ConstraintExpr:
        name = self._parse_field_name()

        tok = self._peek()
        if tok is None:
            raise ParseError("unexpected end of input after field name")

        # IS "value"
        if tok.kind == TokenKind.IS:
            self._advance()
            val_tok = self._expect_one_of(TokenKind.STRING, TokenKind.INT)
            if val_tok.kind == TokenKind.STRING:
                return ExactExpr(name=name, value=val_tok.value)
            return ExactExpr(name=name, value=int(val_tok.value))

        # IN (range | domain | domain mask)
        if tok.kind == TokenKind.IN:
            self._advance()
            return self._parse_in_constraint(name)

        # Comparison operators
        comp_ops = {
            TokenKind.EQ: "==",
            TokenKind.NEQ: "!=",
            TokenKind.GTE: ">=",
            TokenKind.LTE: "<=",
            TokenKind.GT: ">",
            TokenKind.LT: "<",
        }
        if tok.kind in comp_ops:
            self._advance()
            num_tok = self._expect_one_of(TokenKind.INT, TokenKind.FLOAT)
            val: Union[int, float] = float(num_tok.value) if num_tok.kind == TokenKind.FLOAT else int(num_tok.value)
            return ComparisonExpr(name=name, op=comp_ops[tok.kind], value=val)

        raise ParseError(f"expected constraint operator after field {name!r}, got {tok.value!r}", tok.pos)

    def _parse_in_constraint(self, name: str) -> ConstraintExpr:
        tok = self._peek()
        if tok is None:
            raise ParseError("unexpected end of input after IN")

        # Hex mask: 0xNN
        if tok.kind == TokenKind.INT and tok.value.startswith("0x"):
            # Actually our lexer won't produce hex. Let's handle INT that looks hex-ish
            # or handle the common case of a bare number followed by ..
            pass

        # Range: number .. number
        if tok.kind == TokenKind.INT or tok.kind == TokenKind.FLOAT:
            lo_str = tok.value
            lo: Union[int, float] = float(lo_str) if "." in lo_str else int(lo_str)
            self._advance()
            self._expect(TokenKind.DOTDOT)
            hi_tok = self._expect_one_of(TokenKind.INT, TokenKind.FLOAT)
            hi = float(hi_tok.value) if "." in hi_tok.value else int(hi_tok.value)
            return RangeExpr(name=name, lo=int(lo), hi=int(hi))

        # Domain list: [v1, v2, ...]
        if tok.kind == TokenKind.LBRACKET:
            self._advance()
            values: List[str] = []
            if not self._check(TokenKind.RBRACKET):
                val_tok = self._expect_one_of(TokenKind.STRING, TokenKind.INT, TokenKind.IDENT)
                values.append(val_tok.value)
                while self._check(TokenKind.COMMA):
                    self._advance()
                    val_tok = self._expect_one_of(TokenKind.STRING, TokenKind.INT, TokenKind.IDENT)
                    values.append(val_tok.value)
            self._expect(TokenKind.RBRACKET)
            return DomainExpr(name=name, values=values)

        raise ParseError(f"expected range or domain after IN, got {tok.value!r}", tok.pos)

    def _parse_field_name(self) -> str:
        tok = self._expect(TokenKind.IDENT)
        name = tok.value
        while self._check(TokenKind.DOT):
            self._advance()
            part = self._expect(TokenKind.IDENT)
            name = f"{name}.{part.value}"
        return name

    # ── Temporal ─────────────────────────────────────────

    def _parse_temporal(self, constraint: ConstraintExpr) -> TemporalExpr:
        self._expect(TokenKind.FOR)
        # Accept 'duration' keyword or skip it
        if self._check(TokenKind.DURATION):
            self._advance()
        comp_tok = self._peek()
        comp_ops = {
            TokenKind.EQ: "==",
            TokenKind.NEQ: "!=",
            TokenKind.GTE: ">=",
            TokenKind.LTE: "<=",
            TokenKind.GT: ">",
            TokenKind.LT: "<",
        }
        if comp_tok and comp_tok.kind in comp_ops:
            self._advance()
            dur_val, dur_unit = self._parse_duration_value()
            tolerance_val = None
            tolerance_unit = None
            if self._check(TokenKind.WITH):
                self._advance()
                if self._check(TokenKind.TOLERANCE):
                    self._advance()
                tol_comp = self._peek()
                if tol_comp and tol_comp.kind in comp_ops:
                    self._advance()
                tolerance_val, tolerance_unit = self._parse_duration_value()
            return TemporalExpr(
                constraint=constraint,
                comparison_op=comp_ops[comp_tok.kind],
                duration_value=dur_val,
                duration_unit=dur_unit,
                tolerance_value=tolerance_val,
                tolerance_unit=tolerance_unit,
            )
        raise ParseError("expected comparison operator in temporal constraint", self._cur_pos())

    def _parse_duration_value(self) -> tuple[float, str]:
        num_tok = self._expect_one_of(TokenKind.INT, TokenKind.FLOAT)
        val = float(num_tok.value) if num_tok.kind == TokenKind.FLOAT else float(num_tok.value)
        unit_tok = self._expect(TokenKind.IDENT)
        return val, unit_tok.value

    # ── Security ─────────────────────────────────────────

    def _parse_security(self) -> SecurityExpr:
        self._expect(TokenKind.SECURITY)
        if self._check(TokenKind.CLEARANCE):
            self._advance()
            comp_tok = self._peek()
            comp_ops = {
                TokenKind.EQ: "==",
                TokenKind.NEQ: "!=",
                TokenKind.GTE: ">=",
                TokenKind.LTE: "<=",
                TokenKind.GT: ">",
                TokenKind.LT: "<",
            }
            op = ""
            if comp_tok and comp_tok.kind in comp_ops:
                self._advance()
                op = comp_ops[comp_tok.kind]
            val_tok = self._expect(TokenKind.STRING)
            return SecurityExpr(check_type="clearance", op=op, value=val_tok.value)
        # field TRUSTED
        name = self._parse_field_name()
        self._expect(TokenKind.TRUSTED)
        return SecurityExpr(check_type="trusted", field_name=name)

    # ── Metadata ─────────────────────────────────────────

    def _parse_metadata(self) -> MetadataStmt:
        self._expect(TokenKind.AT)
        key_tok = self._expect(TokenKind.IDENT)
        val_tok = self._expect(TokenKind.STRING)
        return MetadataStmt(key=key_tok.value, value=val_tok.value)

    # ── Token helpers ────────────────────────────────────

    def _peek(self) -> Optional[Token]:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _advance(self) -> Token:
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _at_end(self) -> bool:
        return self._pos >= len(self._tokens) or self._tokens[self._pos].kind == TokenKind.EOF

    def _check(self, kind: TokenKind) -> bool:
        tok = self._peek()
        return tok is not None and tok.kind == kind

    def _expect(self, kind: TokenKind) -> Token:
        tok = self._peek()
        if tok is None or tok.kind != kind:
            got = tok.value if tok else "EOF"
            raise ParseError(f"expected {kind.name}, got {got!r}", self._cur_pos())
        return self._advance()

    def _expect_one_of(self, *kinds: TokenKind) -> Token:
        tok = self._peek()
        if tok is None or tok.kind not in kinds:
            got = tok.value if tok else "EOF"
            expected = " or ".join(k.name for k in kinds)
            raise ParseError(f"expected {expected}, got {got!r}", self._cur_pos())
        return self._advance()

    def _cur_pos(self) -> int:
        tok = self._peek()
        return tok.pos if tok else 0
