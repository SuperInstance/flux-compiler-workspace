"""FluxIR — Intermediate representation for the FLUX compiler.

Provides IR instructions, basic blocks, an IR builder, and module
structure.  The IR is the central representation that optimization
passes and code generators operate on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Union


# ── IR Instructions ─────────────────────────────────────────


class IrOpCode(Enum):
    CHECK_RANGE = auto()
    CHECK_DOMAIN = auto()
    CHECK_EXACT = auto()
    CHECK_COMPARISON = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    HALT = auto()
    NOP = auto()
    LOAD_CONST = auto()
    JUMP = auto()
    JUMP_IF_FALSE = auto()
    LABEL = auto()


class HaltReason(Enum):
    PASS = "pass"
    VIOLATION = "violation"
    UNREACHABLE = "unreachable"


@dataclass(frozen=True)
class IrInstruction:
    """A single IR instruction."""
    opcode: IrOpCode
    operands: tuple = ()

    # Convenience factory methods
    @staticmethod
    def check_range(slot: int, lo: int, hi: int) -> "IrInstruction":
        return IrInstruction(IrOpCode.CHECK_RANGE, (slot, lo, hi))

    @staticmethod
    def check_domain(slot: int, mask: int) -> "IrInstruction":
        return IrInstruction(IrOpCode.CHECK_DOMAIN, (slot, mask))

    @staticmethod
    def check_exact(slot: int, value: int) -> "IrInstruction":
        return IrInstruction(IrOpCode.CHECK_EXACT, (slot, value))

    @staticmethod
    def check_comparison(slot: int, op: str, value: Union[int, float]) -> "IrInstruction":
        return IrInstruction(IrOpCode.CHECK_COMPARISON, (slot, op, value))

    @staticmethod
    def and_() -> "IrInstruction":
        return IrInstruction(IrOpCode.AND)

    @staticmethod
    def or_() -> "IrInstruction":
        return IrInstruction(IrOpCode.OR)

    @staticmethod
    def not_() -> "IrInstruction":
        return IrInstruction(IrOpCode.NOT)

    @staticmethod
    def halt(reason: HaltReason = HaltReason.PASS) -> "IrInstruction":
        return IrInstruction(IrOpCode.HALT, (reason.value,))

    @staticmethod
    def nop() -> "IrInstruction":
        return IrInstruction(IrOpCode.NOP)

    @staticmethod
    def load_const(slot: int, value: Union[int, float]) -> "IrInstruction":
        return IrInstruction(IrOpCode.LOAD_CONST, (slot, value))

    @staticmethod
    def label(name: str) -> "IrInstruction":
        return IrInstruction(IrOpCode.LABEL, (name,))

    def __repr__(self) -> str:
        if self.operands:
            ops = ", ".join(repr(o) for o in self.operands)
            return f"{self.opcode.name}({ops})"
        return self.opcode.name


# ── Basic Block & Module ────────────────────────────────────


@dataclass
class BasicBlock:
    """A sequential block of IR instructions with a label."""
    label: str
    instructions: List[IrInstruction] = field(default_factory=list)

    def emit(self, inst: IrInstruction) -> None:
        self.instructions.append(inst)


@dataclass
class IrModule:
    """A complete IR module consisting of basic blocks."""
    name: str
    blocks: List[BasicBlock] = field(default_factory=list)
    slot_map: dict[str, int] = field(default_factory=dict)

    def add_block(self, label: str) -> BasicBlock:
        block = BasicBlock(label=label)
        self.blocks.append(block)
        return block

    @property
    def all_instructions(self) -> List[IrInstruction]:
        result: List[IrInstruction] = []
        for block in self.blocks:
            result.extend(block.instructions)
        return result


# ── IR Builder (lowers AST to IR) ──────────────────────────


from .parser import (
    AndExpr,
    ComparisonExpr,
    ConstraintExpr,
    DomainExpr,
    DomainMaskExpr,
    ExactExpr,
    GroupedExpr,
    NotExpr,
    OrExpr,
    Program,
    RangeExpr,
    SecurityExpr,
    TemporalExpr,
)


class IrBuilder:
    """Lower a parsed FLUX Program into an IrModule."""

    def __init__(self, module_name: str = "flux_main") -> None:
        self._module = IrModule(name=module_name)
        self._slot_counter = 0
        self._block = self._module.add_block("entry")

    def build(self, program: Program) -> IrModule:
        for stmt in program.statements:
            if isinstance(stmt, (RangeExpr, DomainExpr, DomainMaskExpr, ExactExpr,
                                 ComparisonExpr, AndExpr, OrExpr, NotExpr, GroupedExpr)):
                self._lower_constraint(stmt)
            elif isinstance(stmt, TemporalExpr):
                self._lower_constraint(stmt.constraint)
            elif isinstance(stmt, SecurityExpr):
                self._lower_security(stmt)
            # MetadataStmt: skip for IR
        self._block.emit(IrInstruction.halt(HaltReason.PASS))
        return self._module

    def _get_slot(self, name: str) -> int:
        if name not in self._module.slot_map:
            self._module.slot_map[name] = self._slot_counter
            self._slot_counter += 1
        return self._module.slot_map[name]

    def _lower_constraint(self, expr: ConstraintExpr) -> None:
        if isinstance(expr, RangeExpr):
            slot = self._get_slot(expr.name)
            self._block.emit(IrInstruction.check_range(slot, expr.lo, expr.hi))
        elif isinstance(expr, DomainExpr):
            slot = self._get_slot(expr.name)
            mask = _domain_values_to_mask(expr.values)
            self._block.emit(IrInstruction.check_domain(slot, mask))
        elif isinstance(expr, DomainMaskExpr):
            slot = self._get_slot(expr.name)
            self._block.emit(IrInstruction.check_domain(slot, expr.mask))
        elif isinstance(expr, ExactExpr):
            slot = self._get_slot(expr.name)
            val = expr.value if isinstance(expr.value, int) else hash(expr.value)
            self._block.emit(IrInstruction.check_exact(slot, val))
        elif isinstance(expr, ComparisonExpr):
            slot = self._get_slot(expr.name)
            self._block.emit(IrInstruction.check_comparison(slot, expr.op, expr.value))
        elif isinstance(expr, AndExpr):
            self._lower_constraint(expr.left)
            self._lower_constraint(expr.right)
            self._block.emit(IrInstruction.and_())
        elif isinstance(expr, OrExpr):
            self._lower_constraint(expr.left)
            self._lower_constraint(expr.right)
            self._block.emit(IrInstruction.or_())
        elif isinstance(expr, NotExpr):
            self._lower_constraint(expr.expr)
            self._block.emit(IrInstruction.not_())
        elif isinstance(expr, GroupedExpr):
            self._lower_constraint(expr.expr)

    def _lower_security(self, sec: SecurityExpr) -> None:
        # Security constraints become check_exact on a security slot
        if sec.check_type == "clearance":
            slot = self._get_slot(f"_security_clearance")
            self._block.emit(IrInstruction.check_exact(slot, hash(sec.value or "")))
        elif sec.check_type == "trusted" and sec.field_name:
            slot = self._get_slot(f"_security_trusted_{sec.field_name}")
            self._block.emit(IrInstruction.check_exact(slot, 1))


def _domain_values_to_mask(values: List[str]) -> int:
    """Convert domain string values to a bitmask (simple hash-based)."""
    mask = 0
    for v in values:
        mask |= (1 << (abs(hash(v)) % 64))
    return mask


# ── Convenience ─────────────────────────────────────────────

# Alias for backward compat; the original Rust crate uses FluxIR as the
# instruction enum name.
FluxIR = IrInstruction
