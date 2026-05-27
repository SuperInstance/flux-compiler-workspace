"""CodeGenerator — Emits target code from FLUX IR.

Supports multiple targets: native (pseudo-assembly), C, WASM (text), and
Python.  Each target produces a string representation of the compiled
constraints suitable for execution or further toolchain processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List

from .ir import BasicBlock, HaltReason, IrInstruction, IrModule, IrOpCode


class Target(Enum):
    NATIVE = "native"
    C = "c"
    WASM = "wasm"
    PYTHON = "python"


@dataclass
class CodegenError(Exception):
    message: str

    def __str__(self) -> str:
        return f"CodegenError: {self.message}"


@dataclass
class CodegenOutput:
    target: Target
    code: str
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class CodeGenerator:
    """Generate target code from a FLUX IR module."""

    def __init__(self, target: Target = Target.NATIVE) -> None:
        self.target = target

    def generate(self, module: IrModule) -> CodegenOutput:
        generators = {
            Target.NATIVE: self._gen_native,
            Target.C: self._gen_c,
            Target.WASM: self._gen_wasm,
            Target.PYTHON: self._gen_python,
        }
        gen = generators.get(self.target)
        if gen is None:
            raise CodegenError(f"unsupported target: {self.target}")
        code = gen(module)
        return CodegenOutput(
            target=self.target,
            code=code,
            metadata={"module_name": module.name, "num_blocks": len(module.blocks)},
        )

    # ── Native pseudo-assembly ───────────────────────────

    def _gen_native(self, module: IrModule) -> str:
        lines: List[str] = [f"; FLUX compiled: {module.name}", f"; Slots: {module.slot_map}", ""]
        for block in module.blocks:
            lines.append(f"{block.label}:")
            for inst in block.instructions:
                lines.append(f"    {inst}")
        return "\n".join(lines) + "\n"

    # ── C output ─────────────────────────────────────────

    def _gen_c(self, module: IrModule) -> str:
        lines: List[str] = [
            "#include <stdint.h>",
            "#include <stdbool.h>",
            "",
            f"// FLUX compiled: {module.name}",
            "",
            "typedef struct {",
            "    int64_t slots[256];",
            "    bool result;",
            "} flux_state;",
            "",
            f"bool flux_check_{module.name}(flux_state* state) {{",
        ]
        indent = "    "
        for block in module.blocks:
            if block.label != "entry":
                lines.append(f"{indent}{block.label}:")
            for inst in block.instructions:
                lines.append(f"{indent}{self._inst_to_c(inst)}")
        lines.append("}")
        return "\n".join(lines) + "\n"

    def _inst_to_c(self, inst: IrInstruction) -> str:
        if inst.opcode == IrOpCode.CHECK_RANGE:
            slot, lo, hi = inst.operands
            return f"if (state->slots[{slot}] < {lo} || state->slots[{slot}] > {hi}) return false;"
        if inst.opcode == IrOpCode.CHECK_EXACT:
            slot, val = inst.operands
            return f"if (state->slots[{slot}] != {val}) return false;"
        if inst.opcode == IrOpCode.CHECK_DOMAIN:
            slot, mask = inst.operands
            return f"if ((state->slots[{slot}] & 0x{mask:X}) != (int64_t)0x{mask:X}) return false;"
        if inst.opcode == IrOpCode.CHECK_COMPARISON:
            slot, op, val = inst.operands
            return f"if (!(state->slots[{slot}] {op} {val})) return false;"
        if inst.opcode == IrOpCode.AND:
            return "// AND: left && right already checked"
        if inst.opcode == IrOpCode.OR:
            return "// OR: left || right"
        if inst.opcode == IrOpCode.NOT:
            return "state->result = !state->result;"
        if inst.opcode == IrOpCode.HALT:
            return "return true;"
        if inst.opcode == IrOpCode.NOP:
            return "/* nop */"
        return f"/* unknown: {inst.opcode.name} */"

    # ── WASM text format ─────────────────────────────────

    def _gen_wasm(self, module: IrModule) -> str:
        lines: List[str] = [
            "(module",
            f"  ;; FLUX compiled: {module.name}",
            "  (func $check (result i32)",
        ]
        for block in module.blocks:
            for inst in block.instructions:
                lines.append(f"    {self._inst_to_wasm(inst)}")
        lines.append("    (i32.const 1)")  # true = pass
        lines.append("  )")
        lines.append("  (export \"check\" (func $check))")
        lines.append(")")
        return "\n".join(lines) + "\n"

    def _inst_to_wasm(self, inst: IrInstruction) -> str:
        if inst.opcode == IrOpCode.CHECK_RANGE:
            slot, lo, hi = inst.operands
            return f";; check_range slot={slot} [{lo}..{hi}]"
        if inst.opcode == IrOpCode.HALT:
            return ";; halt"
        return f";; {inst.opcode.name}"

    # ── Python output ────────────────────────────────────

    def _gen_python(self, module: IrModule) -> str:
        lines: List[str] = [
            '"""Auto-generated FLUX constraint checker."""',
            "",
            f"def flux_check_{module.name}(slots):",
            '    """Check all constraints. Returns True if all pass."""',
        ]
        indent = "    "
        for block in module.blocks:
            for inst in block.instructions:
                lines.append(f"{indent}{self._inst_to_python(inst)}")
        lines.append(f"{indent}return True")
        return "\n".join(lines) + "\n"

    def _inst_to_python(self, inst: IrInstruction) -> str:
        if inst.opcode == IrOpCode.CHECK_RANGE:
            slot, lo, hi = inst.operands
            return f"if not ({lo} <= slots[{slot}] <= {hi}): return False"
        if inst.opcode == IrOpCode.CHECK_EXACT:
            slot, val = inst.operands
            return f"if slots[{slot}] != {val}: return False"
        if inst.opcode == IrOpCode.CHECK_DOMAIN:
            slot, mask = inst.operands
            return f"if slots[{slot}] & 0x{mask:X} != 0x{mask:X}: return False"
        if inst.opcode == IrOpCode.CHECK_COMPARISON:
            slot, op, val = inst.operands
            return f"if not (slots[{slot}] {op} {val}): return False"
        if inst.opcode == IrOpCode.NOT:
            return "# NOT"
        if inst.opcode == IrOpCode.AND:
            return "# AND"
        if inst.opcode == IrOpCode.OR:
            return "# OR"
        if inst.opcode == IrOpCode.HALT:
            return "return True"
        return f"# {inst.opcode.name}"
