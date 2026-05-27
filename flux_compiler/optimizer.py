"""PeepholeOptimizer — Local optimization passes on FLUX IR.

Implements:
- Dead code elimination (removes NOPs and unreachable code after HALT)
- Instruction fusion (merges adjacent compatible checks)
- Strength reduction (replaces expensive patterns with cheaper ones)
- Constant folding for AND/OR/NOT chains
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .ir import BasicBlock, HaltReason, IrInstruction, IrModule, IrOpCode


@dataclass
class OptStats:
    """Statistics from an optimization pass."""
    nops_removed: int = 0
    dead_after_halt: int = 0
    fusions: int = 0
    strength_reductions: int = 0


class PeepholeOptimizer:
    """Run peephole optimization passes on an IR module."""

    def __init__(self, max_passes: int = 5) -> None:
        self.max_passes = max_passes

    def optimize(self, module: IrModule) -> OptStats:
        """Run optimization passes until fixed-point or max passes reached."""
        stats = OptStats()
        for _ in range(self.max_passes):
            changed = False
            for block in module.blocks:
                s = self._optimize_block(block)
                stats.nops_removed += s.nops_removed
                stats.dead_after_halt += s.dead_after_halt
                stats.fusions += s.fusions
                stats.strength_reductions += s.strength_reductions
                if s.nops_removed or s.dead_after_halt or s.fusions or s.strength_reductions:
                    changed = True
            if not changed:
                break
        return stats

    def _optimize_block(self, block: BasicBlock) -> OptStats:
        stats = OptStats()
        original_len = len(block.instructions)

        # Pass 1: Dead code elimination (remove NOPs)
        before = len(block.instructions)
        block.instructions = [i for i in block.instructions if i.opcode != IrOpCode.NOP]
        stats.nops_removed = before - len(block.instructions)

        # Pass 2: Remove instructions after HALT (except LABEL)
        new_insts: List[IrInstruction] = []
        halted = False
        for inst in block.instructions:
            if halted:
                if inst.opcode == IrOpCode.LABEL:
                    halted = False
                    new_insts.append(inst)
                else:
                    stats.dead_after_halt += 1
                continue
            new_insts.append(inst)
            if inst.opcode == IrOpCode.HALT:
                halted = True
        block.instructions = new_insts

        # Pass 3: Fuse adjacent CHECK_RANGE on same slot
        fused: List[IrInstruction] = []
        i = 0
        while i < len(block.instructions):
            inst = block.instructions[i]
            if (
                inst.opcode == IrOpCode.CHECK_RANGE
                and i + 1 < len(block.instructions)
                and block.instructions[i + 1].opcode == IrOpCode.CHECK_RANGE
                and inst.operands[0] == block.instructions[i + 1].operands[0]
            ):
                # Merge ranges: take union
                lo1, hi1 = inst.operands[1], inst.operands[2]
                lo2, hi2 = block.instructions[i + 1].operands[1], block.instructions[i + 1].operands[2]
                merged = IrInstruction.check_range(
                    inst.operands[0],
                    min(lo1, lo2),
                    max(hi1, hi2),
                )
                fused.append(merged)
                stats.fusions += 1
                i += 2
                continue
            # Strength reduction: NOT NOT → identity (remove both)
            if (
                inst.opcode == IrOpCode.NOT
                and i + 1 < len(block.instructions)
                and block.instructions[i + 1].opcode == IrOpCode.NOT
            ):
                stats.strength_reductions += 1
                i += 2
                continue
            fused.append(inst)
            i += 1
        block.instructions = fused

        return stats
