"""flux_compiler — FLUX constraint compiler toolchain (Python).

Provides lexer, parser, IR, optimizer, and code generation for the
FLUX constraint language used in the GUARD system.
"""

__version__ = "0.1.0"

from .lexer import FluxLexer, Token, TokenKind
from .parser import FluxParser
from .ir import FluxIR, IrInstruction, IrBuilder, IrModule
from .optimizer import PeepholeOptimizer
from .codegen import CodeGenerator, Target

__all__ = [
    "FluxLexer",
    "Token",
    "TokenKind",
    "FluxParser",
    "FluxIR",
    "IrInstruction",
    "IrBuilder",
    "IrModule",
    "PeepholeOptimizer",
    "CodeGenerator",
    "Target",
]
