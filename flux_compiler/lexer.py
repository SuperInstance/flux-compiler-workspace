"""FluxLexer — Tokenizes FLUX constraint source code.

Supports identifiers, integers, ranges, domain masks, boolean combinators
(AND, OR, NOT), comparison operators, parentheses, and temporal/security
keywords.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List


class TokenKind(Enum):
    # Literals
    IDENT = auto()
    INT = auto()
    FLOAT = auto()
    STRING = auto()

    # Keywords
    IN = auto()
    IS = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    FOR = auto()
    WITH = auto()
    SECURITY = auto()
    TRUSTED = auto()
    CLEARANCE = auto()
    TOLERANCE = auto()
    DURATION = auto()

    # Operators
    DOTDOT = auto()      # ..
    AT = auto()           # @
    EQ = auto()           # ==
    NEQ = auto()          # !=
    GTE = auto()          # >=
    LTE = auto()          # <=
    GT = auto()           # >
    LT = auto()           # <

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    DOT = auto()

    # Special
    EOF = auto()


KEYWORDS: dict[str, TokenKind] = {
    "in": TokenKind.IN,
    "is": TokenKind.IS,
    "and": TokenKind.AND,
    "or": TokenKind.OR,
    "not": TokenKind.NOT,
    "for": TokenKind.FOR,
    "with": TokenKind.WITH,
    "security": TokenKind.SECURITY,
    "trusted": TokenKind.TRUSTED,
    "clearance": TokenKind.CLEARANCE,
    "tolerance": TokenKind.TOLERANCE,
    "duration": TokenKind.DURATION,
}


@dataclass(frozen=True)
class Token:
    kind: TokenKind
    value: str
    pos: int = 0


@dataclass
class LexerError(Exception):
    message: str
    pos: int = 0

    def __str__(self) -> str:
        return f"LexerError at position {self.pos}: {self.message}"


class FluxLexer:
    """Tokenize a FLUX source string into a list of Token objects."""

    def __init__(self, source: str) -> None:
        self._source = source
        self._pos = 0
        self._tokens: List[Token] = []

    def tokenize(self) -> List[Token]:
        self._tokens = []
        self._pos = 0

        while self._pos < len(self._source):
            ch = self._source[self._pos]

            # Skip whitespace
            if ch in " \t\r\n":
                self._pos += 1
                continue

            # Skip comments (# to end of line)
            if ch == "#":
                while self._pos < len(self._source) and self._source[self._pos] != "\n":
                    self._pos += 1
                continue

            # String literals
            if ch == '"':
                self._read_string()
                continue

            # Numbers (int or float)
            if ch.isdigit() or (ch == "-" and self._peek_is_digit()):
                self._read_number()
                continue

            # Identifiers and keywords
            if ch.isalpha() or ch == "_":
                self._read_ident()
                continue

            # Two-character operators
            if self._pos + 1 < len(self._source):
                two = self._source[self._pos : self._pos + 2]
                if two == "..":
                    self._emit(TokenKind.DOTDOT, "..")
                    continue
                if two == "==":
                    self._emit(TokenKind.EQ, "==")
                    continue
                if two == "!=":
                    self._emit(TokenKind.NEQ, "!=")
                    continue
                if two == ">=":
                    self._emit(TokenKind.GTE, ">=")
                    continue
                if two == "<=":
                    self._emit(TokenKind.LTE, "<=")
                    continue

            # Single-character tokens
            single_map: dict[str, TokenKind] = {
                "(": TokenKind.LPAREN,
                ")": TokenKind.RPAREN,
                "[": TokenKind.LBRACKET,
                "]": TokenKind.RBRACKET,
                ",": TokenKind.COMMA,
                ".": TokenKind.DOT,
                "@": TokenKind.AT,
                ">": TokenKind.GT,
                "<": TokenKind.LT,
            }
            if ch in single_map:
                self._emit(single_map[ch], ch)
                continue

            raise LexerError(f"unexpected character: {ch!r}", self._pos)

        self._tokens.append(Token(kind=TokenKind.EOF, value="", pos=self._pos))
        return self._tokens

    # ── Helpers ──────────────────────────────────────────

    def _emit(self, kind: TokenKind, value: str) -> None:
        self._tokens.append(Token(kind=kind, value=value, pos=self._pos))
        self._pos += len(value)

    def _peek_is_digit(self) -> bool:
        return (
            self._pos + 1 < len(self._source)
            and self._source[self._pos + 1].isdigit()
        )

    def _read_string(self) -> None:
        start = self._pos
        self._pos += 1  # skip opening quote
        chars: list[str] = []
        while self._pos < len(self._source) and self._source[self._pos] != '"':
            if self._source[self._pos] == "\\" and self._pos + 1 < len(self._source):
                self._pos += 1
                chars.append(self._source[self._pos])
            else:
                chars.append(self._source[self._pos])
            self._pos += 1
        if self._pos >= len(self._source):
            raise LexerError("unterminated string literal", start)
        self._pos += 1  # skip closing quote
        self._tokens.append(Token(kind=TokenKind.STRING, value="".join(chars), pos=start))

    def _read_number(self) -> None:
        start = self._pos
        is_negative = False
        if self._source[self._pos] == "-":
            is_negative = True
            self._pos += 1

        digits: list[str] = []
        while self._pos < len(self._source) and self._source[self._pos].isdigit():
            digits.append(self._source[self._pos])
            self._pos += 1

        if (
            self._pos < len(self._source)
            and self._source[self._pos] == "."
            and self._pos + 1 < len(self._source)
            and self._source[self._pos + 1].isdigit()
        ):
            # Float
            digits.append(".")
            self._pos += 1
            while self._pos < len(self._source) and self._source[self._pos].isdigit():
                digits.append(self._source[self._pos])
                self._pos += 1
            value = ("-" if is_negative else "") + "".join(digits)
            self._tokens.append(Token(kind=TokenKind.FLOAT, value=value, pos=start))
        else:
            value = ("-" if is_negative else "") + "".join(digits)
            self._tokens.append(Token(kind=TokenKind.INT, value=value, pos=start))

    def _read_ident(self) -> None:
        start = self._pos
        chars: list[str] = []
        while self._pos < len(self._source) and (
            self._source[self._pos].isalnum() or self._source[self._pos] == "_"
        ):
            chars.append(self._source[self._pos])
            self._pos += 1
        text = "".join(chars)
        kind = KEYWORDS.get(text.lower(), TokenKind.IDENT)
        self._tokens.append(Token(kind=kind, value=text, pos=start))
