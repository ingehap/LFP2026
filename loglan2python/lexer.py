"""Tokeniser for Loglan (Geolog scripting language)."""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List


class TT(Enum):
    """Token types."""

    # Literals
    NUMBER = auto()
    STRING_LIT = auto()
    IDENT = auto()

    # Structure keywords
    UNIT = auto()
    ENDUNIT = auto()
    SUITE = auto()
    ENDSUITE = auto()

    # Declaration keywords
    PARAMETER = auto()
    INPUT = auto()
    OUTPUT = auto()

    # Type keywords
    REAL = auto()
    INTEGER = auto()
    STRING_TYPE = auto()
    LOGICAL = auto()

    # Control-flow keywords
    IF = auto()
    THEN = auto()
    ELSE = auto()
    ELSEIF = auto()
    ENDIF = auto()
    DO = auto()
    TO = auto()
    STEP = auto()
    ENDDO = auto()
    WHILE = auto()
    ENDWHILE = auto()
    BREAK = auto()
    RETURN = auto()

    # Null-handling keywords
    ISNULL = auto()
    SETNULL = auto()
    NULLVALUE = auto()

    # Logical operators
    AND = auto()
    OR = auto()
    NOT = auto()

    # Boolean constants
    TRUE = auto()
    FALSE = auto()

    # Arithmetic operators
    PLUS = auto()       # +
    MINUS = auto()      # -
    STAR = auto()       # *
    SLASH = auto()      # /
    CARET = auto()      # ^
    STARSTAR = auto()   # **

    # Relational operators
    EQ = auto()     # =
    NE = auto()     # <>
    LT = auto()     # <
    GT = auto()     # >
    LE = auto()     # <=
    GE = auto()     # >=

    # Punctuation
    LPAREN = auto()
    RPAREN = auto()
    COMMA = auto()
    COLON = auto()

    # Meta
    NEWLINE = auto()
    COMMENT = auto()
    EOF = auto()


_KEYWORDS: dict = {
    "UNIT": TT.UNIT,
    "ENDUNIT": TT.ENDUNIT,
    "SUITE": TT.SUITE,
    "ENDSUITE": TT.ENDSUITE,
    "PARAMETER": TT.PARAMETER,
    "PARAM": TT.PARAMETER,
    "INPUT": TT.INPUT,
    "OUTPUT": TT.OUTPUT,
    "REAL": TT.REAL,
    "INTEGER": TT.INTEGER,
    "INT": TT.INTEGER,
    "STRING": TT.STRING_TYPE,
    "LOGICAL": TT.LOGICAL,
    "BOOL": TT.LOGICAL,
    "IF": TT.IF,
    "THEN": TT.THEN,
    "ELSE": TT.ELSE,
    "ELSEIF": TT.ELSEIF,
    "ELIF": TT.ELSEIF,
    "ENDIF": TT.ENDIF,
    "DO": TT.DO,
    "TO": TT.TO,
    "STEP": TT.STEP,
    "ENDDO": TT.ENDDO,
    "WHILE": TT.WHILE,
    "ENDWHILE": TT.ENDWHILE,
    "BREAK": TT.BREAK,
    "RETURN": TT.RETURN,
    "ISNULL": TT.ISNULL,
    "SETNULL": TT.SETNULL,
    "NULLVALUE": TT.NULLVALUE,
    "NULL": TT.NULLVALUE,
    "AND": TT.AND,
    "OR": TT.OR,
    "NOT": TT.NOT,
    "TRUE": TT.TRUE,
    "FALSE": TT.FALSE,
}


@dataclass
class Token:
    """A single lexical token."""

    type: TT
    value: str
    line: int
    col: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.col})"


class LexerError(Exception):
    """Raised when the lexer encounters unexpected input."""


# Pattern list — order is significant (longer patterns first).
_PATTERNS = [
    ("NEWLINE", r"\n"),
    ("SKIP", r"[ \t\r]+"),
    ("COMMENT", r"![^\n]*"),
    ("STRING_LIT", r'"[^"]*"'),
    ("STARSTAR", r"\*\*"),
    ("NUMBER", r"[0-9]+(?:\.[0-9]+)?(?:[eE][+\-]?[0-9]+)?"),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("NE", r"<>"),
    ("LE", r"<="),
    ("GE", r">="),
    ("LT", r"<"),
    ("GT", r">"),
    ("EQ", r"="),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("STAR", r"\*"),
    ("SLASH", r"/"),
    ("CARET", r"\^"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("COMMA", r","),
    ("COLON", r":"),
]

_MASTER = re.compile(
    "|".join(f"(?P<{name}>{pat})" for name, pat in _PATTERNS),
    re.IGNORECASE,
)

_OP_TO_TT: dict = {
    "STARSTAR": TT.STARSTAR,
    "NE": TT.NE,
    "LE": TT.LE,
    "GE": TT.GE,
    "LT": TT.LT,
    "GT": TT.GT,
    "EQ": TT.EQ,
    "PLUS": TT.PLUS,
    "MINUS": TT.MINUS,
    "STAR": TT.STAR,
    "SLASH": TT.SLASH,
    "CARET": TT.CARET,
    "LPAREN": TT.LPAREN,
    "RPAREN": TT.RPAREN,
    "COMMA": TT.COMMA,
    "COLON": TT.COLON,
}


def tokenize(source: str) -> List[Token]:
    """Convert *source* Loglan text into a list of :class:`Token` objects."""
    tokens: List[Token] = []
    line = 1
    line_start = 0
    last_end = 0

    for m in _MASTER.finditer(source):
        start = m.start()
        if start > last_end:
            bad = source[last_end:start]
            raise LexerError(
                f"Unexpected characters at line {line}: {bad!r}"
            )
        last_end = m.end()
        kind = m.lastgroup
        raw = m.group()
        col = start - line_start + 1

        if kind == "SKIP":
            continue
        elif kind == "NEWLINE":
            tokens.append(Token(TT.NEWLINE, "\n", line, col))
            line += 1
            line_start = m.end()
        elif kind == "COMMENT":
            tokens.append(Token(TT.COMMENT, raw[1:].strip(), line, col))
        elif kind == "STRING_LIT":
            tokens.append(Token(TT.STRING_LIT, raw[1:-1], line, col))
        elif kind == "NUMBER":
            tokens.append(Token(TT.NUMBER, raw, line, col))
        elif kind == "IDENT":
            upper = raw.upper()
            tt = _KEYWORDS.get(upper, TT.IDENT)
            # Keep original case for identifiers; normalise keywords to upper.
            val = upper if tt != TT.IDENT else raw
            tokens.append(Token(tt, val, line, col))
        else:
            tokens.append(Token(_OP_TO_TT[kind], raw, line, col))

    # Catch any trailing characters that didn't match any pattern.
    if last_end < len(source):
        bad = source[last_end:]
        raise LexerError(f"Unexpected characters at line {line}: {bad!r}")

    tokens.append(Token(TT.EOF, "", line, 0))
    return tokens
