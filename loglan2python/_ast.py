"""Abstract-syntax-tree (AST) node definitions for Loglan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union


# ---------------------------------------------------------------------------
# Expressions
# ---------------------------------------------------------------------------


@dataclass
class NumberLiteral:
    """A numeric literal, stored as its original source string."""

    raw: str

    @property
    def value(self) -> float:
        return float(self.raw)


@dataclass
class StringLiteral:
    value: str


@dataclass
class BoolLiteral:
    value: bool


@dataclass
class NullLiteral:
    pass


@dataclass
class Identifier:
    name: str


@dataclass
class IsNullExpr:
    operand: "Expr"


@dataclass
class BinaryOp:
    op: str  # '+', '-', '*', '/', '^', '**', '=', '<>', '<', '>', '<=', '>=', 'AND', 'OR'
    left: "Expr"
    right: "Expr"


@dataclass
class UnaryOp:
    op: str  # '-', 'NOT'
    operand: "Expr"


@dataclass
class FunctionCall:
    name: str
    args: List["Expr"]


Expr = Union[
    NumberLiteral,
    StringLiteral,
    BoolLiteral,
    NullLiteral,
    Identifier,
    IsNullExpr,
    BinaryOp,
    UnaryOp,
    FunctionCall,
]


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------


@dataclass
class Comment:
    text: str


@dataclass
class Assignment:
    target: str
    value: Expr


@dataclass
class SetNull:
    target: str


@dataclass
class IfStatement:
    condition: Expr
    then_body: List["Stmt"]
    elseif_clauses: List  # list of (condition, body) tuples
    else_body: Optional[List["Stmt"]]


@dataclass
class DoLoop:
    var: str
    start: Expr
    end: Expr
    step: Optional[Expr]
    body: List["Stmt"]


@dataclass
class WhileLoop:
    condition: Expr
    body: List["Stmt"]


@dataclass
class ReturnStmt:
    value: Optional[Expr]


@dataclass
class BreakStmt:
    pass


@dataclass
class ExprStatement:
    """A function call used as a statement."""

    expr: Expr


Stmt = Union[
    Comment,
    Assignment,
    SetNull,
    IfStatement,
    DoLoop,
    WhileLoop,
    ReturnStmt,
    BreakStmt,
    ExprStatement,
]


# ---------------------------------------------------------------------------
# Declarations and top-level structure
# ---------------------------------------------------------------------------


@dataclass
class VarDecl:
    kind: str  # 'PARAMETER', 'INPUT', 'OUTPUT', 'LOCAL'
    dtype: str  # 'REAL', 'INTEGER', 'STRING', 'LOGICAL'
    name: str
    default: Optional[Expr] = None


@dataclass
class Suite:
    index_name: str  # e.g. 'DEPTH'
    body: List[Stmt] = field(default_factory=list)


@dataclass
class Unit:
    name: str
    declarations: List[VarDecl] = field(default_factory=list)
    pre_suite: List[Stmt] = field(default_factory=list)
    suites: List[Suite] = field(default_factory=list)
    post_suite: List[Stmt] = field(default_factory=list)
