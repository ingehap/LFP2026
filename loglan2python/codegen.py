"""Generate Python (NumPy) source code from a Loglan AST."""

from __future__ import annotations

from io import StringIO
from typing import List, Optional, Set

from ._ast import (
    NumberLiteral,
    StringLiteral,
    BoolLiteral,
    NullLiteral,
    Identifier,
    IsNullExpr,
    BinaryOp,
    UnaryOp,
    FunctionCall,
    Comment,
    Assignment,
    SetNull,
    IfStatement,
    DoLoop,
    WhileLoop,
    ReturnStmt,
    BreakStmt,
    ExprStatement,
    VarDecl,
    Suite,
    Unit,
    Expr,
    Stmt,
)


# Loglan built-in function → Python/NumPy equivalent
_MATH_FUNCS: dict = {
    "LOG": "np.log",
    "LOG10": "np.log10",
    "EXP": "np.exp",
    "SQRT": "np.sqrt",
    "ABS": "abs",
    "SIN": "np.sin",
    "COS": "np.cos",
    "TAN": "np.tan",
    "ASIN": "np.arcsin",
    "ACOS": "np.arccos",
    "ATAN": "np.arctan",
    "ATAN2": "np.arctan2",
    "INT": "int",
    "FLOAT": "float",
    "MOD": "math.fmod",
    "MIN": "min",
    "MAX": "max",
    "SIGN": "np.sign",
}

_TYPE_HINTS: dict = {
    "REAL": "float",
    "INTEGER": "int",
    "STRING": "str",
    "LOGICAL": "bool",
}

_DEFAULT_INIT: dict = {
    "REAL": "0.0",
    "INTEGER": "0",
    "STRING": '""',
    "LOGICAL": "False",
}


class _Ctx:
    """Book-keeping for a single code-generation run."""

    def __init__(self, unit: Unit) -> None:
        self.inputs: Set[str] = {
            d.name.upper() for d in unit.declarations if d.kind == "INPUT"
        }
        self.outputs: Set[str] = {
            d.name.upper() for d in unit.declarations if d.kind == "OUTPUT"
        }
        self.params: Set[str] = {
            d.name.upper() for d in unit.declarations if d.kind == "PARAMETER"
        }
        self.locals: Set[str] = {
            d.name.upper() for d in unit.declarations if d.kind == "LOCAL"
        }
        self.in_suite: bool = False

    def is_array(self, name: str) -> bool:
        """Return *True* when *name* refers to an INPUT or OUTPUT curve."""
        return name.upper() in self.inputs or name.upper() in self.outputs


class Generator:
    """Emits Python source code from a parsed Loglan :class:`Unit`."""

    def __init__(self, unit: Unit) -> None:
        self._unit = unit
        self._ctx = _Ctx(unit)
        self._buf = StringIO()
        self._indent = 0

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _w(self, text: str = "") -> None:
        if text:
            self._buf.write("    " * self._indent)
            self._buf.write(text)
        self._buf.write("\n")

    def _inc(self) -> None:
        self._indent += 1

    def _dec(self) -> None:
        self._indent -= 1

    # ------------------------------------------------------------------
    # Expression generation
    # ------------------------------------------------------------------

    def _expr(self, node: Expr) -> str:
        if isinstance(node, NumberLiteral):
            return node.raw
        if isinstance(node, StringLiteral):
            return repr(node.value)
        if isinstance(node, BoolLiteral):
            return "True" if node.value else "False"
        if isinstance(node, NullLiteral):
            return "np.nan"
        if isinstance(node, Identifier):
            return self._ident(node.name)
        if isinstance(node, IsNullExpr):
            return f"np.isnan({self._expr(node.operand)})"
        if isinstance(node, BinaryOp):
            return self._binop(node)
        if isinstance(node, UnaryOp):
            operand = self._expr(node.operand)
            if node.op == "-":
                return f"(-{operand})"
            if node.op == "NOT":
                return f"(not {operand})"
        if isinstance(node, FunctionCall):
            return self._funcall(node)
        raise ValueError(f"Unknown expression node type: {type(node)}")

    def _ident(self, name: str) -> str:
        py = name.lower()
        if self._ctx.in_suite and self._ctx.is_array(name):
            return f"{py}[_i]"
        return py

    def _binop(self, node: BinaryOp) -> str:
        left = self._expr(node.left)
        right = self._expr(node.right)
        _op_map = {
            "+": "+",
            "-": "-",
            "*": "*",
            "/": "/",
            "^": "**",
            "**": "**",
            "=": "==",
            "<>": "!=",
            "<": "<",
            ">": ">",
            "<=": "<=",
            ">=": ">=",
            "AND": "and",
            "OR": "or",
        }
        py_op = _op_map.get(node.op, node.op)
        return f"({left} {py_op} {right})"

    def _funcall(self, node: FunctionCall) -> str:
        py_func = _MATH_FUNCS.get(node.name.upper(), node.name.lower())
        args = ", ".join(self._expr(a) for a in node.args)
        return f"{py_func}({args})"

    # ------------------------------------------------------------------
    # Statement generation
    # ------------------------------------------------------------------

    def _stmts(self, stmts: List[Stmt]) -> None:
        for s in stmts:
            self._stmt(s)

    def _stmt(self, node: Stmt) -> None:
        if isinstance(node, Comment):
            self._w(f"# {node.text}")
        elif isinstance(node, Assignment):
            self._assignment(node)
        elif isinstance(node, SetNull):
            self._setnull(node)
        elif isinstance(node, IfStatement):
            self._if(node)
        elif isinstance(node, DoLoop):
            self._do(node)
        elif isinstance(node, WhileLoop):
            self._while(node)
        elif isinstance(node, ReturnStmt):
            self._return(node)
        elif isinstance(node, BreakStmt):
            self._w("break")
        elif isinstance(node, ExprStatement):
            self._w(self._expr(node.expr))
        else:
            raise ValueError(f"Unknown statement node type: {type(node)}")

    def _assignment(self, node: Assignment) -> None:
        py = node.target.lower()
        val = self._expr(node.value)
        if self._ctx.in_suite and self._ctx.is_array(node.target):
            self._w(f"{py}[_i] = {val}")
        else:
            self._w(f"{py} = {val}")

    def _setnull(self, node: SetNull) -> None:
        py = node.target.lower()
        if self._ctx.in_suite and self._ctx.is_array(node.target):
            self._w(f"{py}[_i] = np.nan")
        else:
            self._w(f"{py} = np.nan")

    def _if(self, node: IfStatement) -> None:
        self._w(f"if {self._expr(node.condition)}:")
        self._inc()
        self._stmts(node.then_body) if node.then_body else self._w("pass")
        self._dec()
        for ei_cond, ei_body in node.elseif_clauses:
            self._w(f"elif {self._expr(ei_cond)}:")
            self._inc()
            self._stmts(ei_body) if ei_body else self._w("pass")
            self._dec()
        if node.else_body is not None:
            self._w("else:")
            self._inc()
            self._stmts(node.else_body) if node.else_body else self._w("pass")
            self._dec()

    def _do(self, node: DoLoop) -> None:
        var = node.var.lower()
        start = self._expr(node.start)
        end = self._expr(node.end)
        if node.step is not None:
            step = self._expr(node.step)
            self._w(f"for {var} in range(int({start}), int({end}) + 1, int({step})):")
        else:
            self._w(f"for {var} in range(int({start}), int({end}) + 1):")
        self._inc()
        self._stmts(node.body) if node.body else self._w("pass")
        self._dec()

    def _while(self, node: WhileLoop) -> None:
        self._w(f"while {self._expr(node.condition)}:")
        self._inc()
        self._stmts(node.body) if node.body else self._w("pass")
        self._dec()

    def _return(self, node: ReturnStmt) -> None:
        if node.value is not None:
            self._w(f"return {self._expr(node.value)}")
        else:
            self._w("return")

    # ------------------------------------------------------------------
    # Top-level code generation
    # ------------------------------------------------------------------

    def generate(self) -> str:
        unit = self._unit
        decls = unit.declarations
        params = [d for d in decls if d.kind == "PARAMETER"]
        inputs = [d for d in decls if d.kind == "INPUT"]
        outputs = [d for d in decls if d.kind == "OUTPUT"]
        locals_ = [d for d in decls if d.kind == "LOCAL"]

        # File-level header
        self._w('"""')
        self._w(f"Translated from Loglan unit ``{unit.name}`` by loglan2python.")
        self._w('"""')
        self._w("import math")
        self._w()
        self._w("import numpy as np")
        self._w()
        self._w()

        # Build function signature lines
        sig_lines: List[str] = []
        for d in inputs:
            sig_lines.append(f"    {d.name.lower()}: np.ndarray,")
        for d in params:
            hint = _TYPE_HINTS.get(d.dtype, "float")
            if d.default is not None:
                default_val = self._expr(d.default)
                sig_lines.append(f"    {d.name.lower()}: {hint} = {default_val},")
            else:
                sig_lines.append(f"    {d.name.lower()}: {hint},")

        # Return-type annotation
        if len(outputs) == 1:
            ret_type = "np.ndarray"
        elif len(outputs) > 1:
            ret_type = "tuple"
        else:
            ret_type = "None"

        func_name = unit.name.lower()
        if sig_lines:
            self._w(f"def {func_name}(")
            for line in sig_lines:
                self._w(line)
            self._w(f") -> {ret_type}:")
        else:
            self._w(f"def {func_name}() -> {ret_type}:")

        self._inc()

        # Docstring
        self._w('"""')
        self._w(f"Petrophysical calculation: {unit.name}.")
        if inputs or params:
            self._w()
            self._w("Parameters")
            self._w("----------")
            for d in inputs:
                self._w(f"{d.name.lower()} : np.ndarray")
                self._w(f"    Input curve ({d.dtype.lower()}).")
            for d in params:
                hint = _TYPE_HINTS.get(d.dtype, "float")
                if d.default is not None:
                    self._w(f"{d.name.lower()} : {hint}")
                    self._w(f"    Parameter (default {self._expr(d.default)}).")
                else:
                    self._w(f"{d.name.lower()} : {hint}")
                    self._w("    Parameter.")
        if outputs:
            self._w()
            self._w("Returns")
            self._w("-------")
            if len(outputs) == 1:
                d = outputs[0]
                self._w(f"{d.name.lower()} : np.ndarray")
                self._w(f"    Output curve ({d.dtype.lower()}).")
            else:
                for d in outputs:
                    self._w(f"{d.name.lower()} : np.ndarray")
                    self._w(f"    Output curve ({d.dtype.lower()}).")
        self._w('"""')

        # Length of depth array
        ref_arr: Optional[str] = None
        if inputs:
            ref_arr = inputs[0].name.lower()
        elif outputs:
            ref_arr = outputs[0].name.lower()
        if ref_arr:
            self._w(f"_n = len({ref_arr})")
            self._w()

        # Initialise output arrays
        for d in outputs:
            if ref_arr:
                self._w(f"{d.name.lower()} = np.full(_n, np.nan)")
            else:
                self._w(f"{d.name.lower()} = np.array([])")
        if outputs:
            self._w()

        # Initialise local variables
        for d in locals_:
            if d.default is not None:
                self._w(f"{d.name.lower()} = {self._expr(d.default)}")
            else:
                init = _DEFAULT_INIT.get(d.dtype, "0.0")
                self._w(f"{d.name.lower()} = {init}")
        if locals_:
            self._w()

        # Pre-suite statements
        if unit.pre_suite:
            self._stmts(unit.pre_suite)
            self._w()

        # SUITE blocks
        for suite in unit.suites:
            self._suite(suite)

        # Post-suite statements
        if unit.post_suite:
            self._stmts(unit.post_suite)
            self._w()

        # Return statement
        if len(outputs) == 0:
            pass
        elif len(outputs) == 1:
            self._w(f"return {outputs[0].name.lower()}")
        else:
            self._w("return " + ", ".join(d.name.lower() for d in outputs))

        self._dec()
        return self._buf.getvalue()

    def _suite(self, suite: Suite) -> None:
        self._w(f"# SUITE {suite.index_name}")
        self._w("for _i in range(_n):")
        self._inc()
        self._ctx.in_suite = True
        if suite.body:
            self._stmts(suite.body)
        else:
            self._w("pass")
        self._ctx.in_suite = False
        self._dec()
        self._w()


def generate(unit: Unit) -> str:
    """Generate Python source code from a parsed Loglan :class:`Unit` node."""
    return Generator(unit).generate()
