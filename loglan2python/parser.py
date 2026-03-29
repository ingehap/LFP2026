"""Recursive-descent parser for Loglan source code."""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

from .lexer import Token, TT, tokenize
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


class ParseError(Exception):
    """Raised when the parser encounters unexpected tokens."""


class Parser:
    """Recursive-descent parser that produces a :class:`Unit` AST."""

    def __init__(self, tokens: List[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    @property
    def cur(self) -> Token:
        """Return the token currently under the parser's read head."""
        return self._tokens[self._pos]

    def peek(self, offset: int = 1) -> Token:
        idx = self._pos + offset
        return self._tokens[idx] if idx < len(self._tokens) else self._tokens[-1]

    def advance(self) -> Token:
        t = self._tokens[self._pos]
        if self._pos < len(self._tokens) - 1:
            self._pos += 1
        return t

    def expect(self, *types: TT) -> Token:
        if self.cur.type not in types:
            got = self.cur
            names = " or ".join(tt.name for tt in types)
            raise ParseError(
                f"Line {got.line}:{got.col}: expected {names}, "
                f"got {got.type.name} ({got.value!r})"
            )
        return self.advance()

    def match(self, *types: TT) -> bool:
        if self.cur.type in types:
            self.advance()
            return True
        return False

    def skip_newlines(self) -> None:
        while self.cur.type in (TT.NEWLINE, TT.COMMENT):
            self.advance()

    def end_stmt(self) -> Optional[Comment]:
        """Consume an optional inline comment followed by newline(s)."""
        comment: Optional[Comment] = None
        if self.cur.type == TT.COMMENT:
            comment = Comment(self.advance().value)
        while self.cur.type == TT.NEWLINE:
            self.advance()
        return comment

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------

    def parse_unit(self) -> Unit:
        self.skip_newlines()
        self.expect(TT.UNIT)
        name_tok = self.expect(TT.IDENT)
        name = name_tok.value
        self.end_stmt()

        declarations: List[VarDecl] = []
        pre_suite: List[Stmt] = []
        suites: List[Suite] = []
        post_suite: List[Stmt] = []

        # Declarations (and optional pre-suite statements)
        while self.cur.type not in (TT.SUITE, TT.ENDUNIT, TT.EOF):
            decl = self._try_declaration()
            if decl is not None:
                declarations.append(decl)
            else:
                stmt = self._parse_stmt()
                if stmt is not None:
                    pre_suite.append(stmt)

        # SUITE blocks
        while self.cur.type == TT.SUITE:
            suites.append(self._parse_suite())

        # Optional post-suite statements
        while self.cur.type not in (TT.ENDUNIT, TT.EOF):
            stmt = self._parse_stmt()
            if stmt is not None:
                post_suite.append(stmt)

        self.expect(TT.ENDUNIT)
        self.end_stmt()

        return Unit(
            name=name,
            declarations=declarations,
            pre_suite=pre_suite,
            suites=suites,
            post_suite=post_suite,
        )

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------

    def _try_declaration(self) -> Optional[VarDecl]:
        """Try to parse a declaration; return *None* if not applicable.

        Comments and blank lines are intentionally left for ``_parse_stmt``
        so they end up in ``pre_suite`` and are preserved in the output.
        """
        t = self.cur
        # Comments and blank lines are handled by _parse_stmt(), not here.
        if t.type in (TT.COMMENT, TT.NEWLINE):
            return None
        if t.type == TT.PARAMETER:
            return self._parse_param_decl()
        if t.type == TT.INPUT:
            return self._parse_input_decl()
        if t.type == TT.OUTPUT:
            return self._parse_output_decl()
        if t.type in (TT.REAL, TT.INTEGER, TT.STRING_TYPE, TT.LOGICAL):
            return self._parse_local_decl()
        return None

    def _parse_param_decl(self) -> VarDecl:
        self.expect(TT.PARAMETER)
        dtype = self._parse_type_kw()
        name = self.expect(TT.IDENT).value
        default: Optional[Expr] = None
        if self.cur.type == TT.EQ:
            self.advance()
            default = self._parse_expr()
        self.end_stmt()
        return VarDecl(kind="PARAMETER", dtype=dtype, name=name, default=default)

    def _parse_input_decl(self) -> VarDecl:
        self.expect(TT.INPUT)
        name = self.expect(TT.IDENT).value
        self.expect(TT.COLON)
        dtype = self._parse_type_kw()
        self.end_stmt()
        return VarDecl(kind="INPUT", dtype=dtype, name=name)

    def _parse_output_decl(self) -> VarDecl:
        self.expect(TT.OUTPUT)
        name = self.expect(TT.IDENT).value
        self.expect(TT.COLON)
        dtype = self._parse_type_kw()
        self.end_stmt()
        return VarDecl(kind="OUTPUT", dtype=dtype, name=name)

    def _parse_local_decl(self) -> VarDecl:
        dtype = self._parse_type_kw()
        name = self.expect(TT.IDENT).value
        default: Optional[Expr] = None
        if self.cur.type == TT.EQ:
            self.advance()
            default = self._parse_expr()
        self.end_stmt()
        return VarDecl(kind="LOCAL", dtype=dtype, name=name, default=default)

    def _parse_type_kw(self) -> str:
        t = self.cur
        if t.type == TT.REAL:
            self.advance()
            return "REAL"
        if t.type == TT.INTEGER:
            self.advance()
            return "INTEGER"
        if t.type == TT.STRING_TYPE:
            self.advance()
            return "STRING"
        if t.type == TT.LOGICAL:
            self.advance()
            return "LOGICAL"
        raise ParseError(
            f"Line {t.line}: expected type keyword, got {t.type.name} ({t.value!r})"
        )

    # ------------------------------------------------------------------
    # Suite
    # ------------------------------------------------------------------

    def _parse_suite(self) -> Suite:
        self.expect(TT.SUITE)
        index_tok = self.expect(TT.IDENT)
        self.end_stmt()
        body = self._parse_stmt_list({TT.ENDSUITE})
        self.expect(TT.ENDSUITE)
        self.end_stmt()
        return Suite(index_name=index_tok.value, body=body)

    # ------------------------------------------------------------------
    # Statement list
    # ------------------------------------------------------------------

    def _parse_stmt_list(self, end_types: Set[TT]) -> List[Stmt]:
        stmts: List[Stmt] = []
        while self.cur.type not in end_types and self.cur.type != TT.EOF:
            stmt = self._parse_stmt()
            if stmt is not None:
                stmts.append(stmt)
        return stmts

    def _parse_stmt(self) -> Optional[Stmt]:
        t = self.cur
        if t.type == TT.NEWLINE:
            self.advance()
            return None
        if t.type == TT.COMMENT:
            text = self.advance().value
            while self.cur.type == TT.NEWLINE:
                self.advance()
            return Comment(text)
        if t.type == TT.IF:
            return self._parse_if()
        if t.type == TT.DO:
            return self._parse_do()
        if t.type == TT.WHILE:
            return self._parse_while()
        if t.type == TT.SETNULL:
            return self._parse_setnull()
        if t.type == TT.RETURN:
            return self._parse_return()
        if t.type == TT.BREAK:
            self.advance()
            self.end_stmt()
            return BreakStmt()
        if t.type == TT.IDENT:
            if self.peek().type == TT.EQ:
                return self._parse_assignment()
            return self._parse_expr_stmt()
        # Local type declarations inside a SUITE body (treated as no-op here).
        if t.type in (TT.REAL, TT.INTEGER, TT.STRING_TYPE, TT.LOGICAL):
            self._parse_local_decl()
            return None
        # Defensive: skip stray declaration keywords.
        if t.type in (TT.PARAMETER, TT.INPUT, TT.OUTPUT):
            while self.cur.type not in (TT.NEWLINE, TT.EOF):
                self.advance()
            self.end_stmt()
            return None
        raise ParseError(
            f"Line {t.line}:{t.col}: unexpected token {t.type.name} ({t.value!r})"
        )

    def _parse_assignment(self) -> Assignment:
        name = self.expect(TT.IDENT).value
        self.expect(TT.EQ)
        value = self._parse_expr()
        self.end_stmt()
        return Assignment(target=name, value=value)

    def _parse_setnull(self) -> SetNull:
        self.expect(TT.SETNULL)
        name = self.expect(TT.IDENT).value
        self.end_stmt()
        return SetNull(target=name)

    def _parse_return(self) -> ReturnStmt:
        self.expect(TT.RETURN)
        value: Optional[Expr] = None
        if self.cur.type not in (TT.NEWLINE, TT.EOF, TT.COMMENT):
            value = self._parse_expr()
        self.end_stmt()
        return ReturnStmt(value=value)

    def _parse_expr_stmt(self) -> ExprStatement:
        expr = self._parse_expr()
        self.end_stmt()
        return ExprStatement(expr=expr)

    # ------------------------------------------------------------------
    # Control flow
    # ------------------------------------------------------------------

    def _parse_if(self) -> IfStatement:
        self.expect(TT.IF)
        cond = self._parse_expr()
        self.expect(TT.THEN)
        self.end_stmt()
        then_body = self._parse_stmt_list({TT.ELSEIF, TT.ELSE, TT.ENDIF})
        elseif_clauses = []
        else_body: Optional[List[Stmt]] = None

        while self.cur.type == TT.ELSEIF:
            self.advance()
            ei_cond = self._parse_expr()
            self.expect(TT.THEN)
            self.end_stmt()
            ei_body = self._parse_stmt_list({TT.ELSEIF, TT.ELSE, TT.ENDIF})
            elseif_clauses.append((ei_cond, ei_body))

        if self.cur.type == TT.ELSE:
            self.advance()
            self.end_stmt()
            else_body = self._parse_stmt_list({TT.ENDIF})

        self.expect(TT.ENDIF)
        self.end_stmt()
        return IfStatement(
            condition=cond,
            then_body=then_body,
            elseif_clauses=elseif_clauses,
            else_body=else_body,
        )

    def _parse_do(self) -> DoLoop:
        self.expect(TT.DO)
        var = self.expect(TT.IDENT).value
        self.expect(TT.EQ)
        start = self._parse_expr()
        self.expect(TT.TO)
        end = self._parse_expr()
        step: Optional[Expr] = None
        if self.cur.type == TT.STEP:
            self.advance()
            step = self._parse_expr()
        self.end_stmt()
        body = self._parse_stmt_list({TT.ENDDO})
        self.expect(TT.ENDDO)
        self.end_stmt()
        return DoLoop(var=var, start=start, end=end, step=step, body=body)

    def _parse_while(self) -> WhileLoop:
        self.expect(TT.WHILE)
        cond = self._parse_expr()
        self.end_stmt()
        body = self._parse_stmt_list({TT.ENDWHILE})
        self.expect(TT.ENDWHILE)
        self.end_stmt()
        return WhileLoop(condition=cond, body=body)

    # ------------------------------------------------------------------
    # Expressions (recursive descent, respecting precedence)
    # ------------------------------------------------------------------

    def _parse_expr(self) -> Expr:
        return self._parse_or()

    def _parse_or(self) -> Expr:
        left = self._parse_and()
        while self.cur.type == TT.OR:
            self.advance()
            right = self._parse_and()
            left = BinaryOp("OR", left, right)
        return left

    def _parse_and(self) -> Expr:
        left = self._parse_not()
        while self.cur.type == TT.AND:
            self.advance()
            right = self._parse_not()
            left = BinaryOp("AND", left, right)
        return left

    def _parse_not(self) -> Expr:
        if self.cur.type == TT.NOT:
            self.advance()
            return UnaryOp("NOT", self._parse_not())
        return self._parse_comparison()

    _COMP_OPS = {TT.EQ, TT.NE, TT.LT, TT.GT, TT.LE, TT.GE}
    _COMP_MAP = {
        TT.EQ: "=",
        TT.NE: "<>",
        TT.LT: "<",
        TT.GT: ">",
        TT.LE: "<=",
        TT.GE: ">=",
    }

    def _parse_comparison(self) -> Expr:
        left = self._parse_addition()
        if self.cur.type in self._COMP_OPS:
            op = self._COMP_MAP[self.cur.type]
            self.advance()
            right = self._parse_addition()
            return BinaryOp(op, left, right)
        return left

    def _parse_addition(self) -> Expr:
        left = self._parse_multiplication()
        while self.cur.type in (TT.PLUS, TT.MINUS):
            op = "+" if self.cur.type == TT.PLUS else "-"
            self.advance()
            right = self._parse_multiplication()
            left = BinaryOp(op, left, right)
        return left

    def _parse_multiplication(self) -> Expr:
        left = self._parse_power()
        while self.cur.type in (TT.STAR, TT.SLASH):
            op = "*" if self.cur.type == TT.STAR else "/"
            self.advance()
            right = self._parse_power()
            left = BinaryOp(op, left, right)
        return left

    def _parse_power(self) -> Expr:
        base = self._parse_unary()
        if self.cur.type in (TT.CARET, TT.STARSTAR):
            self.advance()
            exp = self._parse_unary()
            return BinaryOp("**", base, exp)
        return base

    def _parse_unary(self) -> Expr:
        if self.cur.type == TT.MINUS:
            self.advance()
            return UnaryOp("-", self._parse_unary())
        if self.cur.type == TT.PLUS:
            self.advance()
            return self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> Expr:
        t = self.cur
        if t.type == TT.NUMBER:
            self.advance()
            return NumberLiteral(raw=t.value)
        if t.type == TT.STRING_LIT:
            self.advance()
            return StringLiteral(t.value)
        if t.type == TT.TRUE:
            self.advance()
            return BoolLiteral(True)
        if t.type == TT.FALSE:
            self.advance()
            return BoolLiteral(False)
        if t.type == TT.NULLVALUE:
            self.advance()
            return NullLiteral()
        if t.type == TT.ISNULL:
            self.advance()
            self.expect(TT.LPAREN)
            operand = self._parse_expr()
            self.expect(TT.RPAREN)
            return IsNullExpr(operand)
        if t.type == TT.LPAREN:
            self.advance()
            expr = self._parse_expr()
            self.expect(TT.RPAREN)
            return expr
        if t.type == TT.IDENT:
            name = t.value
            self.advance()
            if self.cur.type == TT.LPAREN:
                self.advance()
                args = []
                if self.cur.type != TT.RPAREN:
                    args.append(self._parse_expr())
                    while self.cur.type == TT.COMMA:
                        self.advance()
                        args.append(self._parse_expr())
                self.expect(TT.RPAREN)
                return FunctionCall(name=name, args=args)
            return Identifier(name=name)
        raise ParseError(
            f"Line {t.line}:{t.col}: unexpected token {t.type.name} "
            f"({t.value!r}) in expression"
        )


def parse(source: str) -> Unit:
    """Tokenise and parse Loglan *source* into a :class:`Unit` AST node."""
    tokens = tokenize(source)
    return Parser(tokens).parse_unit()
