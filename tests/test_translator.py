"""Tests for the loglan2python translator."""

import math

import numpy as np
import pytest

from loglan2python.lexer import LexerError, TT, tokenize
from loglan2python.parser import ParseError, parse
from loglan2python.codegen import generate
from loglan2python import translate, translate_file


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exec(source: str) -> dict:
    """Translate Loglan *source* and exec the result.  Return the namespace."""
    py_code = translate(source)
    ns: dict = {}
    exec(compile(py_code, "<translated>", "exec"), ns)
    return ns


# ---------------------------------------------------------------------------
# Shared Loglan snippets
# ---------------------------------------------------------------------------

_SIMPLE = """\
UNIT simple
  PARAMETER REAL X = 1.0
  INPUT  A : REAL
  OUTPUT B : REAL
  SUITE DEPTH
    B = A + X
  ENDSUITE
ENDUNIT
"""

_NULL = """\
UNIT nulltest
  INPUT  A : REAL
  OUTPUT B : REAL
  SUITE DEPTH
    IF ISNULL(A) THEN
      SETNULL B
    ELSE
      B = A * 2.0
    ENDIF
  ENDSUITE
ENDUNIT
"""


# ===========================================================================
# Lexer tests
# ===========================================================================

class TestLexer:
    def test_keywords_are_case_insensitive(self):
        tokens = tokenize("UNIT unit Unit")
        types = [t.type for t in tokens if t.type != TT.NEWLINE]
        assert types[:3] == [TT.UNIT, TT.UNIT, TT.UNIT]

    def test_integer_literal(self):
        tok = tokenize("42")[0]
        assert tok.type == TT.NUMBER
        assert tok.value == "42"

    def test_float_literal(self):
        tok = tokenize("3.14")[0]
        assert tok.type == TT.NUMBER
        assert tok.value == "3.14"

    def test_scientific_notation(self):
        tok = tokenize("1.5e-3")[0]
        assert tok.type == TT.NUMBER

    def test_comment_text_is_stripped(self):
        tok = tokenize("! this is a comment\n")[0]
        assert tok.type == TT.COMMENT
        assert tok.value == "this is a comment"

    def test_string_literal_strips_quotes(self):
        tok = tokenize('"hello world"')[0]
        assert tok.type == TT.STRING_LIT
        assert tok.value == "hello world"

    def test_ne_operator(self):
        assert tokenize("<>")[0].type == TT.NE

    def test_le_operator(self):
        assert tokenize("<=")[0].type == TT.LE

    def test_ge_operator(self):
        assert tokenize(">=")[0].type == TT.GE

    def test_starstar_operator(self):
        assert tokenize("**")[0].type == TT.STARSTAR

    def test_isnull_keyword(self):
        assert tokenize("ISNULL")[0].type == TT.ISNULL

    def test_setnull_keyword(self):
        assert tokenize("SETNULL")[0].type == TT.SETNULL

    def test_eof_always_last(self):
        tokens = tokenize("")
        assert tokens[-1].type == TT.EOF

    def test_unexpected_char_raises(self):
        with pytest.raises(LexerError):
            tokenize("@")

    def test_multiline_token_positions(self):
        tokens = tokenize("A\nB\n")
        newlines = [t for t in tokens if t.type == TT.NEWLINE]
        assert len(newlines) == 2
        assert newlines[0].line == 1
        assert newlines[1].line == 2


# ===========================================================================
# Parser tests
# ===========================================================================

class TestParser:
    def test_unit_name(self):
        unit = parse(_SIMPLE)
        assert unit.name == "simple"

    def test_parameter_declaration(self):
        unit = parse(_SIMPLE)
        params = [d for d in unit.declarations if d.kind == "PARAMETER"]
        assert len(params) == 1
        assert params[0].name == "X"
        assert params[0].dtype == "REAL"
        assert params[0].default is not None

    def test_input_declaration(self):
        unit = parse(_SIMPLE)
        inputs = [d for d in unit.declarations if d.kind == "INPUT"]
        assert len(inputs) == 1
        assert inputs[0].name == "A"

    def test_output_declaration(self):
        unit = parse(_SIMPLE)
        outputs = [d for d in unit.declarations if d.kind == "OUTPUT"]
        assert len(outputs) == 1
        assert outputs[0].name == "B"

    def test_suite_parsed(self):
        unit = parse(_SIMPLE)
        assert len(unit.suites) == 1
        assert unit.suites[0].index_name == "DEPTH"

    def test_if_statement(self):
        from loglan2python._ast import IfStatement
        unit = parse(_NULL)
        stmts = [s for s in unit.suites[0].body if isinstance(s, IfStatement)]
        assert len(stmts) == 1

    def test_elseif_clauses(self):
        src = """\
UNIT t
  INPUT A : REAL
  OUTPUT B : REAL
  SUITE DEPTH
    IF A < 0.0 THEN
      B = -1.0
    ELSEIF A = 0.0 THEN
      B = 0.0
    ELSE
      B = 1.0
    ENDIF
  ENDSUITE
ENDUNIT
"""
        from loglan2python._ast import IfStatement
        unit = parse(src)
        if_stmt = next(s for s in unit.suites[0].body if isinstance(s, IfStatement))
        assert len(if_stmt.elseif_clauses) == 1
        assert if_stmt.else_body is not None

    def test_do_loop(self):
        src = """\
UNIT t
  INPUT A : REAL
  OUTPUT B : REAL
  SUITE DEPTH
    B = 0.0
    DO I = 1 TO 3
      B = B + A
    ENDDO
  ENDSUITE
ENDUNIT
"""
        from loglan2python._ast import DoLoop
        unit = parse(src)
        stmts = [s for s in unit.suites[0].body if isinstance(s, DoLoop)]
        assert len(stmts) == 1
        assert stmts[0].var == "I"

    def test_while_loop(self):
        src = """\
UNIT t
  INPUT A : REAL
  OUTPUT B : REAL
  REAL CNT
  SUITE DEPTH
    B = A
    CNT = 0.0
    WHILE CNT < 3.0
      B = B * 2.0
      CNT = CNT + 1.0
    ENDWHILE
  ENDSUITE
ENDUNIT
"""
        from loglan2python._ast import WhileLoop
        unit = parse(src)
        stmts = [s for s in unit.suites[0].body if isinstance(s, WhileLoop)]
        assert len(stmts) == 1

    def test_comments_preserved(self):
        src = """\
UNIT t
  ! a comment
  INPUT A : REAL
  OUTPUT B : REAL
  SUITE DEPTH
    ! depth comment
    B = A
  ENDSUITE
ENDUNIT
"""
        from loglan2python._ast import Comment
        unit = parse(src)
        assert any(isinstance(s, Comment) for s in unit.suites[0].body)

    def test_missing_unit_name_raises(self):
        with pytest.raises(ParseError):
            parse("UNIT\n")

    def test_power_caret(self):
        from loglan2python._ast import BinaryOp
        src = """\
UNIT t
  INPUT A : REAL
  OUTPUT B : REAL
  SUITE DEPTH
    B = A ^ 2.0
  ENDSUITE
ENDUNIT
"""
        unit = parse(src)
        stmt = unit.suites[0].body[0]
        assert isinstance(stmt.value, BinaryOp)
        assert stmt.value.op == "**"

    def test_logical_and_or(self):
        from loglan2python._ast import BinaryOp, IfStatement
        src = """\
UNIT t
  INPUT A : REAL
  INPUT B : REAL
  OUTPUT C : REAL
  SUITE DEPTH
    IF ISNULL(A) OR ISNULL(B) THEN
      SETNULL C
    ELSE
      C = A + B
    ENDIF
  ENDSUITE
ENDUNIT
"""
        unit = parse(src)
        if_stmt = next(
            s for s in unit.suites[0].body if isinstance(s, IfStatement)
        )
        assert isinstance(if_stmt.condition, BinaryOp)
        assert if_stmt.condition.op == "OR"


# ===========================================================================
# Code-generation tests
# ===========================================================================

class TestCodegen:
    def test_function_definition(self):
        code = generate(parse(_SIMPLE))
        assert "def simple(" in code

    def test_numpy_import(self):
        code = generate(parse(_SIMPLE))
        assert "import numpy as np" in code

    def test_output_initialised_with_nan(self):
        code = generate(parse(_SIMPLE))
        assert "np.full(_n, np.nan)" in code

    def test_depth_loop(self):
        code = generate(parse(_SIMPLE))
        assert "for _i in range(_n):" in code

    def test_input_array_indexed(self):
        code = generate(parse(_SIMPLE))
        assert "a[_i]" in code

    def test_output_array_indexed(self):
        code = generate(parse(_SIMPLE))
        assert "b[_i]" in code

    def test_isnull_maps_to_np_isnan(self):
        code = generate(parse(_NULL))
        assert "np.isnan(a[_i])" in code

    def test_setnull_maps_to_nan(self):
        code = generate(parse(_NULL))
        assert "b[_i] = np.nan" in code

    def test_comment_preserved(self):
        src = """\
UNIT t
  INPUT A : REAL
  OUTPUT B : REAL
  SUITE DEPTH
    ! my comment
    B = A
  ENDSUITE
ENDUNIT
"""
        code = generate(parse(src))
        assert "# my comment" in code

    def test_parameter_becomes_keyword_arg(self):
        code = generate(parse(_SIMPLE))
        assert "x: float = 1.0" in code

    def test_multiple_outputs_return_tuple(self):
        src = """\
UNIT t
  INPUT  A : REAL
  OUTPUT B : REAL
  OUTPUT C : REAL
  SUITE DEPTH
    B = A * 2.0
    C = A + 10.0
  ENDSUITE
ENDUNIT
"""
        code = generate(parse(src))
        assert "return b, c" in code

    def test_caret_becomes_power(self):
        src = """\
UNIT t
  INPUT  A : REAL
  OUTPUT B : REAL
  SUITE DEPTH
    B = A ^ 2.0
  ENDSUITE
ENDUNIT
"""
        code = generate(parse(src))
        assert "**" in code

    def test_ne_becomes_not_equal(self):
        src = """\
UNIT t
  INPUT  A : REAL
  OUTPUT B : REAL
  SUITE DEPTH
    IF A <> 0.0 THEN
      B = 1.0
    ELSE
      B = 0.0
    ENDIF
  ENDSUITE
ENDUNIT
"""
        code = generate(parse(src))
        assert "!=" in code


# ===========================================================================
# Integration tests – translated code is actually executed
# ===========================================================================

class TestIntegration:
    def test_simple_addition(self):
        ns = _exec(_SIMPLE)
        a = np.array([1.0, 2.0, 3.0, np.nan])
        result = ns["simple"](a, x=0.5)
        np.testing.assert_array_almost_equal(result[:3], [1.5, 2.5, 3.5])

    def test_null_propagation(self):
        ns = _exec(_NULL)
        a = np.array([1.0, np.nan, 3.0])
        result = ns["nulltest"](a)
        assert result[0] == pytest.approx(2.0)
        assert np.isnan(result[1])
        assert result[2] == pytest.approx(6.0)

    def test_shale_volume_clipped(self):
        src = """\
UNIT loglan
  PARAMETER REAL GRMIN = 0.0
  PARAMETER REAL GRMAX = 100.0
  INPUT  GR     : REAL
  OUTPUT VSHALE : REAL
  SUITE DEPTH
    IF ISNULL(GR) THEN
      SETNULL VSHALE
    ELSE
      VSHALE = (GR - GRMIN) / (GRMAX - GRMIN)
      IF VSHALE > 1.0 THEN
        VSHALE = 1.0
      ELSEIF VSHALE < 0.0 THEN
        VSHALE = 0.0
      ENDIF
    ENDIF
  ENDSUITE
ENDUNIT
"""
        ns = _exec(src)
        gr = np.array([-10.0, 0.0, 50.0, 100.0, 150.0, np.nan])
        vsh = ns["loglan"](gr, grmin=0.0, grmax=100.0)
        assert vsh[0] == pytest.approx(0.0)   # clipped low
        assert vsh[1] == pytest.approx(0.0)
        assert vsh[2] == pytest.approx(0.5)
        assert vsh[3] == pytest.approx(1.0)
        assert vsh[4] == pytest.approx(1.0)   # clipped high
        assert np.isnan(vsh[5])               # null preserved

    def test_archie_water_saturation(self):
        src = """\
UNIT archie
  PARAMETER REAL A  = 1.0
  PARAMETER REAL M  = 2.0
  PARAMETER REAL N  = 2.0
  PARAMETER REAL RW = 0.05
  INPUT  PHI : REAL
  INPUT  RT  : REAL
  OUTPUT SW  : REAL
  SUITE DEPTH
    IF ISNULL(PHI) OR ISNULL(RT) THEN
      SETNULL SW
    ELSEIF PHI <= 0.0 OR RT <= 0.0 THEN
      SETNULL SW
    ELSE
      SW = (A * RW / (PHI^M * RT)) ^ (1.0 / N)
      IF SW > 1.0 THEN
        SW = 1.0
      ELSEIF SW < 0.0 THEN
        SW = 0.0
      ENDIF
    ENDIF
  ENDSUITE
ENDUNIT
"""
        ns = _exec(src)
        phi = np.array([0.2, 0.3, np.nan, 0.0])
        rt = np.array([10.0, 5.0, 10.0, 10.0])
        sw = ns["archie"](phi, rt)
        expected_0 = math.sqrt(0.05 / (0.2**2 * 10.0))
        assert sw[0] == pytest.approx(expected_0)
        assert not np.isnan(sw[1])
        assert np.isnan(sw[2])    # null propagated
        assert np.isnan(sw[3])    # phi <= 0 → null

    def test_do_loop(self):
        src = """\
UNIT sumloop
  INPUT  A : REAL
  OUTPUT B : REAL
  SUITE DEPTH
    B = 0.0
    DO I = 1 TO 3
      B = B + A
    ENDDO
  ENDSUITE
ENDUNIT
"""
        ns = _exec(src)
        a = np.array([1.0, 2.0])
        b = ns["sumloop"](a)
        np.testing.assert_array_almost_equal(b, [3.0, 6.0])

    def test_while_loop(self):
        src = """\
UNIT doubling
  INPUT  A : REAL
  OUTPUT B : REAL
  REAL CNT
  SUITE DEPTH
    B = A
    CNT = 0.0
    WHILE CNT < 3.0
      B = B * 2.0
      CNT = CNT + 1.0
    ENDWHILE
  ENDSUITE
ENDUNIT
"""
        ns = _exec(src)
        a = np.array([1.0, 2.0])
        b = ns["doubling"](a)
        np.testing.assert_array_almost_equal(b, [8.0, 16.0])

    def test_math_functions(self):
        src = """\
UNIT mathtest
  INPUT  A : REAL
  OUTPUT B : REAL
  OUTPUT C : REAL
  SUITE DEPTH
    B = LOG(A)
    C = SQRT(ABS(A))
  ENDSUITE
ENDUNIT
"""
        ns = _exec(src)
        a = np.array([1.0, math.e, 4.0])
        b, c = ns["mathtest"](a)
        np.testing.assert_array_almost_equal(b, [0.0, 1.0, math.log(4.0)])
        np.testing.assert_array_almost_equal(c, [1.0, math.sqrt(math.e), 2.0])

    def test_multiple_outputs(self):
        src = """\
UNIT multiout
  INPUT  A : REAL
  OUTPUT B : REAL
  OUTPUT C : REAL
  SUITE DEPTH
    B = A * 2.0
    C = A + 10.0
  ENDSUITE
ENDUNIT
"""
        ns = _exec(src)
        a = np.array([1.0, 2.0, 3.0])
        b, c = ns["multiout"](a)
        np.testing.assert_array_almost_equal(b, [2.0, 4.0, 6.0])
        np.testing.assert_array_almost_equal(c, [11.0, 12.0, 13.0])

    def test_power_operator(self):
        src = """\
UNIT powertest
  INPUT  A : REAL
  OUTPUT B : REAL
  SUITE DEPTH
    B = A ^ 2.0
  ENDSUITE
ENDUNIT
"""
        ns = _exec(src)
        a = np.array([2.0, 3.0, 4.0])
        b = ns["powertest"](a)
        np.testing.assert_array_almost_equal(b, [4.0, 9.0, 16.0])

    def test_elseif_branch(self):
        src = """\
UNIT elseiftest
  INPUT  A : REAL
  OUTPUT B : REAL
  SUITE DEPTH
    IF A < 0.0 THEN
      B = -1.0
    ELSEIF A = 0.0 THEN
      B = 0.0
    ELSE
      B = 1.0
    ENDIF
  ENDSUITE
ENDUNIT
"""
        ns = _exec(src)
        a = np.array([-5.0, 0.0, 5.0])
        b = ns["elseiftest"](a)
        np.testing.assert_array_almost_equal(b, [-1.0, 0.0, 1.0])

    def test_not_operator(self):
        src = """\
UNIT nottest
  INPUT  A : REAL
  OUTPUT B : REAL
  SUITE DEPTH
    IF NOT ISNULL(A) THEN
      B = A
    ELSE
      B = -999.0
    ENDIF
  ENDSUITE
ENDUNIT
"""
        ns = _exec(src)
        a = np.array([1.0, np.nan, 3.0])
        b = ns["nottest"](a)
        assert b[0] == pytest.approx(1.0)
        assert b[1] == pytest.approx(-999.0)
        assert b[2] == pytest.approx(3.0)

    def test_local_variable_scope(self):
        """LOCAL variables are scalars, not indexed arrays."""
        src = """\
UNIT localtest
  INPUT  A : REAL
  OUTPUT B : REAL
  REAL SCALE = 3.0
  SUITE DEPTH
    B = A * SCALE
  ENDSUITE
ENDUNIT
"""
        ns = _exec(src)
        a = np.array([1.0, 2.0, 4.0])
        b = ns["localtest"](a)
        np.testing.assert_array_almost_equal(b, [3.0, 6.0, 12.0])

    def test_translate_file(self, tmp_path):
        """translate_file() reads a file and writes the result."""
        src = tmp_path / "test.log"
        dst = tmp_path / "test.py"
        src.write_text(_SIMPLE, encoding="utf-8")
        code = translate_file(src, dst)
        assert dst.exists()
        assert "def simple(" in code
        assert dst.read_text(encoding="utf-8") == code

    def test_example_shale_volume(self):
        """End-to-end: translate and run examples/shale_volume.log."""
        from pathlib import Path
        log_path = Path(__file__).parent.parent / "examples" / "shale_volume.log"
        ns = _exec(log_path.read_text(encoding="utf-8"))
        gr = np.array([0.0, 75.0, 150.0, np.nan])
        vsh = ns["shale_volume"](gr, grmin=0.0, grmax=150.0)
        assert vsh[0] == pytest.approx(0.0)
        assert vsh[1] == pytest.approx(0.5)
        assert vsh[2] == pytest.approx(1.0)
        assert np.isnan(vsh[3])

    def test_example_water_saturation(self):
        """End-to-end: translate and run examples/water_saturation.log."""
        from pathlib import Path
        log_path = (
            Path(__file__).parent.parent / "examples" / "water_saturation.log"
        )
        ns = _exec(log_path.read_text(encoding="utf-8"))
        phi = np.array([0.2, np.nan, 0.0])
        rt = np.array([10.0, 10.0, 10.0])
        sw = ns["water_saturation"](phi, rt)
        assert sw[0] == pytest.approx(math.sqrt(0.05 / (0.04 * 10.0)))
        assert np.isnan(sw[1])
        assert np.isnan(sw[2])

    def test_example_porosity(self):
        """End-to-end: translate and run examples/porosity.log."""
        from pathlib import Path
        log_path = Path(__file__).parent.parent / "examples" / "porosity.log"
        ns = _exec(log_path.read_text(encoding="utf-8"))
        rhob = np.array([2.65, 2.15, np.nan])
        phin = np.array([0.0, 0.25, 0.20])
        phid, phit = ns["porosity"](rhob, phin)
        # rhob=2.65 → phid=(2.65-2.65)/(2.65-1.0)=0  → phit=(0+0)/2=0
        assert phid[0] == pytest.approx(0.0)
        assert phit[0] == pytest.approx(0.0)
        # rhob=2.15 → phid=(2.65-2.15)/1.65≈0.303
        assert phid[1] == pytest.approx(0.5 / 1.65, rel=1e-4)
        # rhob=nan → phid=nan → phit=nan
        assert np.isnan(phid[2])
        assert np.isnan(phit[2])
