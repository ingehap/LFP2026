"""High-level API for Loglan → Python translation."""

from __future__ import annotations

from pathlib import Path

from .parser import parse
from .codegen import generate


def translate(source: str) -> str:
    """Translate a Loglan *source* string into Python source code."""
    unit = parse(source)
    return generate(unit)


def translate_file(
    input_path: "str | Path",
    output_path: "str | Path | None" = None,
) -> str:
    """Translate the Loglan file at *input_path*.

    If *output_path* is provided the generated Python is written there as
    well.  The generated source is always returned as a string.
    """
    source = Path(input_path).read_text(encoding="utf-8")
    py_code = translate(source)
    if output_path is not None:
        Path(output_path).write_text(py_code, encoding="utf-8")
    return py_code
