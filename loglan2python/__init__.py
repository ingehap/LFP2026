"""Loglan → Python translator package.

Translate legacy `Loglan <https://www.aspentech.com/en/products/subsurface-science/geolog>`_
source files (Geolog scripting language) into idiomatic Python / NumPy.

Quick start::

    from loglan2python import translate, translate_file

    # from a string
    py_code = translate(loglan_source_text)

    # from a file (writes <name>.py alongside the input by default)
    py_code = translate_file("well.log", "well.py")
"""

from .translator import translate, translate_file

__all__ = ["translate", "translate_file"]
__version__ = "0.1.0"
