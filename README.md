# LFP2026 — Loglan → Python Translator

Translate legacy [Loglan](https://www.aspentech.com/en/products/subsurface-science/geolog)
source files — the scripting language embedded in **AspenTech Geolog** — into idiomatic
Python / NumPy.

## What is Loglan?

Loglan is the built-in scripting language of **Geolog**, a petrophysical interpretation
platform by AspenTech (formerly Paradigm).  Geolog programs operate on well-log *curves*:
depth-indexed arrays of measurements.  A typical Loglan program:

* declares input curves, output curves and scalar parameters,
* loops over every depth sample inside a `SUITE DEPTH` block, and
* applies petrophysical calculations at each sample.

## Feature mapping

| Loglan construct | Generated Python / NumPy |
|---|---|
| `UNIT name` / `ENDUNIT` | `def name(…):` |
| `PARAMETER REAL X = 1.0` | keyword argument `x: float = 1.0` |
| `INPUT GR : REAL` | positional argument `gr: np.ndarray` |
| `OUTPUT VSHALE : REAL` | `vshale = np.full(_n, np.nan)` + `return vshale` |
| `SUITE DEPTH` / `ENDSUITE` | `for _i in range(_n):` |
| `ISNULL(x)` | `np.isnan(x[_i])` |
| `SETNULL x` | `x[_i] = np.nan` |
| `IF / THEN / ELSEIF / ELSE / ENDIF` | `if / elif / else` |
| `DO i = 1 TO n` / `ENDDO` | `for i in range(1, n+1):` |
| `WHILE cond` / `ENDWHILE` | `while cond:` |
| `LOG`, `EXP`, `SQRT`, `ABS`, … | `np.log`, `np.exp`, `np.sqrt`, `abs`, … |
| `^` or `**` (power) | `**` |
| `<>` (not-equal) | `!=` |
| `AND` / `OR` / `NOT` | `and` / `or` / `not` |
| `! comment` | `# comment` |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command line

```bash
# Writes examples/shale_volume.py alongside the source
python -m loglan2python examples/shale_volume.log

# Specify an explicit output path
python -m loglan2python examples/water_saturation.log -o /tmp/water_sat.py
```

### Python API

```python
from loglan2python import translate, translate_file

# Translate a source string
py_code = translate(loglan_source_text)

# Translate a file (writes <name>.py alongside the input by default)
py_code = translate_file("well.log", "well.py")
```

## Examples

Three representative petrophysical calculations are provided in `examples/`:

| File | Description |
|---|---|
| `shale_volume.log` | Linear shale volume from gamma ray |
| `water_saturation.log` | Archie water saturation |
| `porosity.log` | Density-neutron crossplot total porosity |

Translate all examples at once:

```bash
for f in examples/*.log; do python -m loglan2python "$f"; done
```

## Running the tests

```bash
pip install pytest numpy
pytest tests/
```

## Project layout

```
loglan2python/
├── __init__.py      exposes translate() and translate_file()
├── lexer.py         tokeniser (regex-based)
├── _ast.py          AST node dataclasses
├── parser.py        recursive-descent parser
├── codegen.py       Python / NumPy code generator
├── translator.py    high-level API
└── __main__.py      CLI entry-point
examples/
├── shale_volume.log
├── water_saturation.log
└── porosity.log
tests/
└── test_translator.py
```