---
name: script-to-ipynb
description: Convert Python scripts (.py) to Jupyter notebooks (.ipynb)
license: MIT
compatibility: opencode
metadata:
  audience: developers
  use_case: notebook conversion
---

## What I do

I convert Python scripts into Jupyter notebooks by:
- Parsing Python scripts to identify logical sections
- Converting docstrings and comments to markdown cells
- Preserving code blocks as executable cells
- Generating properly formatted .ipynb JSON files

## When to use me

Use this when you need to:
- Convert a Python script to an interactive Jupyter notebook
- Create a notebook from existing code for documentation
- Transform analysis scripts into explorable notebooks

## How to use

The skill uses the `script_to_ipynb.py` module located in the project root.

### Python API

```python
import sys
sys.path.insert(0, 'D:/github/py/sphere_n')
from script_to_ipynb import convert_script

# Convert a script
result = convert_script("input.py", "output.ipynb", title="My Notebook")

# Or use the class directly
from script_to_ipynb import ScriptToNotebookConverter

converter = ScriptToNotebookConverter()
converter.convert("script.py", "notebook.ipynb")
```

### Requirements

```bash
pip install nbformat jupytext
```

### Features

- **Cell Detection**: Identifies imports, functions, classes, and main execution
- **Section Comments**: Recognizes `# ---`, `# ===`, and `###` as section headers
- **Docstrings**: Converts module docstrings to title markdown
- **UTF-8 Support**: Handles special characters properly
- **Custom Titles**: Optional notebook title parameter

### Example Output

The converter creates notebooks with:
- Title markdown cell (if title provided)
- Import statements in first code cell
- Configuration in separate cells
- Functions/classes with preserved docstrings
- Visualization code in its own cell

### Usage in OpenCode

When you need to convert a script:

1. Tell me which Python file to convert
2. Optionally specify output filename and title
3. I'll use the converter module to create the .ipynb file
