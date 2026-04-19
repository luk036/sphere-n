# How to Create an OpenCode Skill

This tutorial explains how to create a new skill for OpenCode, using the `script-to-ipynb` skill as a practical example.

## What is a Skill?

A skill is a reusable capability that can be invoked during an OpenCode session. Skills allow you to:
- Extend OpenCode's functionality with domain-specific tools
- Encapsulate complex operations into simple, callable units
- Share specialized knowledge across sessions

## Skill Structure

A skill consists of:

1. **SKILL.md** - Definition file in `.opencode/skills/<skill-name>/`
2. **Implementation** - Python module that provides the actual functionality

## Step-by-Step Guide

### Step 1: Create the Skill Directory

Create a directory structure:

```
.opencode/skills/<skill-name>/
```

For our example:
```
.opencode/skills/script-to-ipynb/
```

### Step 2: Create SKILL.md

The SKILL.md file defines your skill's metadata and documentation.

```yaml
---
name: script-to-ipynb
description: Convert Python scripts (.py) to Jupyter notebooks (.ipynb)
license: MIT
compatibility: opencode
metadata:
  audience: developers
  use_case: notebook conversion
---
```

#### Frontmatter Fields

| Field | Required | Description |
|-------|----------|-------------|
| `name` | Yes | Unique skill identifier (kebab-case) |
| `description` | Yes | Brief description of what the skill does |
| `license` | No | License for the skill |
| `compatibility` | Yes | Set to `opencode` |
| `metadata` | No | Additional metadata (audience, use_case) |

#### Documentation Sections

After the frontmatter, add:

```markdown
## What I do

Describe what the skill does in simple terms.

## When to use me

List use cases when this skill is appropriate.

## How to use

Explain the Python API and any requirements.

### Python API


### Requirements


### Features

- Feature 1
- Feature 2

### Example Output

Describe or show example output.
```

### Step 3: Implement the Skill

Create a Python module that implements the skill's functionality. This typically goes in your project root or a convenient location.

#### Example: script_to_ipynb.py

```python
#!/usr/bin/env python
"""Script to Jupyter Notebook Converter Skill."""

import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import nbformat
    from nbformat import v4 as nbf
except ImportError:
    nbformat = None


class ScriptToNotebookConverter:
    """Converter class for transforming Python scripts to Jupyter notebooks."""

    def __init__(
        self,
        include_outputs: bool = False,
        kernel_name: str = "python3",
        python_version: str = "3.8.0",
    ) -> None:
        self.include_outputs = include_outputs
        self.kernel_name = kernel_name
        self.python_version = python_version

    def parse_script(self, script_path: str) -> List[Tuple[str, str]]:
        """Parse a Python script and extract cells."""
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.parse_content(content)

    def parse_content(self, content: str) -> List[Tuple[str, str]]:
        """Parse Python script content into cells."""
        cells: List[Tuple[str, str]] = []
        lines = content.split("\n")

        # Implementation details...

        return cells

    def create_notebook(self, cells: List[Tuple[str, str]]) -> dict:
        """Create a notebook structure from cells."""
        if nbformat is None:
            raise ImportError("nbformat is required.")

        nb = nbf.new_notebook()
        for cell_type, content in cells:
            if cell_type == "markdown":
                nb.cells.append(nbf.new_markdown_cell(content))
            else:
                cell = nbf.new_code_cell(content)
                if not self.include_outputs:
                    cell["outputs"] = []
                cell["execution_count"] = None
                nb.cells.append(cell)
        return nb

    def convert(
        self,
        script_path: str,
        output_path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> str:
        """Convert a Python script to a Jupyter notebook."""
        script_path = Path(script_path)
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        if output_path is None:
            output_path = str(script_path.with_suffix(".ipynb"))
        else:
            output_path = str(output_path)

        cells = self.parse_script(str(script_path))
        if title:
            cells.insert(0, ("markdown", f"# {title}"))

        nb = self.create_notebook(cells)
        with open(output_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        return output_path


def convert_script(
    script_path: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """Convenience function to convert a script to notebook."""
    converter = ScriptToNotebookConverter()
    return converter.convert(script_path, output_path, title)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m script_to_ipynb <script.py> [output.ipynb]")
        sys.exit(1)
    script_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    result = convert_script(script_path, output_path)
    print(f"Created notebook: {result}")
```

### Step 4: Key Implementation Patterns

#### Cell Detection Logic

The core of a notebook converter is deciding what goes into each cell:

```python
def _is_section_comment(self, line: str) -> bool:
    """Check if line is a section comment."""
    stripped = line.strip()
    return (
        stripped.startswith("# ")
        and (
            stripped.endswith("---")
            or stripped.endswith("===")
        )
    ) or (stripped.startswith("###") and len(stripped) > 3)

def _is_standalone_comment(self, line: str, lines: List[str], index: int) -> bool:
    """Check if comment should be converted to markdown cell.

    Only top-level comments (indentation 0) become markdown cells.
    """
    stripped = line.strip()

    # Must be a comment at column 0
    if not stripped.startswith("#"):
        return False

    # Must be at indentation level 0 (not inside a function/class)
    leading_spaces = len(line) - len(line.lstrip())
    if leading_spaces > 0:
        return False

    # Skip delimiters and section comments
    if self._is_cell_delimiter(line):
        return False
    if self._is_section_comment(line):
        return False

    # Check if followed by code
    has_code_after = False
    for j in range(index + 1, len(lines)):
        next_line = lines[j].strip()
        if not next_line:
            continue
        if next_line.startswith("#"):
            continue
        has_code_after = True
        break

    return has_code_after
```

#### Creating Notebook Cells

```python
def create_notebook(self, cells: List[Tuple[str, str]]) -> dict:
    """Create notebook structure from cells."""
    nb = nbf.new_notebook()

    for cell_type, content in cells:
        if cell_type == "markdown":
            nb.cells.append(nbf.new_markdown_cell(content))
        else:
            cell = nbf.new_code_cell(content)
            cell["outputs"] = []
            cell["execution_count"] = None
            nb.cells.append(cell)

    return nb
```

### Step 5: Testing Your Skill

#### Manual Testing

```python
import sys
sys.path.insert(0, 'D:/path/to/your/project')
from script_to_ipynb import convert_script

# Test conversion
result = convert_script("input.py", "output.ipynb", title="Test")
print(f"Created: {result}")
```

#### Validation

Always validate generated notebooks:

```python
import json

with open('output.ipynb') as f:
    nb = json.load(f)

# Check cell count
print(f"Cells: {len(nb['cells'])}")

# Verify syntax
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        try:
            compile(source, f'cell_{i}', 'exec')
            print(f"Cell {i}: OK")
        except SyntaxError as e:
            print(f"Cell {i}: SYNTAX ERROR - {e}")
```

### Step 6: Using Your Skill in OpenCode

Once created, skills can be invoked in OpenCode sessions:

```python
# Load the skill
skill(name="script-to-ipynb")

# Use it via the skill tool
# The skill description in SKILL.md guides how it's used
```

## Best Practices

1. **Keep SKILL.md concise** - Focus on what the skill does and when to use it
2. **Handle dependencies gracefully** - Use try/except for optional imports
3. **Validate output** - Always verify generated notebooks compile correctly
4. **Provide clear error messages** - Help users understand what went wrong
5. **Support both CLI and API** - Allow both command-line and programmatic use

## Common Issues

### Issue: Comments Inside Classes Become Markdown

**Problem**: Comments inside functions/classes are incorrectly converted to markdown cells.

**Solution**: Check indentation level - only comments at indentation 0 should become markdown:

```python
leading_spaces = len(line) - len(line.lstrip())
if leading_spaces > 0:
    return False
```

### Issue: Broken Code Cells

**Problem**: Code is split across cells incorrectly, breaking Python syntax.

**Solution**: Ensure complete code blocks stay together. Don't split on every comment.

## Summary

To create an OpenCode skill:

1. Create `.opencode/skills/<name>/SKILL.md` with YAML frontmatter and documentation
2. Implement the functionality in a Python module
3. Test thoroughly - validate all generated notebooks compile
4. Document the API and requirements clearly

The `script-to-ipynb` skill demonstrates all these patterns and is a good reference implementation.
