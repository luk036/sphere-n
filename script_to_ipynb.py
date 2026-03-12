#!/usr/bin/env python
r"""Script to Jupyter Notebook Converter Skill.

This module provides functionality to convert Python scripts (.py files)
into Jupyter notebooks (.ipynb files).

## Overview

This skill enables AI agents to:
1. Parse Python scripts and identify logical sections
2. Convert script structure to notebook cells (code + markdown)
3. Generate properly formatted .ipynb JSON files

## Usage

```python
from script_to_ipynb import ScriptToNotebookConverter

converter = ScriptToNotebookConverter()
converter.convert("script.py", "notebook.ipynb")
```

## Cell Detection Patterns

- Import statements → First code cell
- Class/function definitions → Separate code cells
- Docstrings and comments → Markdown cells
- Print statements or output → Code cells
- Visualization code → Code cells

## Supported Features

- Manual conversion with custom parsing rules
- Jupytext integration for advanced conversion
- UTF-8 encoding support
- Custom cell boundaries via comments

## Requirements

- Python 3.8+
- nbformat (for notebook structure)
- jupytext (optional, for advanced features)

## Installation

```bash
pip install nbformat jupytext
```
"""

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
    r"""Converter class for transforming Python scripts to Jupyter notebooks.

    This class handles the conversion of Python scripts into Jupyter notebook
    format, preserving code structure and converting comments/docstrings
    to markdown cells for documentation.

    Attributes:
        include_outputs: Whether to include output placeholders
        kernel_name: Name of the kernel to use (default: python3)
        python_version: Python version string for metadata
    """

    def __init__(
        self,
        include_outputs: bool = False,
        kernel_name: str = "python3",
        python_version: str = "3.8.0",
    ) -> None:
        r"""Initialize the converter.

        Args:
            include_outputs: Whether to include empty output arrays
            kernel_name: Name of the Jupyter kernel
            python_version: Python version for metadata
        """
        self.include_outputs = include_outputs
        self.kernel_name = kernel_name
        self.python_version = python_version

    def parse_script(self, script_path: str) -> List[Tuple[str, str]]:
        r"""Parse a Python script and extract cells.

        This method analyzes the script and identifies logical sections
        to convert into notebook cells. It detects:
        - Module docstrings → markdown
        - Section comments (### or ---) → markdown
        - Regular comments at start of blocks → markdown
        - Code blocks → code cells

        Args:
            script_path: Path to the Python script

        Returns:
            List of (cell_type, content) tuples
        """
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        return self.parse_content(content)

    def parse_content(self, content: str) -> List[Tuple[str, str]]:
        r"""Parse Python script content into cells.

        Args:
            content: Python script content as string

        Returns:
            List of (cell_type, content) tuples
        """
        cells: List[Tuple[str, str]] = []
        lines = content.split("\n")

        # Skip module docstring if present
        i = 0
        if self._is_module_docstring(lines):
            # Find end of docstring
            docstring_end = self._find_docstring_end(lines, 0)
            if docstring_end > 0:
                docstring_content = "\n".join(lines[1:docstring_end])
                cells.append(("markdown", self._clean_docstring(docstring_content)))
                i = docstring_end + 1

        current_code: List[str] = []

        while i < len(lines):
            line = lines[i]

            # Check for cell delimiter comments
            if self._is_cell_delimiter(line):
                # Save current code cell
                if current_code:
                    cells.append(("code", "\n".join(current_code)))
                    current_code = []

                # Extract markdown content
                markdown_content = self._extract_delimiterMarkdown(line, lines, i)
                if markdown_content:
                    cells.append(("markdown", markdown_content))
                i += 1
                continue

            # Check for section comments (### or --- or # ===)
            if self._is_section_comment(line):
                if current_code:
                    cells.append(("code", "\n".join(current_code)))
                    current_code = []

                # Get section title
                section_title = self._extract_section_title(line, lines, i)
                cells.append(("markdown", section_title))
                i += 1
                continue

            # Check for standalone comment at indentation level 0
            if self._is_standalone_comment(line, lines, i):
                if current_code:
                    cells.append(("code", "\n".join(current_code)))
                    current_code = []

                comment_text = line.strip().lstrip("#")
                cells.append(("markdown", comment_text))
                i += 1
                continue

            # Regular code line
            current_code.append(line)
            i += 1

        # Add remaining code cell
        if current_code:
            cells.append(("code", "\n".join(current_code)))

        return cells

    def _is_module_docstring(self, lines: List[str]) -> bool:
        r"""Check if the script starts with a module docstring.

        Args:
            lines: List of lines from the script

        Returns:
            True if first non-empty line is a docstring
        """
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            return stripped.startswith('"""') or stripped.startswith("'''")
        return False

    def _find_docstring_end(self, lines: List[str], start: int) -> int:
        r"""Find the end line of a docstring.

        Args:
            lines: List of lines
            start: Starting line index

        Returns:
            Index of the line containing closing triple quotes
        """
        quote = '"""' if '"""' in lines[start] else "'''"

        for i in range(start + 1, len(lines)):
            if quote in lines[i]:
                return i
        return len(lines) - 1

    def _clean_docstring(self, docstring: str) -> str:
        r"""Clean a docstring for use as markdown.

        Args:
            docstring: Raw docstring content

        Returns:
            Cleaned markdown-formatted string
        """
        # Remove leading/trailing quotes and whitespace
        cleaned = docstring.strip()
        # Convert title case headers
        cleaned = re.sub(r"^=+\s*$", "", cleaned, flags=re.MULTILINE)
        return cleaned.strip()

    def _is_cell_delimiter(self, line: str) -> bool:
        r"""Check if line is a cell delimiter comment.

        Args:
            line: Line to check

        Returns:
            True if line is a cell delimiter
        """
        stripped = line.strip()
        # Support various cell delimiter formats
        return (
            stripped.startswith("# %%")
            or stripped.startswith("# %% [markdown]")
            or stripped.startswith("# <codecell>")
            or stripped.startswith("# +")
        )

    def _extract_delimiterMarkdown(
        self, line: str, lines: List[str], index: int
    ) -> Optional[str]:
        r"""Extract markdown content from cell delimiter.

        Args:
            line: Current line
            lines: All lines
            index: Current index

        Returns:
            Markdown content or None
        """
        stripped = line.strip()

        # Handle # %% [markdown] format
        if "[markdown]" in stripped.lower():
            # Get content after [markdown]
            match = re.search(r"\[markdown\]\s*(.*)", stripped, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # For other delimiters, look ahead for markdown content
        if index + 1 < len(lines):
            next_line = lines[index + 1].strip()
            if next_line.startswith("#"):
                return next_line.lstrip("# ").strip()

        return None

    def _is_section_comment(self, line: str) -> bool:
        r"""Check if line is a section comment (header).

        Args:
            line: Line to check

        Returns:
            True if line is a section comment
        """
        stripped = line.strip()
        return (
            stripped.startswith("# ")
            and (
                stripped.endswith("---")
                or stripped.endswith("===")
                or re.match(r"#\s*={3,}", stripped)
                or re.match(r"#\s*-{3,}", stripped)
            )
        ) or (stripped.startswith("###") and len(stripped) > 3)

    def _extract_section_title(self, line: str, lines: List[str], index: int) -> str:
        r"""Extract section title from comment.

        Args:
            line: Current line
            lines: All lines
            index: Current index

        Returns:
            Section title as markdown
        """
        stripped = line.strip()

        # Remove # and leading/trailing symbols
        title = stripped.lstrip("#").strip()
        title = re.sub(r"[-=]{3,}$", "", title).strip()

        # Convert to proper markdown header
        return f"## {title}" if title else "## Section"

    def _is_standalone_comment(self, line: str, lines: List[str], index: int) -> bool:
        r"""Check if comment should be converted to markdown cell.

        Only comments at indentation level 0 (top-level) should become markdown cells.
        Comments inside functions/classes (indented) should remain as code.

        Args:
            line: Current line
            lines: All lines
            index: Current index

        Returns:
            True if comment should become a markdown cell
        """
        stripped = line.strip()

        # Must be a comment at column 0
        if not stripped.startswith("#"):
            return False

        # Must be at indentation level 0 (not inside a function/class)
        # Calculate leading whitespace
        leading_spaces = len(line) - len(line.lstrip())
        if leading_spaces > 0:
            return False

        # Not a cell delimiter
        if self._is_cell_delimiter(line):
            return False

        # Not a section comment
        if self._is_section_comment(line):
            return False

        # Check if followed by code (not just comment lines)
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

    def create_notebook(self, cells: List[Tuple[str, str]]) -> dict:
        r"""Create a notebook structure from cells.

        Args:
            cells: List of (cell_type, content) tuples

        Returns:
            Notebook JSON structure
        """
        if nbformat is None:
            raise ImportError("nbformat is required. Install: pip install nbformat")

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
        r"""Convert a Python script to a Jupyter notebook.

        Args:
            script_path: Path to input Python script
            output_path: Path for output notebook (default: script.ipynb)
            title: Optional title for the notebook

        Returns:
            Path to the created notebook

        Raises:
            FileNotFoundError: If script_path doesn't exist
            ImportError: If nbformat is not installed
        """
        script_path = Path(script_path)
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")

        # Determine output path
        if output_path is None:
            output_path = str(script_path.with_suffix(".ipynb"))
        else:
            output_path = str(output_path)

        # Parse script into cells
        cells = self.parse_script(str(script_path))

        # Add title as first cell if provided
        if title:
            cells.insert(0, ("markdown", f"# {title}"))

        # Create notebook
        nb = self.create_notebook(cells)

        # Write notebook
        with open(output_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)

        return output_path


def convert_script(
    script_path: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    r"""Convenience function to convert a script to notebook.

    Args:
        script_path: Path to input Python script
        output_path: Path for output notebook
        title: Optional title

    Returns:
        Path to created notebook
    """
    converter = ScriptToNotebookConverter()
    return converter.convert(script_path, output_path, title)


def main() -> None:
    r"""Command-line interface for the converter."""
    if len(sys.argv) < 2:
        print("Usage: python -m script_to_ipynb <script.py> [output.ipynb]")
        sys.exit(1)

    script_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        result = convert_script(script_path, output_path)
        print(f"Created notebook: {result}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
