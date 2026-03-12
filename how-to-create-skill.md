# 如何创建 OpenCode 技能

本教程介绍如何为 OpenCode 创建新技能，以 `script-to-ipynb` 技能作为实际示例。

## 什么是技能？

技能是在 OpenCode 会话期间可调用的可重用功能。技能允许您：

- 使用领域特定工具扩展 OpenCode 功能
- 将复杂操作封装为简单、可调用的单元
- 在会话之间共享专业知识

## 技能结构

技能由以下部分组成：

1. **SKILL.md** - 定义文件，位于 `.opencode/skills/<技能名称>/`
2. **实现** - 提供实际功能的 Python 模块

## 逐步指南

### 步骤 1：创建技能目录

创建目录结构：

```
.opencode/skills/<技能名称>/
```

本示例中：
```
.opencode/skills/script-to-ipynb/
```

### 步骤 2：创建 SKILL.md

SKILL.md 文件定义技能的元数据和文档。

```yaml
---
name: script-to-ipynb
description: 将 Python 脚本 (.py) 转换为 Jupyter notebook (.ipynb)
license: MIT
compatibility: opencode
metadata:
  audience: developers
  use_case: notebook 转换
---
```

#### Frontmatter 字段

| 字段 | 必需 | 描述 |
|------|------|------|
| `name` | 是 | 唯一技能标识符（kebab-case 格式）|
| `description` | 是 | 技能功能的简要描述 |
| `license` | 否 | 技能许可证 |
| `compatibility` | 是 | 设置为 `opencode` |
| `metadata` | 否 | 附加元数据（audience, use_case）|

#### 文档部分

在 frontmatter 之后，添加：

```markdown
## 我做什么

用简单的语言描述技能的功能。

## 何时使用

列出适合使用此技能的场景。

## 如何使用

解释 Python API 和任何要求。

### Python API

```python
# 展示代码示例
```

### 依赖要求

```bash
# 列出 pip 依赖
```

### 功能特性

- 功能 1
- 功能 2

### 示例输出

描述或展示示例输出。
```

### 步骤 3：实现技能

创建一个实现技能功能的 Python 模块。通常放在项目根目录或方便的位置。

#### 示例：script_to_ipynb.py

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
    """将 Python 脚本转换为 Jupyter notebook 的转换器类。"""

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
        """解析 Python 脚本并提取单元格。"""
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.parse_content(content)

    def parse_content(self, content: str) -> List[Tuple[str, str]]:
        """将 Python 脚本内容解析为单元格。"""
        cells: List[Tuple[str, str]] = []
        lines = content.split("\n")
        
        # 实现细节...
        
        return cells

    def create_notebook(self, cells: List[Tuple[str, str]]) -> dict:
        """从单元格创建 notebook 结构。"""
        if nbformat is None:
            raise ImportError("需要安装 nbformat。")
        
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
        """将 Python 脚本转换为 Jupyter notebook。"""
        script_path = Path(script_path)
        if not script_path.exists():
            raise FileNotFoundError(f"找不到脚本: {script_path}")

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
    """将脚本转换为 notebook 的便捷函数。"""
    converter = ScriptToNotebookConverter()
    return converter.convert(script_path, output_path, title)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python -m script_to_ipynb <脚本.py> [输出.ipynb]")
        sys.exit(1)
    script_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    result = convert_script(script_path, output_path)
    print(f"已创建 notebook: {result}")
```

### 步骤 4：关键实现模式

#### 单元格检测逻辑

notebook 转换器的核心是决定什么内容进入每个单元格：

```python
def _is_section_comment(self, line: str) -> bool:
    """检查行是否是章节注释。"""
    stripped = line.strip()
    return (
        stripped.startswith("# ")
        and (
            stripped.endswith("---")
            or stripped.endswith("===")
        )
    ) or (stripped.startswith("###") and len(stripped) > 3)

def _is_standalone_comment(self, line: str, lines: List[str], index: int) -> bool:
    """检查注释是否应该转换为 markdown 单元格。
    
    只有顶层注释（缩进为 0）才会成为 markdown 单元格。
    """
    stripped = line.strip()
    
    # 必须是第 0 列的注释
    if not stripped.startswith("#"):
        return False
    
    # 必须在缩进级别 0（不在函数/类内部）
    leading_spaces = len(line) - len(line.lstrip())
    if leading_spaces > 0:
        return False
    
    # 跳过分隔符和章节注释
    if self._is_cell_delimiter(line):
        return False
    if self._is_section_comment(line):
        return False
    
    # 检查后面是否有代码
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

#### 创建 Notebook 单元格

```python
def create_notebook(self, cells: List[Tuple[str, str]]) -> dict:
    """从单元格创建 notebook 结构。"""
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

### 步骤 5：测试您的技能

#### 手动测试

```python
import sys
sys.path.insert(0, 'D:/path/to/your/project')
from script_to_ipynb import convert_script

# 测试转换
result = convert_script("input.py", "output.ipynb", title="测试")
print(f"已创建: {result}")
```

#### 验证

始终验证生成的 notebook：

```python
import json

with open('output.ipynb') as f:
    nb = json.load(f)

# 检查单元格数量
print(f"单元格: {len(nb['cells'])}")

# 验证语法
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        try:
            compile(source, f'cell_{i}', 'exec')
            print(f"单元格 {i}: 正常")
        except SyntaxError as e:
            print(f"单元格 {i}: 语法错误 - {e}")
```

### 步骤 6：在 OpenCode 中使用您的技能

创建后，可以在 OpenCode 会话中调用技能：

```python
# 加载技能
skill(name="script-to-ipynb")

# 通过技能工具使用
# SKILL.md 中的技能描述指导如何使用
```

## 最佳实践

1. **保持 SKILL.md 简洁** - 专注于技能的功能和使用场景
2. **优雅处理依赖** - 对可选导入使用 try/except
3. **验证输出** - 始终验证生成的 notebook 可以正确编译
4. **提供清晰的错误消息** - 帮助用户了解出了什么问题
5. **同时支持 CLI 和 API** - 允许命令行和编程方式使用

## 常见问题

### 问题：类内部的注释变成 Markdown

**问题**：函数/类内部的注释被错误地转换为 markdown 单元格。

**解决方案**：检查缩进级别 - 只有缩进为 0 的注释才应该成为 markdown：

```python
leading_spaces = len(line) - len(line.lstrip())
if leading_spaces > 0:
    return False
```

### 问题：代码单元格被破坏

**问题**：代码被错误地分割到各个单元格，破坏了 Python 语法。

**解决方案**：确保完整的代码块放在一起。不要在每个注释处都分割。

## 总结

创建 OpenCode 技能：

1. 创建带有 YAML frontmatter 和文档的 `.opencode/skills/<名称>/SKILL.md`
2. 在 Python 模块中实现功能
3. 彻底测试 - 验证所有生成的 notebook 都能编译
4. 清晰记录 API 和依赖要求

`script-to-ipynb` 技能展示了所有这些模式，是一个很好的参考实现。
