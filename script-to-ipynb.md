# Converting Python Scripts to Jupyter Notebooks

## Overview

This guide explains how to convert Python scripts (`.py` files) into Jupyter notebooks (`.ipynb` files). Jupyter notebooks provide an interactive environment that's ideal for data analysis, experimentation, and documentation.

## Table of Contents

1. [Manual Conversion Method](#manual-conversion-method)
2. [Automated Conversion Tools](#automated-conversion-tools)
3. [Best Practices](#best-practices)
4. [Common Issues and Solutions](#common-issues-and-solutions)
5. [Step-by-Step Example](#step-by-step-example)

---

## Manual Conversion Method

### Understanding Jupyter Notebook Structure

Jupyter notebooks are JSON files with the following structure:

```json
{
  "cells": [
    {
      "cell_type": "code" | "markdown",
      "metadata": {},
      "source": ["line 1", "line 2", ...],
      "execution_count": null,
      "outputs": []
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {...}
  },
  "nbformat": 4,
  "nbformat_minor": 4
}
```

### Step-by-Step Manual Conversion

1. **Create the notebook structure template**

2. **Parse the Python script to identify logical sections**
   - Import statements → First code cell
   - Class/function definitions → Separate code cells
   - Docstrings and comments → Markdown cells
   - Print statements or output → Code cells
   - Visualization code → Code cells

3. **Map script content to notebook cells**

**Example mapping:**

```python
# Script content:
"""
Module: Data Processing
Author: John Doe
Date: 2026-01-01
"""
import pandas as pd
import numpy as np

def process_data(df):
    """Process the input dataframe."""
    return df.dropna()

# Main execution
df = pd.read_csv('data.csv')
result = process_data(df)
print(result.head())
```

**Converted to notebook cells:**

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# Data Processing Module\n",
        "\n",
        "**Author:** John Doe  \n",
        "**Date:** 2026-01-01"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def process_data(df):\n",
        "    \"\"\"Process the input dataframe.\"\"\"\n",
        "    return df.dropna()"
      ]
    },
    {
      "cell_type": "markdown",
      "source": ["## Main Execution"]
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv('data.csv')\n",
        "result = process_data(df)\n",
        "print(result.head())"
      ]
    }
  ]
}
```

4. **Write the notebook as JSON**

```python
import json

notebook = {
    'cells': [...],  # Your converted cells
    'metadata': {...},
    'nbformat': 4,
    'nbformat_minor': 4
}

with open('notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
```

---

## Automated Conversion Tools

### 1. `p2j` (Python to Jupyter)

A command-line tool for converting Python scripts to notebooks.

```bash
pip install p2j
p2j script.py -o notebook.ipynb
```

### 2. `jupytext`

A powerful tool that supports bidirectional conversion between scripts and notebooks.

```bash
pip install jupytext
jupytext --to notebook script.py
```

**Features:**
- Preserves markdown comments as notebook cells
- Supports paired formats (script + notebook sync)
- Handles Jupyter notebooks as Python scripts with special comments

**Using paired formats:**

```bash
# Create paired notebook
jupytext --set-formats ipynb,py:percent script.py

# Edit either file, they stay in sync
# script.py will have cell markers like:
# %% [markdown]
# # Title
# %%
# import numpy as np
```

### 3. `ipython` nbconvert

Convert notebooks to scripts and vice versa.

```bash
# Convert script to notebook (requires manual cell marking)
jupyter nbconvert --to notebook script.py

# For better results, use cell markers:
# %% [markdown]
# # Markdown cell
# %%
# print("Code cell")
```

### 4. Custom Python Script

Create a conversion script for complete control:

```python
#!/usr/bin/env python
"""Convert Python script to Jupyter notebook"""

import re
import json

def parse_script_to_cells(script_path):
    """Parse Python script and convert to notebook cells."""
    
    with open(script_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cells = []
    current_cell = []
    cell_type = 'code'
    
    for line in lines:
        # Detect markdown sections (comments starting with #)
        if line.strip().startswith('#') and not current_cell:
            cell_type = 'markdown'
            content = line.strip().replace('# ', '')
            cells.append({
                'cell_type': cell_type,
                'metadata': {},
                'source': [content]
            })
        elif line.strip().startswith('"""') or line.strip().startswith("'''"):
            # Docstring → markdown
            pass
        else:
            # Code
            current_cell.append(line)
    
    # Add remaining code as last cell
    if current_cell:
        cells.append({
            'cell_type': 'code',
            'metadata': {},
            'source': current_cell,
            'execution_count': None,
            'outputs': []
        })
    
    return cells

def create_notebook(cells, title='Converted Notebook'):
    """Create notebook structure."""
    
    notebook = {
        'cells': cells,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'codemirror_mode': {'name': 'ipython', 'version': 3},
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.8.0'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }
    
    return notebook

def convert_script_to_notebook(script_path, output_path):
    """Convert script to notebook."""
    
    cells = parse_script_to_cells(script_path)
    notebook = create_notebook(cells)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f'Notebook created: {output_path}')

# Usage
if __name__ == '__main__':
    convert_script_to_notebook('script.py', 'notebook.ipynb')
```

---

## Best Practices

### 1. Cell Organization

**Logical grouping:**
- Imports together in first cell
- Related functions/classes in same cell
- Main execution logic in separate cell
- Visualizations in dedicated cell

**Example:**

```python
# Cell 1: Imports
import numpy as np
import matplotlib.pyplot as plt

# Cell 2: Configuration
CONFIG = {
    'batch_size': 32,
    'epochs': 100
}

# Cell 3: Helper functions
def preprocess(data):
    return data / 255.0

# Cell 4: Main model
class Model:
    def __init__(self):
        pass

# Cell 5: Training
model = Model()
# ... training code ...

# Cell 6: Visualization
plt.plot(history)
plt.show()
```

### 2. Adding Documentation

**Convert comments to markdown:**

```python
# Script:
# Load the dataset
# -----------------
# This section loads the CSV file and performs basic cleaning
data = pd.read_csv('data.csv')

# Notebook:
# Markdown cell:
"""
## Loading the Dataset

This section loads the CSV file and performs basic cleaning.
"""

# Code cell:
data = pd.read_csv('data.csv')
```

### 3. Handling Print Statements

**Keep print statements in code cells** - they provide output documentation:

```python
# Code cell:
print("Starting data processing...")
print(f"Loaded {len(data)} rows")
print("Processing complete!")
```

### 4. Managing Dependencies

**Document required packages in a markdown cell:**

```markdown
## Requirements

```bash
pip install numpy pandas matplotlib
```

## Environment

- Python 3.8+
- NumPy 1.19+
- Pandas 1.2+
```

### 5. Adding Execution Metadata

**Track execution order:**

```json
{
  "cell_type": "code",
  "execution_count": 1,  // Increment as cells are executed
  "metadata": {},
  "outputs": [...]
}
```

---

## Common Issues and Solutions

### Issue 1: Character Encoding Problems

**Problem:** Greek letters or special characters get corrupted.

**Solution:**

```python
# Always use UTF-8 encoding
with open('notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

# Avoid problematic characters in code
# Use ASCII alternatives:
# α → alpha
# μ → mu
# σ → std (for standard deviation)
```

### Issue 2: Long Functions in Single Cell

**Problem:** Functions are too long to read easily.

**Solution:** Split into logical sections:

```python
# Cell 1: Function definition
def complex_function(x, y, z):
    """Main function docstring."""
    # Implementation
    pass

# Cell 2: Helper functions
def helper1(x):
    return x * 2

def helper2(y):
    return y + 10

# Cell 3: Usage example
result = complex_function(1, 2, 3)
print(result)
```

### Issue 3: Missing Imports

**Problem:** Notebook cells executed out of order cause import errors.

**Solution:**

```python
# Cell 1 (always run first): All imports
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add restart button comment
# "⚠️ Run this cell first!"
```

### Issue 4: Large Data Objects in Memory

**Problem:** Variables persist across cell executions.

**Solution:**

```python
# Add cleanup cell at the end
# Markdown:
"""
## Cleanup

Clear variables to free memory.
"""

# Code:
import gc
del large_dataset
gc.collect()
```

### Issue 5: Path Issues

**Problem:** Relative paths break when notebook is moved.

**Solution:**

```python
# Use pathlib for robust paths
from pathlib import Path

# Get notebook directory
NOTEBOOK_DIR = Path.cwd()
DATA_PATH = NOTEBOOK_DIR / 'data' / 'dataset.csv'

# Use in code
data = pd.read_csv(DATA_PATH)
```

---

## Step-by-Step Example

### Converting a Complete Analysis Script

**Original script (`analysis.py`):**

```python
"""
Customer Churn Analysis
========================

This script analyzes customer churn data and builds a prediction model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_data(filepath):
    """Load and clean the dataset."""
    df = pd.read_csv(filepath)
    df = df.dropna()
    return df

def preprocess_features(df):
    """Preprocess feature columns."""
    # One-hot encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols)
    return df_encoded

def train_model(X_train, y_train):
    """Train Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model

def main():
    """Main execution function."""
    print("Loading data...")
    df = load_data('customer_data.csv')
    print(f"Loaded {len(df)} records")
    
    print("\nPreprocessing features...")
    df_processed = preprocess_features(df)
    
    print("\nSplitting data...")
    X = df_processed.drop('churn', axis=1)
    y = df_processed['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    print("\nTraining model...")
    model = train_model(X_train, y_train)
    
    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print("\nFeature importance:")
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    importance.head(10).plot.bar(x='feature', y='importance')
    plt.title('Top 10 Feature Importances')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("\nPlot saved to feature_importance.png")

if __name__ == '__main__':
    main()
```

**Converted notebook (`analysis.ipynb`):**

```json
{
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# Customer Churn Analysis\n",
        "\n",
        "This notebook analyzes customer churn data and builds a prediction model."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import classification_report\n",
        "import matplotlib.pyplot as plt"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Configuration\n",
        "\n",
        "Set random seed and test size for reproducibility."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "RANDOM_STATE = 42\n",
        "TEST_SIZE = 0.2"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Data Loading\n",
        "\n",
        "Load and clean the dataset."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "def load_data(filepath):\n",
        "    \"\"\"Load and clean the dataset.\"\"\"\n",
        "    df = pd.read_csv(filepath)\n",
        "    df = df.dropna()\n",
        "    return df"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Feature Preprocessing\n",
        "\n",
        "Preprocess feature columns with one-hot encoding."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "def preprocess_features(df):\n",
        "    \"\"\"Preprocess feature columns.\"\"\"\n",
        "    # One-hot encode categorical variables\n",
        "    categorical_cols = df.select_dtypes(include=['object']).columns\n",
        "    df_encoded = pd.get_dummies(df, columns=categorical_cols)\n",
        "    return df_encoded"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Model Training\n",
        "\n",
        "Train Random Forest classifier."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "def train_model(X_train, y_train):\n",
        "    \"\"\"Train Random Forest classifier.\"\"\"\n",
        "    model = RandomForestClassifier(\n",
        "        n_estimators=100,\n",
        "        random_state=RANDOM_STATE\n",
        "    )\n",
        "    model.fit(X_train, y_train)\n",
        "    return model"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Main Analysis\n",
        "\n",
        "Execute the complete analysis pipeline."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"Loading data...\")\n",
        "df = load_data('customer_data.csv')\n",
        "print(f\"Loaded {len(df)} records\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"\\nPreprocessing features...\")\n",
        "df_processed = preprocess_features(df)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"\\nSplitting data...\")\n",
        "X = df_processed.drop('churn', axis=1)\n",
        "y = df_processed['churn']\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"\\nTraining model...\")\n",
        "model = train_model(X_train, y_train)"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Model Evaluation\n",
        "\n",
        "Evaluate model performance on test set."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"\\nEvaluating model...\")\n",
        "y_pred = model.predict(X_test)\n",
        "print(classification_report(y_test, y_pred))"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Feature Importance\n",
        "\n",
        "Analyze which features contribute most to predictions."
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "print(\"\\nFeature importance:\")\n",
        "importance = pd.DataFrame({\n",
        "    'feature': X.columns,\n",
        "    'importance': model.feature_importances_\n",
        "}).sort_values('importance', ascending=False)\n",
        "print(importance.head(10))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Plot feature importance\n",
        "plt.figure(figsize=(10, 6))\n",
        "importance.head(10).plot.bar(x='feature', y='importance')\n",
        "plt.title('Top 10 Feature Importances')\n",
        "plt.xticks(rotation=45)\n",
        "plt.tight_layout()\n",
        "plt.savefig('feature_importance.png')\n",
        "print(\"\\nPlot saved to feature_importance.png\")\n",
        "plt.show()"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {"name": "ipython", "version": 3},
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.8.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}
```

---

## Advanced Techniques

### 1. Adding Widgets

```python
from ipywidgets import interact, IntSlider

@interact(n=IntSlider(min=1, max=100, value=10))
def show_first_n_rows(n):
    return df.head(n)
```

### 2. Creating Interactive Plots

```python
%matplotlib widget
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
line, = ax.plot(x, y)
plt.show()
```

### 3. Using Magic Commands

```python
# Time execution
%%time
result = slow_function()

# Profile memory
%%prun
result = memory_intensive_function()

# Run bash commands
%%bash
ls -lh data/
```

### 4. Adding HTML/CSS Styling

```python
from IPython.display import HTML

HTML("""
<style>
.warning {
    background-color: #fff3cd;
    border: 1px solid #ffc107;
    padding: 10px;
    border-radius: 5px;
}
</style>
<div class="warning">
    <strong>Warning:</strong> This operation may take several minutes.
</div>
""")
```

---

## Summary

Converting Python scripts to Jupyter notebooks involves:

1. **Understanding the JSON structure** of notebooks
2. **Mapping script sections** to appropriate cell types
3. **Adding documentation** through markdown cells
4. **Organizing code** into logical, executable units
5. **Handling dependencies** and execution order

**Key takeaways:**
- Use automated tools like `jupytext` for efficient conversion
- Organize cells logically for better readability
- Add markdown documentation between code sections
- Handle encoding issues carefully with UTF-8
- Test the notebook after conversion

**Recommended workflow:**
1. Use `jupytext` for initial conversion
2. Manually refine cell boundaries
3. Add markdown documentation
4. Test execution order
5. Add interactive elements if needed

---

## Resources

- [Jupyter Documentation](https://jupyter.org/documentation)
- [Jupytext Documentation](https://jupytext.readthedocs.io/)
- [nbconvert Documentation](https://nbconvert.readthedocs.io/)
- [Jupyter Notebook Format](https://nbformat.readthedocs.io/en/latest/)

---

**Author:** Generated Report  
**Date:** 2026-03-11  
**Version:** 1.0