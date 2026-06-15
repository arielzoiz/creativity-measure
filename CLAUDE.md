
# Project Guidelines

## Environment

This project uses **conda**. The environment is named `creativity-measure`.
Always activate it before running anything:

    conda activate creativity-measure

Run all Python commands, scripts, tests, and installs inside this environment.
Do not use the base environment or a different venv.

## Type Checking

This project uses strict static type checking (**Pylance** / **Pyright**).
When you finish implementing a part, run the type checker and fix any errors
before moving on:

    pyright

Run it inside the `creativity-measure` conda environment.
Write fully type-annotated code so it passes cleanly.

## Jupyter Notebooks

When reading or editing `.ipynb` files, use the **notebook MCP server**.
Do NOT parse, read, or write the notebook JSON directly.

- To read a notebook's contents, use the notebook MCP tools.
- To add, edit, or run cells, use the notebook MCP tools.
- Editing the raw JSON risks corrupting cell structure, outputs, and metadata.