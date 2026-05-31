# creativity-measure

## Setup

**1. Create and activate a conda environment:**
```bash
conda create -n creativity-measure python=3.12
conda activate creativity-measure
```

**2. Install the package and dependencies:**
```bash
cd creativity-measure
python -m pip install -e ".[dev]"
```

**3. Register the Jupyter kernel:**
```bash
python -m pip install ipykernel
python -m ipykernel install --user --name creativity-measure --display-name "creativity-measure"
```

## Usage

**Each session**, activate the environment first:
```bash
conda activate creativity-measure
```

**Running the notebook in VS Code:**

1. Open `ipynb` files in VS Code
2. Click the kernel selector (top-right corner)
3. Select **creativity-measure** from the list (if it doesn't appear, reload the window with `Cmd+Shift+P` → "Developer: Reload Window")
4. Set the Python interpreter to match: `Cmd+Shift+P` → "Python: Select Interpreter" → choose `/usr/local/Caskroom/miniconda/base/envs/creativity-measure/bin/python` (might require a reload as well)

**Running tests:**
```bash
pytest tests/
```

**When done**, deactivate the environment to restore your default shell:
```bash
conda deactivate
```
