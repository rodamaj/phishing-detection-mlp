# Phishing URL Detection with MLP

This project uses a Multilayer Perceptron (MLP) to classify URLs as phishing or legitimate based on engineered features.

## 🧰 Tech Stack
- Python 3.12.x
- PyTorch
- Scikit-learn
- Pandas / NumPy
- Matplotlib
- TensorFlow

---

## ⚙️ Environment Setup

It is recommended to use a virtual environment to manage dependencies.

### 1. Create a virtual environment

```bash
python3 -m venv venv
```

### 2. Activate the environment

- On macOS/Linux:
```bash
source venv/bin/activate
```

- On Windows:
```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install development dependencies

Use these dependencies when working with notebooks or contributing to the project.

```bash
pip install -r requirements-dev.txt
```

### 5. Configure notebook output stripping

The repository is configured to strip notebook outputs with `nbstripout` before committing.
After installing the development dependencies, enable the Git filter locally:

```bash
nbstripout --install
```

You can verify the installation with:

```bash
nbstripout --status
```

### 6. Register the notebook kernel

Register the virtual environment as a Jupyter kernel so notebooks use the project dependencies:

```bash
python -m ipykernel install --user --name phishing-detection-mlp --display-name "Python (phishing-detection-mlp)"
```

Then, in Jupyter or VS Code, select the kernel named:

```text
Python (phishing-detection-mlp)
```

### 7. How to run

```bash
python main.py
```

On the first run, the project trains the MLP and saves the training artifacts in
the `artifacts/` directory:

- `model.keras`
- `scaler.pkl`
- `metadata.json`

On later runs, the program loads those artifacts and skips retraining when they
are compatible with the current feature columns and training configuration.

To force a new training run:

```bash
python main.py --retrain
```

To use a different artifact directory:

```bash
python main.py --artifact-dir artifacts-experiment
```
