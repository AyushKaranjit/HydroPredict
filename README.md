## Prerequisites

•⁠  ⁠Python 3.10+
•⁠  ⁠pip (bundled with Python)

Verify installation:

⁠ bash
python --version
python -m pip --version
 ⁠

If ⁠ pip ⁠ fails, use ⁠ python -m pip ⁠ instead of ⁠ pip ⁠.

## Setup

1.⁠ ⁠Clone and enter the project:

⁠ bash
git clone <this-repo-url>
cd <file-name>
 ⁠

2.⁠ ⁠Create and activate a virtual environment:

Windows PowerShell:

⁠ powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
 ⁠

macOS/Linux:

⁠ bash
python -m venv .venv
source .venv/bin/activate
 ⁠

3.⁠ ⁠Install dependencies:

⁠ bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
 ⁠

## Run Options

### Option A: Launch dashboard with existing trained model (fastest)

Use this when model files already exist in ⁠ models/ ⁠.

⁠ bash
streamlit run app.py
 ⁠

Open ⁠ http://localhost:8501 ⁠.

### Option B: Retrain model and launch dashboard

Use this if you want fresh training.

Required dataset:

•⁠  ⁠⁠ data/nepali_multi_district.csv ⁠

Run:

⁠ bash
python run.py
 ⁠

This trains from the dataset and then starts Streamlit.

## Manual Pipeline

⁠ bash
# Train model only (requires data/nepali_multi_district.csv)
python -m src.model

# Launch UI only
streamlit run app.py
 ⁠

## Common Issues

•⁠  ⁠⁠ pip install python3 ⁠ fails:
   - ⁠ python3 ⁠ is not a package. Install dependencies with:
   - ⁠ python -m pip install -r requirements.txt ⁠
•⁠  ⁠⁠ FileNotFoundError: data/nepali_multi_district.csv ⁠:
   - Use Option A (⁠ streamlit run app.py ⁠) if you only want to run the UI with pre-trained artifacts.
   - Or add the dataset and use Option B.
•⁠  ⁠⁠ streamlit ⁠ not found:
   - Activate your virtual environment and reinstall requirements.
# NAMI-5
