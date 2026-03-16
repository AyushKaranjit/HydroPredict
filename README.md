## Quick Start

1. **Clone the repo & enter the folder**
   ```bash
   git clone <this-repo-url>
   cd hydropower-hackathon
   ```

2. **Install dependencies**
   - Make sure you have a recent **Python 3** (e.g. 3.10+).
   - Then run:
     ```bash
     pip install -r requirements.txt
     ```

3. **Ensure the main dataset exists**
   - Confirm that the file `data/nepali_multi_district.csv` is present.
   - If it’s missing, copy/download it into the `data/` folder **with exactly this filename**.

4. **Run everything with one command (recommended)**
   From the project root:
   ```bash
   python run.py
   ```
   This will:
   - Train the model on `data/nepali_multi_district.csv`.
   - Launch the Streamlit dashboard.

   Then open `http://localhost:8501` in your browser.

---

## Manual Setup (alternative)

If you prefer to run each step yourself:

```bash
# (Optional) Generate synthetic demo data
python -m src.data_generator

# Train the model on data/nepali_multi_district.csv
python -m src.model

# Launch the dashboard
streamlit run app.py
```

The only required dataset is `data/nepali_multi_district.csv`; the other CSVs are optional/demo outputs produced by the data generator.
