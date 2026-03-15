"""
One-command setup: generates data, trains model, launches dashboard.
Usage: python run.py
"""

import subprocess
import sys


def main():
    print("=" * 60)
    print("  HydroPredict AI — Setup & Launch")
    print("=" * 60)

    print("\n[1/3] Generating Nepal weather & hydrology data...")
    from src.data_generator import save_all_data
    save_all_data()

    print("\n[2/3] Training XGBoost forecasting model...")
    from src.model import train_model
    train_model()

    print("\n[3/3] Launching Streamlit dashboard...")
    print("=" * 60)
    print("  Open http://localhost:8501 in your browser")
    print("=" * 60)
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true"])


if __name__ == "__main__":
    main()
