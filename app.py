import sys
sys.dont_write_bytecode = True   

import os
import subprocess
import time
import webbrowser
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent
API_URL = "http://127.0.0.1:8000"
DASHBOARD_URL = "http://localhost:8501"

NEEDED_DATA = ["train.csv", "test_unlabelled.csv", "sample_submission.csv", "partners.csv", "products.csv"]
NEEDED_MODEL = ["model.joblib", "scorer_config.json", "partner_snapshot.csv", "serial_counts.json"]


def files_missing(folder, names):
    return [n for n in names if not (BASE / folder / n).exists()]


def prepare():
    if files_missing("model", NEEDED_MODEL):
        missing_data = files_missing("data", NEEDED_DATA)
        if missing_data:
            print("The model is not built yet and these files are missing from data/:")
            print("  " + ", ".join(missing_data))
            print("Copy the Kestrel data pack into the data/ folder and run again.")
            return False
        print("Model not found. Building it first (about a minute)...")
        subprocess.run([sys.executable, "run_pipeline.py"], cwd=BASE, check=True)
    missing_ref = files_missing("data", ["partners.csv", "products.csv"])
    if missing_ref:
        print("Please copy these files into data/ and run again: " + ", ".join(missing_ref))
        return False
    return True


def wait_for_api(seconds=90):
    for _ in range(seconds):
        try:
            if requests.get(API_URL + "/health", timeout=2).status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def main():
    if not prepare():
        return

    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(BASE))

    print("Starting the scoring service...")
    api = subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app",
                            "--host", "127.0.0.1", "--port", "8000"], cwd=BASE, env=env)
    if not wait_for_api():
        print("The scoring service did not start. Look at the messages above.")
        api.terminate()
        return

    print("Starting the dashboard...")
    dashboard = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard.py",
                                  "--server.port", "8501", "--server.headless", "true",
                                  "--browser.gatherUsageStats", "false"], cwd=BASE, env=env)
    time.sleep(4)
    print("\nOpen the dashboard:", DASHBOARD_URL)
    print("API docs:          ", API_URL + "/docs")
    print("Press Ctrl+C to stop.\n")
    webbrowser.open(DASHBOARD_URL)

    try:
        dashboard.wait()
    except KeyboardInterrupt:
        pass
    finally:
        dashboard.terminate()
        api.terminate()


if __name__ == "__main__":
    main()