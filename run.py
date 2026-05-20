"""
Start both the Streamlit app and the FastAPI webhook server.
Usage:  python3 run.py
"""

import subprocess
import sys
import threading
import time


def run_webhook():
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "webhook:app", "--host", "0.0.0.0", "--port", "8000"],
        check=True,
    )


def run_streamlit():
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"],
        check=True,
    )


if __name__ == "__main__":
    print("Starting webhook server on :8000 ...")
    t = threading.Thread(target=run_webhook, daemon=True)
    t.start()

    time.sleep(1)  # give uvicorn a moment to bind

    print("Starting Streamlit on :8501 ...")
    run_streamlit()
