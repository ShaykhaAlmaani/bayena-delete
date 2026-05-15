"""
run.py — Start both the FastAPI backend and the Dash frontend.

Usage:
    python run.py

  This starts:
    - FastAPI on  http://localhost:8000  (API + docs at /docs)
    - Dash UI on  http://localhost:8050  (dashboard interface)
"""

import os
import subprocess
import sys
import threading
import time


def run_backend():
    subprocess.run(
        [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", os.getenv("APP_HOST", "0.0.0.0"),
            "--port", os.getenv("APP_PORT", "8000"),
        ],
        check=False,
    )


def run_frontend():
    time.sleep(2)   # Give the backend a moment to start
    subprocess.run(
        [
            sys.executable, "-m", "app.ui.dashboard_app",
        ],
        check=False,
    )


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Bayena / بيّنة  —  AI Dashboard Platform")
    print("=" * 60)
    print("  Backend API  →  http://localhost:8000")
    print("  API Docs     →  http://localhost:8000/docs")
    print("  Dashboard UI →  http://localhost:8050")
    print("=" * 60 + "\n")

    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()

    run_frontend()
