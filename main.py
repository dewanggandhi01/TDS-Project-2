"""Thin wrapper to expose FastAPI app when importing from project root.

Prefer importing and running `app` from `app.main`.
"""

from app.main import app  # noqa: F401

if __name__ == "__main__":
    try:
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        raise SystemExit("uvicorn is required. Install dependencies with: pip install -r requirements.txt")