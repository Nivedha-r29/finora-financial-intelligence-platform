"""Launch the Finora dashboard without starting any training runs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    app_path = Path(__file__).resolve().parent / "dashboard" / "app.py"
    if not app_path.is_file():
        raise FileNotFoundError(f"Dashboard entry point was not found: {app_path}")
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        cwd=app_path.parents[1],
    )


if __name__ == "__main__":
    raise SystemExit(main())
