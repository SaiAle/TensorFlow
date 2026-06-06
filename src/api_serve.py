from __future__ import annotations

import argparse
import os
from pathlib import Path

from .api import create_app


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--reload", action="store_true")
    args = p.parse_args(argv)

    data_dir = args.data_dir or Path(os.getcwd()) / "var"
    data_dir.mkdir(parents=True, exist_ok=True)

    import uvicorn

    uvicorn.run(
        create_app(data_dir),
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0
