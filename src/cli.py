from __future__ import annotations

import argparse
from pathlib import Path

from .core import apply_style
from .utils import ensure
from .models import resolve


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stylize")
    p.add_argument("content", type=Path, help="Content image path")
    p.add_argument("style", type=Path, help="Style image path")
    p.add_argument("--out", type=Path, default=Path("out.png"), help="Output path")
    p.add_argument("--model", default=None, help="TF Hub model URL or alias")
    p.add_argument("--cache-dir", type=Path, default=None, help="Cache directory")
    p.add_argument("--list-models", action="store_true", help="List fallback models")
    return p


def main(argv: list[str] | None = None) -> int:
    from .models import FALLBACK_MODELS, DEFAULT_MODEL

    args = build_parser().parse_args(argv)
    if args.list_models:
        print("Default:", DEFAULT_MODEL)
        for m in FALLBACK_MODELS:
            print("-", m["alias"], m["url"])
        return 0

    if not args.content.exists() or not args.style.exists():
        raise SystemExit("content and style must exist")

    cache_dir = ensure(args.cache_dir) if args.cache_dir else None
    model_url = resolve(args.model)

    summary = apply_style(
        args.content,
        args.style,
        args.out,
        model_url=model_url,
        cache_dir=cache_dir,
    )
    print(summary)
    return 0
