from __future__ import annotations

import argparse
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, web_dir: Path, api_base: str, **kwargs) -> None:
        super().__init__(*args, directory=str(web_dir), **kwargs)
        self.api_base = api_base

    def do_GET(self) -> None:  # noqa: N802
        return super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        return super().end_headers()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--web-dir", type=Path, default=Path(__file__).resolve().parent.parent / "web")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5173)
    p.add_argument("--api", default="http://127.0.0.1:8000")
    args = p.parse_args(argv)

    web_dir = args.web_dir.resolve()
    if not web_dir.exists():
        raise SystemExit(f"missing web dir: {web_dir}")

    def factory(*a, **kw):
        return Handler(*a, web_dir=web_dir, api_base=args.api, **kw)

    with ThreadingHTTPServer((args.host, args.port), factory) as server:
        print(f"creator UI on http://{args.host}:{args.port}  (api -> {args.api})")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
