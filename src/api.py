from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from .core import apply_style
from .utils import ensure
from .models import resolve


class AvatarIn(BaseModel):
    name: str
    style_rel_path: str


class AvatarOut(BaseModel):
    id: str
    name: str
    style_rel_path: str


def _presets_path(data_dir: Path) -> Path:
    return data_dir / "presets.json"


def _load_presets(path: Path) -> dict[str, AvatarOut]:
    if not path.exists():
        return {}
    import json

    raw = json.loads(path.read_text())
    out: dict[str, AvatarOut] = {}
    for item in raw:
        out[item["id"]] = AvatarOut(**item)
    return out


def _save_presets(path: Path, presets: dict[str, AvatarOut]) -> None:
    path.write_text(json.dumps([v.model_dump() for v in presets.values()], indent=2))


def create_app(data_dir: Path) -> FastAPI:
    app = FastAPI(title="Style Transfer")
    presets_file = _presets_path(data_dir)

    @app.get("/")
    def read_root() -> dict[str, str]:
        return {"service": "style-transfer-api", "status": "ok"}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "data_dir": str(data_dir)}

    @app.post("/transfer", response_model=dict)
    async def transfer(
        content: bytes | None = None,
        style: bytes | None = None,
        model: str | None = None,
        cache: bool = False,
    ) -> dict:
        out_path = data_dir / "out.png"
        cache_dir = ensure(data_dir / ".cache") if cache else None

        if content is None or style is None:
            return {"out": "", "bytes": 0, "cached": False}

        content_path = data_dir / "content_upload.bin"
        style_path = data_dir / "style_upload.bin"
        content_path.write_bytes(content)
        style_path.write_bytes(style)

        summary = apply_style(
            content_path,
            style_path,
            out_path,
            model_url=resolve(model),
            cache_dir=cache_dir,
        )
        return summary

    @app.post("/presets", response_model=AvatarOut)
    def create_preset(payload: AvatarIn) -> AvatarOut:
        import uuid

        preset_id = uuid.uuid4().hex[:8]
        preset = AvatarOut(id=preset_id, name=payload.name, style_rel_path=payload.style_rel_path)
        presets = _load_presets(presets_file)
        presets[preset_id] = preset
        _save_presets(presets_file, presets)
        return preset

    @app.get("/presets", response_model=list[AvatarOut])
    def list_presets() -> list[AvatarOut]:
        return list(_load_presets(presets_file).values())

    return app
