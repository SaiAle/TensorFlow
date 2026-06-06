from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .core import apply_style
from .utils import ensure
from .models import resolve


class TransferResponse(BaseModel):
    url: str
    bytes: int
    cached: bool


class PresetIn(BaseModel):
    name: str
    style_rel_path: str


class PresetOut(BaseModel):
    id: str
    name: str
    style_rel_path: str


def _presets_path(data_dir: Path) -> Path:
    return data_dir / "presets.json"


def _load_presets(path: Path) -> dict[str, PresetOut]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: dict[str, PresetOut] = {}
    for item in raw:
        out[item["id"]] = PresetOut(**item)
    return out


def _save_presets(path: Path, presets: dict[str, PresetOut]) -> None:
    path.write_text(json.dumps([v.model_dump() for v in presets.values()], indent=2))


def create_app(data_dir: Path) -> FastAPI:
    app = FastAPI(title="Style Transfer")
    presets_file = _presets_path(data_dir)
    out_dir = ensure(data_dir / "outputs")
    app.mount("/outputs", StaticFiles(directory=str(out_dir)), name="outputs")

    @app.get("/")
    def read_root() -> dict[str, str]:
        return {"service": "style-transfer-api", "status": "ok"}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "data_dir": str(data_dir)}

    @app.post("/transfer", response_model=TransferResponse)
    async def transfer(
        content: UploadFile = File(...),
        style: UploadFile = File(...),
        model: str | None = None,
        cache: bool = False,
    ) -> TransferResponse:
        try:
            content_path = data_dir / content.filename
            style_path = data_dir / style.filename
            content_path.write_bytes(await content.read())
            style_path.write_bytes(await style.read())
        except Exception as exc:
            raise HTTPException(400, f"invalid upload: {exc}") from exc

        out_path = out_dir / f"{content_path.stem}__{style_path.stem}.png"
        cache_dir = ensure(data_dir / ".cache") if cache else None

        summary = apply_style(
            content_path,
            style_path,
            out_path,
            model_url=resolve(model),
            cache_dir=cache_dir,
        )
        return TransferResponse(url=f"/outputs/{out_path.name}", bytes=summary["bytes"], cached=summary.get("cached", False))

    @app.post("/transfer/preset", response_model=TransferResponse)
    async def transfer_by_preset(
        content: UploadFile = File(...),
        preset_id: str | None = None,
        model: str | None = None,
        cache: bool = False,
    ) -> TransferResponse:
        if not preset_id:
            raise HTTPException(400, "preset_id is required")
        presets = _load_presets(presets_file)
        preset = presets.get(preset_id)
        if preset is None:
            raise HTTPException(404, f"unknown preset_id: {preset_id}")

        content_path = data_dir / content.filename
        content_path.write_bytes(await content.read())
        style_path = (data_dir / preset.style_rel_path).resolve()

        if not content_path.exists():
            raise HTTPException(404, f"missing content file: {content.filename}")
        if not style_path.exists():
            raise HTTPException(500, f"preset style missing: {preset.style_rel_path}")

        out_path = out_dir / f"{content_path.stem}__{preset.id}.png"
        cache_dir = ensure(data_dir / ".cache") if cache else None

        summary = apply_style(
            content_path,
            style_path,
            out_path,
            model_url=resolve(model),
            cache_dir=cache_dir,
        )
        return TransferResponse(url=f"/outputs/{out_path.name}", bytes=summary["bytes"], cached=summary.get("cached", False))

    @app.post("/presets", response_model=PresetOut)
    def create_preset(payload: PresetIn) -> PresetOut:
        style_path = (data_dir / payload.style_rel_path).resolve()
        if not style_path.exists():
            raise HTTPException(400, f"missing style file: {payload.style_rel_path}")
        preset_id = __import__("uuid").uuid4().hex[:8]
        preset = PresetOut(id=preset_id, name=payload.name, style_rel_path=payload.style_rel_path)
        presets = _load_presets(presets_file)
        presets[preset_id] = preset
        _save_presets(presets_file, presets)
        return preset

    @app.get("/presets", response_model=list[PresetOut])
    def list_presets() -> list[PresetOut]:
        return list(_load_presets(presets_file).values())

    @app.post("/styles")
    async def upload_style(style: UploadFile = File(...)) -> dict[str, str]:
        styles_dir = ensure(data_dir / "styles")
        dest = styles_dir / style.filename
        dest.write_bytes(await style.read())
        return {"style_rel_path": f"styles/{style.filename}"}

    return app
