from __future__ import annotations

from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from .core import apply_style
from .auth import has_key, keys_path, create_key, revoke_key
from .utils import ensure
from .models import resolve

_auth = HTTPBearer(auto_error=False)


def _resolve_key(data_dir: Path, creds: HTTPAuthorizationCredentials | None) -> str:
    key = (creds.credentials if creds and creds.credentials else "").strip()
    if not key:
        raise HTTPException(401, "Missing API key. Use Authorization: Bearer <key>")
    if not has_key(keys_path(data_dir), key):
        raise HTTPException(403, "Invalid API key")
    return key


class TransferResponse(BaseModel):
    out: str
    bytes: int
    cached: bool


class PresetIn(BaseModel):
    name: str
    style_rel_path: str


class PresetOut(BaseModel):
    id: str
    name: str
    style_rel_path: str


class TransferByPresetRequest(BaseModel):
    content_filename: str
    preset_id: str
    model: str | None = None
    cache: bool = False


def _presets_path(data_dir: Path) -> Path:
    return data_dir / "presets.json"


def _load_presets(path: Path) -> dict[str, PresetOut]:
    if not path.exists():
        return {}
    import json

    raw = json.loads(path.read_text())
    out: dict[str, PresetOut] = {}
    for item in raw:
        out[item["id"]] = PresetOut(**item)
    return out


def _save_presets(path: Path, presets: dict[str, PresetOut]) -> None:
    import json

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
            content_bytes = await content.read()
            style_bytes = await style.read()
            content_path.write_bytes(content_bytes)
            style_path.write_bytes(style_bytes)
        except Exception as exc:
            raise HTTPException(400, f"invalid upload: {exc}") from exc

        out_path = data_dir / f"out_{content_path.stem}_{style_path.stem}.png"
        cache_dir = ensure(data_dir / ".cache") if cache else None

        summary = apply_style(
            content_path,
            style_path,
            out_path,
            model_url=resolve(model),
            cache_dir=cache_dir,
        )
        return TransferResponse(**summary)

    @app.post("/transfer/preset", response_model=TransferResponse)
    async def transfer_by_preset(payload: TransferByPresetRequest) -> TransferResponse:
        presets = _load_presets(presets_file)
        preset = presets.get(payload.preset_id)
        if preset is None:
            raise HTTPException(404, f"unknown preset_id: {payload.preset_id}")

        content_path = (data_dir / payload.content_filename).resolve()
        style_path = (data_dir / preset.style_rel_path).resolve()

        if not content_path.exists():
            raise HTTPException(404, f"missing content file: {payload.content_filename}")
        if not style_path.exists():
            raise HTTPException(500, f"preset style file missing: {preset.style_rel_path}")
        if not str(content_path).startswith(str(data_dir.resolve())):
            raise HTTPException(400, "content_filename must stay inside data_dir")

        out_path = data_dir / f"out_{content_path.stem}__{preset.id}.png"
        cache_dir = ensure(data_dir / ".cache") if payload.cache else None

        summary = apply_style(
            content_path,
            style_path,
            out_path,
            model_url=resolve(payload.model),
            cache_dir=cache_dir,
        )
        return TransferResponse(**summary)

    @app.post("/presets", response_model=PresetOut)
    async def create_preset(payload: PresetIn) -> PresetOut:
        import uuid

        style_path = (data_dir / payload.style_rel_path).resolve()
        if not style_path.exists():
            raise HTTPException(400, f"missing style file: {payload.style_rel_path}")
        if not str(style_path).startswith(str(data_dir.resolve())):
            raise HTTPException(400, "style_rel_path must stay inside data_dir")

        preset_id = uuid.uuid4().hex[:8]
        preset = PresetOut(id=preset_id, name=payload.name, style_rel_path=payload.style_rel_path)
        presets = _load_presets(presets_file)
        presets[preset_id] = preset
        _save_presets(presets_file, presets)
        return preset

    @app.get("/presets", response_model=list[PresetOut])
    def list_presets() -> list[PresetOut]:
        return list(_load_presets(presets_file).values())

    return app
