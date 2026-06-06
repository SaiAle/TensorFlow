from __future__ import annotations

from pathlib import Path


def keys_path(data_dir: Path) -> Path:
    return data_dir / "api_keys.json"


def load_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    import json

    try:
        return set(json.loads(path.read_text()))
    except Exception:
        return set()


def save_keys(path: Path, keys: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(sorted(keys)))


def create_key(path: Path) -> str:
    keys = load_keys(path)
    new_key = __import__("secrets").token_urlsafe(24)
    keys.add(new_key)
    save_keys(path, keys)
    return new_key


def revoke_key(path: Path, key: str) -> bool:
    keys = load_keys(path)
    if key not in keys:
        return False
    keys.discard(key)
    save_keys(path, keys)
    return True


def has_key(path: Path, key: str) -> bool:
    return key in load_keys(path)
