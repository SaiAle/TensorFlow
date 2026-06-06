from __future__ import annotations

DEFAULT_MODEL = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"

FALLBACK_MODELS: list[dict[str, str]] = [
    {"alias": "magenta", "url": "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"},
    {"alias": "fast-style", "url": "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"},
]


def resolve(url: str | None = None, alias: str | None = None) -> str:
    if url:
        return url
    if alias:
        for model in FALLBACK_MODELS:
            if model["alias"] == alias:
                return model["url"]
    return DEFAULT_MODEL
