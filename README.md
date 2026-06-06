# Style Transfer Prototype

Try it in one click
-------------------
- Google Colab: open `colab_run.ipynb` and run the first cell.
- Local API: see "Deploy anywhere" below.

Prototype implementation of the college assignment in https://github.com/SaiAle/TensorFlow.git, which implemented Fast Arbitrary Neural Style Transfer with TensorFlow and TF-Hub. This repo preserves the core idea and wraps it for production use.

- `stylize` CLI: one-off transfer
- `style-api`: FastAPI REST service
- `run_batch`: worker style grid for automation
- Cached execution: skips recomputation when inputs haven’t changed
- Fallback models: aliases plus explicit TF Hub URLs

Deploy anywhere
---------------
---------------
Option A - local venv:
1. python -m venv .venv
2. source .venv/bin/activate
3. pip install -e '.[dev]'
4. pytest tests/test_core.py -q
5. style-api --host 0.0.0.0 --port 8000

Option B - Docker:
- docker compose up --build
- API available at http://localhost:8000
- Uploaded assets and presets are stored in ./var


Sample APIs / client calls
---------------------------
Transfer one image pair

    stylize path/to/content.jpg path/to/style.jpg --out out.png --cache-dir .cache


List configured models

    stylize --list-models


Run local HTTP API

    style-api --host 0.0.0.0 --port 8000 --reload


HTTP request

    curl -X POST http://localhost:8000/transfer \
      -F "content=@content.jpg" \
      -F "style=@style.jpg" \
      -F "cache=true"


Batch transfer all content + style pairs under two folders

    python -c "from src.worker import run_batch; print(run_batch(Path('samples/content'), Path('samples/style'), Path('out')))"


Run smoke test

    pytest tests/test_core.py -q