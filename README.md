# Fast Style Transfer

Arbitrary fast neural style transfer, packaged as a standalone CLI and Python
library. It blends a **content** image with a **style** image using the
pretrained [Magenta arbitrary-image-stylization](https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2)
model from TensorFlow Hub — no training required, runs in seconds on CPU.

> Originally a Google Colab notebook; refactored into a runnable, installable
> project.

## Install

```bash
git clone <this-repo>
cd fast-style-transfer
python -m pip install -e .
```

Requires Python 3.10+. The model (~a few MB) downloads on first run and is
cached under `~/.cache/styletransfer` (override with `STYLETRANSFER_CACHE`).

## Usage

### CLI

```bash
# Local files
styletransfer content.jpg style.jpg -o stylized.png

# URLs work too
styletransfer \
  https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Golden_Gate_Bridge_from_Battery_Spencer.jpg/640px-Golden_Gate_Bridge_from_Battery_Spencer.jpg \
  https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg \
  -o golden_wave.png
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `-o, --output` | `stylized.png` | Output path (`.jpg`/`.jpeg` → JPEG, else PNG) |
| `--content-size` | `384` | Content image size |
| `--style-size` | `256` | Style image size (256 recommended) |
| `--no-style-blur` | off | Skip style-image smoothing |

**Batch mode** — if `style` is a directory, the content image is stylized with
every image in it and results are written to the `-o` output directory:

```bash
styletransfer photo.jpg styles/ -o out/
```

### Web UI

```bash
python -m pip install -e ".[webui]"
python webui/app.py            # open the printed local URL
```

### Docker

```bash
docker build -t fast-style-transfer .
docker run --rm -p 7860:7860 fast-style-transfer   # web UI at http://localhost:7860
```

### Library

```python
from styletransfer import load_image, stylize, save_image

content = load_image("content.jpg", (384, 384))
style = load_image("style.jpg", (256, 256))
result = stylize(content, style)
save_image(result, "out.png")
```

## Development

```bash
python -m pip install -e ".[dev]"
pytest          # image-I/O tests run offline; stylize needs the model
```

## Project layout

```
src/styletransfer/
  engine.py   # load model, stylize()  — the only ML code
  images.py   # load / crop / resize / save (path, URL, or array)
  cli.py      # `styletransfer` command (single + batch mode)
  config.py   # model handle, sizes, cache dir
webui/app.py  # Gradio web UI
tests/        # offline unit tests
Dockerfile    # CPU container serving the web UI
.github/workflows/ci.yml
```

## Roadmap

- [x] **Phase 0/1** — extract notebook → package + working CLI
- [x] **Phase 2** — Gradio web UI, batch mode, GitHub Actions CI, Docker
- [ ] **Phase 3** — FastAPI endpoint, Hugging Face Spaces demo, style-strength control

## Credits

Based on the TensorFlow Hub example and *Exploring the structure of a
real-time, arbitrary neural artistic stylization network* (Ghiasi et al.,
BMVC 2017). Licensed under Apache 2.0.
