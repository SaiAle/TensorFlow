# Lightweight CPU image for the style-transfer CLI / web UI.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer), then the package + web UI extra.
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir -e ".[webui]"

COPY webui ./webui

# Cache the TF-Hub model inside the image so containers start fast.
ENV STYLETRANSFER_CACHE=/app/.cache
RUN python -c "from styletransfer.engine import load_model; load_model()"

EXPOSE 7860
# Bind Gradio to all interfaces so it is reachable from outside the container.
ENV GRADIO_SERVER_NAME=0.0.0.0
CMD ["python", "webui/app.py"]
