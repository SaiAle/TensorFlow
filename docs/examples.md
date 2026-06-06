# API docs

Base URL

- Local: http://localhost:8000

Health

curl -s http://localhost:8000/health

Create API key (requires existing key)

curl -s -X POST http://localhost:8000/keys -H "Authorization: Bearer $API_KEY"

List keys (requires existing key)

curl -s http://localhost:8000/keys -H "Authorization: Bearer $API_KEY"

Create preset (requires API key)

curl -s -X POST http://localhost:8000/presets \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name":"My Brand","style_rel_path":"styles/my-style.png"}'

List presets

curl -s http://localhost:8000/presets

Upload a standalone style asset

curl -s -X POST http://localhost:8000/styles \
  -F "style=@style.png"

Upload content + style and transfer

curl -s -X POST http://localhost:8000/transfer \
  -F "content=@content.jpg" \
  -F "style=@style.png" \
  -F "cache=true"

Transfer with preset

curl -s -X POST http://localhost:8000/transfer/preset \
  -F "content=@content.jpg" \
  -F "preset_id=<preset_id>"

Response example

{
  "out": "out_content_style.png",
  "bytes": 512000,
  "cached": false
}
