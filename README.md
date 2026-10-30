# SmartUnwrapper

Automatic UV unwrapping and texture application for 3D models using pure Python LSCM implementation.

## Features

- **Semantic Parts (default)**: Grounding DINO tiny proposes named parts from six mesh views; SAM2 refines the proposals into masks, and geodesic propagation assigns hidden faces. Semantic boundaries become real mesh seams, with each part's UV charts packed together as one editable group.
- **Automatic UV Unwrapping**: Pure Python LSCM / Harmonic / Artist (merged-component) unwrapping — fully permissive-licensed (scipy BSD-3, trimesh MIT), no copyleft
- **Texture Application**: Per-vertex bake of a texture onto the unwrapped UV atlas (client-side drag & drop + server bake)
- **Interactive UV editor (Blender-style)**: island click-select (barycentric hit test), Shift+click multi-select, **A** select all, **G** grab/move, **R** rotate, **S** scale around the island center, Ctrl to snap, click/Enter to confirm, Esc to cancel, **Ctrl+Z** undo, Reset layout, mouse-wheel zoom / middle-drag pan / Home fit
- **Improved automatic packing**: per-island rotation onto its minimum-area bounding rectangle, size proportional to 3D surface area, size-proportional gutters, skyline/shelf sweep for tight packing, atlas rescaled to fill [0,1]^2 — no wasted texture space
- **Web UI**: Drag-and-drop interface for easy model and texture processing
- **Multiple Format Support**: OBJ, STL, PLY, GLTF/GLB, FBX input formats
- **Commercial-Friendly License**: Apache-2.0 with all dependencies allowing commercial use (MIT, BSD, ISC, Apache only - no copyleft licenses)

## Architecture

```
SmartUnwrapper/
├── backend/
│   ├── app.py              # FastAPI server
│   ├── unwrap.py           # Core unwrapping logic using pure Python LSCM
│   ├── texture.py          # Texture application logic
│   ├── models.py           # Data models
│   └── utils.py            # Helper functions
├── frontend/
│   ├── index.html          # Web UI
│   ├── style.css           # Styling
│   └── app.js              # Frontend logic
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container deployment
└── LICENSE                 # Apache-2.0
```

## Quick Start

### Local Development

```bash
# Install dependencies (Python 3.12+ recommended)
pip install -r requirements.txt

# Run the server
python backend/app.py   # listens on http://127.0.0.1:8550

# Open http://localhost:8550 in your browser
```

For a local visual test without opening the file picker, use
`http://127.0.0.1:8550/?testSample=HammerChar`. The test route serves only the
bundled `sample/HammerChar.fbx` file and only to loopback clients.

### Enable SAM2 inference

`backend/third_party/SAM2` contains the SAM2.1 checkpoint and a minimal copy of
the official image-inference source/configuration. Install the runtime
dependencies listed in
[`backend/third_party/SAM2/README.md`](backend/third_party/SAM2/README.md) to
enable multiview segmentation. Without those dependencies, **SAM2 Smart Seams**
uses its artist-style geometry fallback and reports that in `status.sam2_mode`.

The default semantic-parts path also uses the Apache-2.0
[`Grounding DINO tiny`](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
detector. Its checkpoint, tokenizer, processor configuration, and license are
bundled in `backend/third_party/GroundingDINO`; Transformers loads them directly
from that folder without downloading the model at runtime.

### Docker

```bash
docker build -t smartunwrapper .
docker run -p 8550:8550 smartunwrapper
```

## API Endpoints

- `POST /api/unwrap` - Upload 3D model (`file`) + optional `method` (SAM2/LSCM/HARMONIC/ARTIST). SAM2 is the default and returns its split mesh, UV coordinates, per-vertex `islands`, and seam diagnostics.
- `POST /api/apply-texture` - Upload model + texture (JSON), returns baked texture image
- `GET /api/methods` - List available unwrapping methods
- `GET /api/health` - Health check

## FBX support

FBX files are loaded by talking to the [Assimp](https://github.com/assimp/assimp) (BSD-3-Clause) C API directly via ctypes — no extra Python package needed.

- **Windows** — a prebuilt `assimp-vc143-mt.dll` (Assimp 6.0.5) ships with the repo in `backend/libs/` and is loaded automatically at runtime.
- **Linux** — `sudo apt-get install libassimp-dev`
- **macOS** — `brew install assimp`

## License

Apache-2.0 - All dependencies are permissively licensed for commercial use (MIT, BSD, ISC, Apache only - no copyleft licenses)
