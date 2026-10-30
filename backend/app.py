"""
FastAPI server for SmartUnwrapper.
Handles API requests and delegates to core logic.
"""
import os
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import uuid
import tempfile
import shutil
import numpy as np
from PIL import Image
from pathlib import Path

try:
    from .models import MeshData, UnwrapRequest, TextureRequest, UnwrapResponse, TextureResponse, HealthResponse
    from .utils import encode_image_to_base64, decode_base64_to_image
    from .unwrap import compute_automatic_uv, compute_sam2_uv, load_mesh
    from .texture import apply_texture_to_uv
except ImportError:
    # Allow running directly: python backend/app.py
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from models import MeshData, UnwrapRequest, TextureRequest, UnwrapResponse, TextureResponse, HealthResponse
    from utils import encode_image_to_base64, decode_base64_to_image
    from unwrap import compute_automatic_uv, compute_sam2_uv, load_mesh
    from texture import apply_texture_to_uv

app = FastAPI(
    title="SmartUnwrapper API",
    description="Automatic UV unwrapping and texture application for 3D models",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for frontend
frontend_dir = Path(__file__).parent.parent / "frontend"
sample_model_path = Path(__file__).resolve().parents[1] / "sample" / "HammerChar.fbx"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/")
async def serve_index():
    """Serve the main web UI."""
    return FileResponse(str(frontend_dir / "index.html"))


@app.get("/api/test/sample/{sample_name}")
async def get_test_sample(sample_name: str, request: Request):
    """Serve the bundled visual-test mesh to the local UI without a file picker."""
    client_host = request.client.host if request.client else ""
    if client_host not in {"127.0.0.1", "::1", "::ffff:127.0.0.1"}:
        raise HTTPException(status_code=404, detail="Not found")
    if sample_name != "HammerChar.fbx" or not sample_model_path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(
        sample_model_path,
        media_type="application/octet-stream",
        filename="HammerChar.fbx",
    )

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        available_methods=["SAM2", "LSCM", "Harmonic", "Artist"]
    )

@app.post("/api/unwrap", response_model=UnwrapResponse)
async def unwrap_mesh(
    file: UploadFile = File(...),
    method: str = Form("SAM2"),
    part_detection_thresh: float = Form(0.22, ge=0.05, le=0.8),
    sam2_pred_iou_thresh: float = Form(0.50, ge=0.0, le=1.0),
    sam2_stability_score_thresh: float = Form(0.75, ge=0.0, le=1.0),
):
    """Detect semantic parts with Grounding DINO/SAM2, then create grouped UV charts."""
    try:
        method_u = (method or "SAM2").upper()
        if method_u not in ("SAM2", "LSCM", "HARMONIC", "ARTIST", "SMART"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown unwrap method: {method}. Available: SAM2, LSCM, Harmonic, Artist",
            )
        if method_u == "SMART":
            method_u = "ARTIST"

        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=f".{file.filename.split('.')[-1]}"
        ) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name

        try:
            # Load mesh using unified loader (trimesh + pyassimp fallback)
            V, F = load_mesh(temp_path)
        finally:
            os.unlink(temp_path)

        # SAM2 creates genuine cut topology, so return its split mesh rather
        # than the original shared-vertex topology.
        if method_u == "SAM2":
            UV, uv_status, islands, V, F = compute_sam2_uv(
                V,
                F,
                part_detection_thresh=part_detection_thresh,
                pred_iou_thresh=sam2_pred_iou_thresh,
                stability_score_thresh=sam2_stability_score_thresh,
            )
        else:
            UV, uv_status, islands = compute_automatic_uv(V, F, method=method_u)

        return UnwrapResponse(
            success=True,
            uv=UV.tolist(),
            islands=np.asarray(islands, dtype=np.int32).tolist(),
            mesh_vertices=V.tolist(),
            mesh_faces=F.tolist(),
            status=uv_status,
            message="UV unwrapping completed successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        return UnwrapResponse(
            success=False,
            message=f"Unwrapping failed: {str(e)}"
        )

@app.post("/api/apply-texture", response_model=TextureResponse)
async def apply_texture(request: TextureRequest):
    """Apply texture to unwrapped mesh."""
    try:
        # Load texture image
        texture_bytes = decode_base64_to_image(request.texture_image)
        texture_image = Image.open(io.BytesIO(texture_bytes)).convert("RGBA")
        
        # Convert mesh data to numpy arrays
        V = np.array(request.mesh.vertices, dtype=np.float64)
        F = np.array(request.mesh.faces, dtype=np.int32)
        
        # Use UV coordinates if provided, otherwise compute them
        if request.uv_coordinates is not None:
            UV = np.array(request.uv_coordinates, dtype=np.float64)
        else:
            UV, _, _ = compute_automatic_uv(
            V, F, method="SAM2"
            )
        
        # Apply texture
        textured_image = apply_texture_to_uv(
            texture_image, 
            UV,
            width=1024, 
            height=1024
        )
        
        # Encode as base64 for response
        buffered = io.BytesIO()
        textured_image.save(buffered, format="PNG")
        texture_b64 = encode_image_to_base64(buffered.getvalue())
        
        return TextureResponse(
            success=True,
            texture_url=f"data:image/png;base64,{texture_b64}",
            status={"success": True, "message": "Texture applied successfully"},
            message="Texture applied successfully"
        )
        
    except Exception as e:
        return TextureResponse(
            success=False,
            message=f"Texture application failed: {str(e)}"
        )

@app.get("/api/methods", response_model=List[str])
async def get_methods():
    """List available UV unwrapping methods."""
    return ["SAM2", "LSCM", "Harmonic", "Artist"]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8550)
