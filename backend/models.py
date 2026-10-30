"""
Data models for API responses.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class MeshData(BaseModel):
    """3D mesh data structure."""
    vertices: List[List[float]]
    faces: List[List[int]]
    uv: Optional[List[List[float]]] = None


class UnwrapRequest(BaseModel):
    """Request for UV unwrapping."""
    mesh: MeshData
    method: str = Field(default="SAM2", description="Unwrapping method: SAM2, LSCM, Harmonic or Artist")


class TextureRequest(BaseModel):
    """Request for texture application."""
    mesh: MeshData
    texture_image: str  # Base64 encoded image
    uv_coordinates: Optional[List[List[float]]] = None


class UnwrapResponse(BaseModel):
    """Response from UV unwrapping."""
    success: bool
    uv: Optional[List[List[float]]] = None
    islands: Optional[List[int]] = None  # per-vertex island (UV patch) index
    mesh_vertices: Optional[List[List[float]]] = None
    mesh_faces: Optional[List[List[int]]] = None
    status: Dict[str, Any] = {}
    message: str = ""


class TextureResponse(BaseModel):
    """Response from texture application."""
    success: bool
    texture_url: Optional[str] = None
    status: Dict[str, Any] = {}
    message: str = ""


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    available_methods: List[str] = ["SAM2", "LSCM", "Harmonic", "Artist"]
