"""
Helper utilities for mesh and file processing.
"""
import base64
import io
from typing import Tuple
import numpy as np


def mesh_to_dict(
    V: np.ndarray, F: np.ndarray, UV: np.ndarray = None
) -> dict:
    """Convert mesh arrays to dictionary format."""
    return {
        "vertices": V.tolist(),
        "faces": F.tolist(),
        "uv": UV.tolist() if UV is not None else None,
    }


def dict_to_mesh(mesh_dict: dict) -> tuple:
    """Convert dictionary format to numpy arrays."""
    V = np.array(mesh_dict.get("vertices", []), dtype=np.float64)
    F = np.array(mesh_dict.get("faces", []), dtype=np.int32)
    UV = np.array(mesh_dict.get("uv", []), dtype=np.float64)
    return V, F, UV


def create_mesh_response(
    V: np.ndarray, F: np.ndarray, UV: np.ndarray = None, success: bool = True, message: str = ""
) -> dict:
    """Create standardized mesh response."""
    return {
        "success": success,
        "mesh": {
            "vertices": V.tolist(),
            "faces": F.tolist(),
            "uv": UV.tolist() if UV is not None and success else None,
        },
        "status": {
            "success": success,
            "message": message,
            "uv_method": "LSCM" if UV is not None else None,
        },
    }


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode('utf-8')


def decode_base64_to_image(base64_string: str) -> bytes:
    """Decode base64 string to image bytes. Accepts raw base64 or a data URL."""
    if base64_string.startswith("data:"):
        base64_string = base64_string.split(",", 1)[1]
    return base64.b64decode(base64_string)
