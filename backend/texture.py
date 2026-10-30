"""
Texture application logic for unwrapped meshes.
This module handles texture loading and UV mapping.
"""
from PIL import Image, ImageDraw
import numpy as np
from typing import Tuple, Dict, Any, Optional
import os


def load_texture_from_upload(file_path: str) -> Image.Image:
    """Load texture image from file path."""
    image = Image.open(file_path).convert("RGBA")
    return image


def apply_texture_to_uv(
    texture: Image.Image, 
    UV: np.ndarray, 
    width: int = 1024, 
    height: int = 1024
) -> Image.Image:
    """
    Apply texture to UV coordinates.
    
    Args:
        texture: PIL Image to apply
        UV: Nx2 array of UV coordinates in [0,1]
        width: Output texture width
        height: Output texture height
        
    Returns:
        PIL Image with texture applied
    """
    # Resize texture to UV space
    resized_texture = texture.resize((width, height), Image.Resampling.LANCZOS)
    
    # Create output texture with proper padding
    output_texture = Image.new(
        'RGBA', (width, height), (0, 0, 0, 0)
    )

    # Map each UV coordinate to texture pixel
    for i, uv_coord in enumerate(UV):
        u = int(uv_coord[0] * (width - 1))
        v = int((1 - uv_coord[1]) * (height - 1))

        if 0 <= u < width and 0 <= v < height:
            # Get the color from texture and paste to the corresponding position
            output_texture.putpixel((u, v), resized_texture.getpixel((u, v)))

    return output_texture


def generate_texture_coordinate_map(UV: np.ndarray, resolution: int = 1024) -> Image.Image:
    """
    Generate a UV coordinate visualization image.
    
    Args:
        UV: Nx2 array of UV coordinates in [0,1]
        resolution: Image resolution
        
    Returns:
        PIL Image showing UV layout
    """
    img = Image.new('RGB', (resolution, resolution), (255, 255, 255))
    draw = ImageDraw.ImageDraw.Draw(img)
    
    for i, uv in enumerate(UV):
        x = int(uv[0] * (resolution - 1))
        y = int((1 - uv[1]) * (resolution - 1))
        draw.point((x, y), fill=(255, 0, 0))
    
    return img
