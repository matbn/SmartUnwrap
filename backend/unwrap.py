"""
Core UV unwrapping logic using pure Python implementation.
This module contains geometry processing operations with permissive licenses (MIT/BSD).
No copyleft dependencies - replaces libigl with pure Python/scipy implementation.

UV core design (vectorized, O(F) assembly):
- Per-face barycentric gradients assemble both the cotan Dirichlet matrix L
  and the antisymmetric conformal coupling K in a single pass.
- True LSCM: solve the symmetric block system  [[L, K], [-K, L]] [u; v] = 0
  (Levy et al. conformal energy), pinning 2 well-separated boundary vertices
  per connected component.
- Harmonic: boundary loops are laid out by LSCM, then the interior is
  relaxed with a Laplace solve.
- Connected components are solved independently; closed components (no
  boundary) fall back to a lat-long projection on principal axes.
- Each island is rotated onto its minimum-area bounding rectangle, kept
  proportional to its 3D surface area, given a gutter proportional to its
  size, and shelf-packed with a width sweep that maximizes atlas usage.
  The finished atlas is rescaled to fill [0,1]^2 with a small border.
- The per-vertex island index is returned so clients can select and
  transform individual UV islands.
"""
import ctypes
import os
import time
import warnings
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components
import trimesh
try:
    from .sam2_unwrap import discover_sam2_regions, split_on_region_seams, _semantic_part_regions
except ImportError:  # Allows ``python backend/app.py`` imports.
    from sam2_unwrap import discover_sam2_regions, split_on_region_seams, _semantic_part_regions

# ============================================================
# Direct Assimp ctypes loader (FBX support)
# ------------------------------------------------------------
# We bind the Assimp C API (aiImportFile / aiReleaseImport) directly
# instead of using the pyassimp PyPI package: pyassimp 5.2.5 declares
# assimp 5.x ctypes structs, which crash with "NULL pointer access"
# against the official assimp 6.x Windows DLLs (ABI changed).
# Only the struct HEADS are declared here — these offsets are
# ABI-stable across assimp 5.x and 6.x (new fields are appended at
# the end of each struct).
# Assimp is BSD-3-Clause licensed: https://github.com/assimp/assimp
# ============================================================
_LIBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")

# aiProcess flags (stable values from assimp postprocess.h)
_AI_PROCESS_JOIN_IDENTICAL_VERTICES = 0x2
_AI_PROCESS_TRIANGULATE = 0x8
_AI_PROCESS_PRETRANSFORM_VERTICES = 0x100
_AI_PROCESS_SORT_BY_PTYPE = 0x8000


class _AiSceneHead(ctypes.Structure):
    _fields_ = [
        ("mFlags", ctypes.c_uint),                       # offset 0
        ("mRootNode", ctypes.c_void_p),                  # offset 8
        ("mNumMeshes", ctypes.c_uint),                   # offset 16
        ("mMeshes", ctypes.POINTER(ctypes.c_void_p)),    # offset 24
    ]


class _AiFaceHead(ctypes.Structure):
    _fields_ = [
        ("mNumIndices", ctypes.c_uint),                  # offset 0
        ("mIndices", ctypes.POINTER(ctypes.c_uint)),     # offset 8
    ]


class _AiMeshHead(ctypes.Structure):
    _fields_ = [
        ("mPrimitiveTypes", ctypes.c_uint),              # offset 0
        ("mNumVertices", ctypes.c_uint),                 # offset 4
        ("mNumFaces", ctypes.c_uint),                    # offset 8
        ("mVertices", ctypes.c_void_p),                  # offset 16
        ("mNormals", ctypes.c_void_p),
        ("mTangents", ctypes.c_void_p),
        ("mBitangents", ctypes.c_void_p),
        ("mColors", ctypes.c_void_p * 8),
        ("mTextureCoords", ctypes.c_void_p * 8),
        ("mNumUVComponents", ctypes.c_uint * 8),
        ("mFaces", ctypes.POINTER(_AiFaceHead)),
    ]


def _load_assimp_dll():
    """Locate and load the Assimp shared library."""
    if os.name == "nt":
        candidates = ("assimp-vc143-mt.dll", "assimp.dll")
        # 1. bundled with the repo (backend/libs)
        for name in candidates:
            path = os.path.join(_LIBS_DIR, name)
            if os.path.exists(path):
                try:
                    return ctypes.WinDLL(path)
                except OSError:
                    pass
        # 2. anywhere on the system PATH
        for name in candidates:
            try:
                return ctypes.WinDLL(name)
            except OSError:
                pass
    else:
        for name in ("libassimp.so.6", "libassimp.so", "libassimp.dylib"):
            try:
                return ctypes.CDLL(name)
            except OSError:
                pass
    raise OSError(
        "Assimp library not found. On Windows, the DLL is bundled in backend/libs/ "
        "(assimp-vc143-mt.dll). On Linux: apt-get install libassimp-dev; macOS: brew install assimp"
    )


def load_fbx_via_assimp(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load mesh geometry from an FBX file (or any Assimp-supported format)
    via the Assimp C API. Returns (V, F) with all node transforms baked
    to world space and all scene meshes concatenated.
    """
    dll = _load_assimp_dll()
    dll.aiGetErrorString.restype = ctypes.c_char_p
    dll.aiGetErrorString.argtypes = []
    dll.aiImportFile.restype = ctypes.c_void_p
    dll.aiImportFile.argtypes = [ctypes.c_char_p, ctypes.c_uint]
    dll.aiReleaseImport.restype = None
    dll.aiReleaseImport.argtypes = [ctypes.c_void_p]

    flags = (_AI_PROCESS_JOIN_IDENTICAL_VERTICES |
             _AI_PROCESS_TRIANGULATE |
             _AI_PROCESS_PRETRANSFORM_VERTICES |
             _AI_PROCESS_SORT_BY_PTYPE)

    scene_ptr = dll.aiImportFile(os.fsencode(file_path), flags)
    if not scene_ptr:
        err = dll.aiGetErrorString()
        raise ValueError(
            f"Assimp failed to import '{os.path.basename(file_path)}': "
            f"{err.decode(errors='replace') if err else 'unknown error'}"
        )

    try:
        scene = ctypes.cast(scene_ptr, ctypes.POINTER(_AiSceneHead)).contents
        n_meshes = scene.mNumMeshes
        if n_meshes == 0:
            raise ValueError("File contains no meshes.")

        mesh_ptrs = ctypes.cast(scene.mMeshes, ctypes.POINTER(ctypes.c_void_p))[:n_meshes]

        verts_parts: list = []
        faces_parts: list = []
        offset = 0
        for mp in mesh_ptrs:
            mesh = ctypes.cast(mp, ctypes.POINTER(_AiMeshHead)).contents
            nV, nF = int(mesh.mNumVertices), int(mesh.mNumFaces)
            if nV == 0 or mesh.mVertices is None:
                continue

            vbuf = (ctypes.c_float * (nV * 3)).from_address(mesh.mVertices)
            V = np.frombuffer(vbuf, dtype=np.float32).reshape(nV, 3).astype(np.float64)
            verts_parts.append(V)

            if mesh.mFaces is not None:
                tri_idx: list = []
                face_arr = mesh.mFaces
                for i in range(nF):
                    f = face_arr[i]
                    if f.mNumIndices == 3 and f.mIndices is not None:
                        idx = f.mIndices[:3]
                        tri_idx.append((idx[0], idx[1], idx[2]))
                if tri_idx:
                    faces_parts.append(np.array(tri_idx, dtype=np.int32) + offset)
            offset += nV

        if not verts_parts or not faces_parts:
            raise ValueError("No triangle geometry found in the file.")

        V = np.ascontiguousarray(np.vstack(verts_parts))
        F = np.ascontiguousarray(np.vstack(faces_parts))
        return V, F
    finally:
        dll.aiReleaseImport(scene_ptr)


def load_mesh(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a 3D mesh from a file, supporting OBJ, STL, PLY, GLTF, FBX, and more.

    Priority:
    1. trimesh (OBJ, STL, PLY, GLTF, GLB) - MIT license
    2. Assimp via direct ctypes (FBX) - BSD-3-Clause license

    Args:
        file_path: Path to the 3D model file

    Returns:
        V: Nx3 array of vertex positions
        F: Mx3 array of face indices
    """
    ext = os.path.splitext(file_path)[1].lower()

    # Step 1: Try trimesh for supported formats (MIT license)
    try:
        mesh = trimesh.load(file_path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            # Concatenate all geometries in the scene
            meshes = []
            for obj in mesh.geometry.values():
                if isinstance(obj, trimesh.Trimesh):
                    meshes.append(obj)
            if meshes:
                mesh = trimesh.util.concatenate(meshes)
        if isinstance(mesh, trimesh.Trimesh):
            V = np.asarray(mesh.vertices, dtype=np.float64)
            F = np.asarray(mesh.faces, dtype=np.int32)
            V = np.ascontiguousarray(V, dtype=np.float64)
            F = np.ascontiguousarray(F, dtype=np.int32)
            if len(V) > 0 and len(F) > 0:
                return V, F
    except Exception:
        pass  # Fall back to next method

    # Step 2: Assimp (BSD-3) for FBX via direct ctypes binding
    if ext == '.fbx':
        try:
            V, F = load_fbx_via_assimp(file_path)
            if len(V) > 0 and len(F) > 0:
                return np.ascontiguousarray(V, dtype=np.float64), \
                       np.ascontiguousarray(F, dtype=np.int32)
        except Exception as e:
            raise ValueError(
                f"FBX loading failed: {e}. "
                f"The Assimp DLL (assimp-vc143-mt.dll) is bundled in backend/libs/ — "
                f"see README 'FBX support' if it is missing."
            ) from e

    # If we get here, no loader succeeded
    raise ValueError(
        f"Cannot load file: {file_path}. "
        f"Supported formats: .obj, .stl, .ply, .gltf, .glb (via trimesh) + .fbx (via Assimp)"
    )


def load_mesh_from_buffer(
    vertices: np.ndarray, faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Load mesh from numpy arrays."""
    V = np.asarray(vertices, dtype=np.float64)
    F = np.asarray(faces, dtype=np.int32)
    return V, F


# ------------------------------------------------------------------
# Vectorized UV core (O(F) assembly, per-component solves)
# ------------------------------------------------------------------

def _cube_projection(V: np.ndarray) -> np.ndarray:
    """Last-resort fallback: project the mesh onto its two dominant axes."""
    V = np.asarray(V, dtype=np.float64)
    if len(V) == 0:
        return np.zeros((0, 2))
    span = V.max(axis=0) - V.min(axis=0)
    order = np.argsort(span)[::-1]
    return normalize_uv_coordinates(V[:, [order[0], order[1]]].copy())


def _cotan_matrices(V: np.ndarray, F: np.ndarray) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    """
    Assemble in vectorized O(F) time:
      L = sum over faces of A_T * G_T^T G_T       (cotan Dirichlet matrix)
      K = sum over faces of A_T * G_T^T [n x] G_T (antisymmetric conformal
          coupling; [n x] is the +90 degree in-plane rotation)
    where G_T holds the 3D gradient vectors of the barycentric basis
    functions of face T. The LSCM block system is then
        [[L, K], [-K, L]] [u; v] = 0.
    """
    F = np.asarray(F, dtype=np.int64)
    N = len(V)
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    p0, p1, p2 = V[i0], V[i1], V[i2]

    n = np.cross(p1 - p0, p2 - p0)
    twice_area = np.linalg.norm(n, axis=1)
    degenerate = twice_area < 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        n_hat = n / twice_area[:, None]
    n_hat[degenerate] = 0.0
    area = np.where(degenerate, 0.0, 0.5 * twice_area)

    # gradient vectors of the barycentric basis functions (per face, per corner)
    g = np.empty((len(F), 3, 3), dtype=np.float64)
    g[:, 0] = np.cross(n_hat, p2 - p1)
    g[:, 1] = np.cross(n_hat, p0 - p2)
    g[:, 2] = np.cross(n_hat, p1 - p0)
    g /= np.where(degenerate, 1.0, 2.0 * twice_area)[:, None, None]

    L_face = np.einsum("fai,fbi->fab", g, g) * area[:, None, None]
    K_face = np.einsum(
        "fai,fbi->fab", g, np.cross(n_hat[:, None, :], g)
    ) * area[:, None, None]

    rows = np.repeat(F, 3, axis=1).ravel()
    cols = np.tile(F, (1, 3)).ravel()
    L = sp.coo_matrix((L_face.ravel(), (rows, cols)), shape=(N, N)).tocsr()
    K = sp.coo_matrix((K_face.ravel(), (rows, cols)), shape=(N, N)).tocsr()
    # remove floating-point drift so L stays symmetric, K antisymmetric
    L = (L + L.T) * 0.5
    K = (K - K.T) * 0.5
    return L, K


def _boundary_loops(F: np.ndarray) -> Tuple[int, list]:
    """
    Boundary edges = edges used by exactly one triangle. Walk them into
    ordered closed loops. Returns (n_boundary_edges, loops) where each loop
    is a list of vertex indices in walking order.
    """
    F = np.asarray(F, dtype=np.int64)
    if len(F) == 0:
        return 0, []
    n_max = int(F.max()) + 1
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    E.sort(axis=1)
    keys = E[:, 0] * n_max + E[:, 1]
    uniq, counts = np.unique(keys, return_counts=True)
    bnd = uniq[counts == 1]
    if len(bnd) == 0:
        return 0, []
    a = (bnd // n_max).tolist()
    b = (bnd % n_max).tolist()

    adj: Dict[int, list] = defaultdict(list)
    for x, y in zip(a, b):
        adj[x].append(y)
        adj[y].append(x)

    loops: list = []
    unvisited = set(adj.keys())
    while unvisited:
        start = unvisited.pop()
        loop = [start]
        prev, cur = -1, start
        while True:
            nxt = -1
            for x in adj[cur]:
                if x != prev and x in unvisited:
                    nxt = x
                    break
            if nxt < 0:
                break
            unvisited.discard(nxt)
            loop.append(nxt)
            prev, cur = cur, nxt
        loops.append(loop)
    return int(len(bnd)), loops


def _vertex_components(N: int, F: np.ndarray) -> Tuple[int, np.ndarray]:
    """Connected components of the vertex adjacency graph."""
    if N == 0:
        return 0, np.zeros(0, dtype=np.int64)
    if len(F) == 0:
        return N, np.arange(N, dtype=np.int64)
    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    G = sp.coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(N, N))
    n_comp, labels = connected_components(G, directed=False)
    return int(n_comp), labels


def _latlong_uv(Vc: np.ndarray) -> np.ndarray:
    """
    Lat-long projection around the component's principal axes. Used for
    closed (sphere-like) components, where a conformal map would require
    cutting the surface open.
    """
    Vc = np.asarray(Vc, dtype=np.float64)
    if len(Vc) < 3:
        return np.full((len(Vc), 2), 0.5)
    X = Vc - Vc.mean(axis=0)
    cov = (X.T @ X) / len(X)
    _, axes = np.linalg.eigh(cov)
    pole = axes[:, 2]                      # largest-variance axis becomes the pole
    e1 = axes[:, 0]
    e2 = np.cross(pole, e1)
    x, y, z = X @ e1, X @ e2, X @ pole
    rho = np.maximum(np.sqrt(x * x + y * y + z * z), 1e-12)
    u = np.arctan2(y, x) / (2.0 * np.pi) + 0.5
    v = np.arccos(np.clip(z / rho, -1.0, 1.0)) / np.pi
    return np.column_stack([u, v])


def _min_area_rect_rotation(uv: np.ndarray) -> float:
    """
    Angle (radians) that best aligns a patch with its minimum-area bounding
    rectangle. Evaluated on the convex hull (rotating-calipers style: the
    optimal rectangle always has one side collinear with a hull edge).
    """
    uv = np.asarray(uv, dtype=np.float64)
    n_pts = len(uv)
    if n_pts < 3:
        return 0.0
    if n_pts <= 8:
        pts = uv                       # no hull needed, and no Qhull call cost
    else:
        try:
            from scipy.spatial import ConvexHull
            pts = uv[ConvexHull(uv).vertices]
        except Exception:
            pts = uv
    if len(pts) < 3:
        return 0.0
    # candidate directions: hull edges, subsampled for very dense hulls
    d = np.diff(np.vstack([pts, pts[:1]]), axis=0)
    angs = np.arctan2(d[:, 1], d[:, 0])
    if len(angs) > 32:
        angs = angs[np.linspace(0, len(angs) - 1, 32).astype(int)]
    angs = np.concatenate([angs, angs + 0.5 * np.pi])

    c, s = np.cos(angs), np.sin(angs)
    ux, uy = uv[:, 0][None, :], uv[:, 1][None, :]
    x = ux * c[:, None] + uy * s[:, None]
    y = uy * c[:, None] - ux * s[:, None]
    areas = (x.max(axis=1) - x.min(axis=1)) * (y.max(axis=1) - y.min(axis=1))
    return float(angs[int(np.argmin(areas))])


def _rotate_uv(uv: np.ndarray, angle: float) -> np.ndarray:
    """Rotate UVs by -angle so the minimum-area rectangle becomes axis aligned."""
    if abs(angle) < 1e-12:
        return uv
    c, s = np.cos(angle), np.sin(angle)
    return np.column_stack([
        uv[:, 0] * c + uv[:, 1] * s,
        -uv[:, 0] * s + uv[:, 1] * c,
    ])


def _prepare_island(uv_c: np.ndarray, area_share: float) -> Tuple[np.ndarray, float, float]:
    """
    Turn a raw island layout into an axis-aligned atlas box:
      1. rotate to its minimal bounding rectangle (tight boxes pack better),
      2. normalize the shape so its longest side is 1 (aspect preserved),
      3. scale by sqrt(3D area share) so atlas space tracks surface area.
    Returns (uv_boxed, width, height) with the island's lower-left at (0, 0).
    """
    s_i = max(area_share, 0.0) ** 0.5
    if len(uv_c) == 0 or not np.all(np.isfinite(uv_c)):
        return np.zeros_like(uv_c), 0.02 * s_i + 1e-3, 0.02 * s_i + 1e-3

    uv_r = _rotate_uv(uv_c, _min_area_rect_rotation(uv_c))
    mn = uv_r.min(axis=0)
    size = uv_r.max(axis=0) - mn
    maxdim = float(size.max())
    if maxdim < 1e-12:
        return np.zeros_like(uv_r), 0.02 * s_i + 1e-3, 0.02 * s_i + 1e-3
    uv_norm = (uv_r - mn) / maxdim
    return (uv_norm * s_i,
            float(size[0] / maxdim * s_i),
            float(size[1] / maxdim * s_i))


def _triangle_uv(Vc: np.ndarray) -> np.ndarray:
    """
    Exact conformal layout of a single triangle: the isometric one. LSCM has
    zero conformal energy on a triangle, so this is the same result a solve
    would give - without paying for a sparse linear solve per triangle.
    """
    p0, p1, p2 = (np.asarray(p, dtype=np.float64) for p in Vc[:3])
    e = p1 - p0
    L = float(np.linalg.norm(e))
    if L < 1e-12:
        return np.zeros((3, 2))
    ex = e / L
    d = p2 - p0
    return np.array([
        [0.0, 0.0],
        [L, 0.0],
        [float(d @ ex), float(np.linalg.norm(np.cross(ex, d)))],
    ])


def _pack_patches(
    boxes: list, target_w: float = 1.0, pads: Optional[list] = None
) -> Tuple[list, Tuple[float, float]]:
    """
    Shelf-pack patches into a rectangle of width `target_w` without overlap.
    `pads[i]` is extra gutter added around patch i on every side. Returns
    (placements, (extent_w, extent_h)); placements locate the *inner* box, so
    the final transform for patch i is  uv_global = uv_local + placement_i.
    """
    n = len(boxes)
    if pads is None:
        pads = [0.0] * n
    outer = [(boxes[i][0] + 2.0 * pads[i], boxes[i][1] + 2.0 * pads[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: (-outer[i][1], -outer[i][0]))
    placements = [(0.0, 0.0)] * n
    x = y = row_h = 0.0
    max_w = 0.0
    for i in order:
        w, h = outer[i]
        if x > 0.0 and x + w > target_w + 1e-9:
            y += row_h
            x = 0.0
            row_h = 0.0
        placements[i] = (x + pads[i], y + pads[i])
        x += w
        row_h = max(row_h, h)
        max_w = max(max_w, x)
    return placements, (max_w, y + row_h)


def _skyline_pack(outer: list, target_w: float) -> Tuple[list, Tuple[float, float]]:
    """
    Bottom-left skyline packing: every box is dropped into the lowest (then
    leftmost) not-yet-covered slot of the current skyline. Much tighter than
    shelf packing for atlases that mix big islands with hundreds of scraps.
    Returns (placements, (extent_w, extent_h)).
    """
    n = len(outer)
    order = sorted(range(n), key=lambda i: (-outer[i][1], -outer[i][0]))
    placements = [(0.0, 0.0)] * n
    xs: list = [0.0]
    ys: list = [0.0]
    max_x = 0.0
    max_y = 0.0

    for i in order:
        w, h = outer[i]
        best = None
        m = len(xs)
        for k in range(m):
            x = xs[k]
            if x + w > target_w + 1e-9:
                break
            y = ys[k]
            j = k + 1
            lim = x + w - 1e-12
            while j < m and xs[j] < lim:
                if ys[j] > y:
                    y = ys[j]
                j += 1
            if best is None or y < best[1] - 1e-12 or (
                y < best[1] + 1e-12 and x < best[0]
            ):
                best = (x, y)
                if y <= 1e-12:
                    break
        if best is None:                      # box wider than the shelf
            best = (0.0, max(ys))
        x, y = best
        placements[i] = (x, y)

        x2 = x + w
        H = y + h
        if H > max_y:
            max_y = H
        if x2 > max_x:
            max_x = x2

        # height under x2 before the update becomes the tail segment's height
        tail = ys[-1]
        for k in range(m):
            end = xs[k + 1] if k + 1 < m else target_w
            if xs[k] <= x2 < end or (k == m - 1 and x2 >= end):
                tail = ys[k]
                break

        nxs, nys = [], []
        for k in range(m):
            s = xs[k]
            end = xs[k + 1] if k + 1 < m else target_w
            if end <= x + 1e-12 or s >= x2 - 1e-12:
                nxs.append(s)
                nys.append(ys[k])
        nxs.append(x)
        nys.append(H)
        if x2 < target_w - 1e-12:
            nxs.append(x2)
            nys.append(tail)

        idx = sorted(range(len(nxs)), key=nxs.__getitem__)
        xs = [nxs[k] for k in idx]
        ys = [nys[k] for k in idx]
        ax, ay = [xs[0]], [ys[0]]        # merge equal-height neighbours
        for k in range(1, len(xs)):
            if abs(ys[k] - ay[-1]) < 1e-12:
                continue
            ax.append(xs[k])
            ay.append(ys[k])
        xs, ys = ax, ay

    return placements, (max_x, max_y)


def _finalize_atlas(
    patch_meta: list, n_verts: int, border: float = 0.008
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Pack the prepared islands into an atlas and normalize it into [0, 1]^2.

    Improvements over a single fixed-width shelf:
      * every island keeps a gutter proportional to its own size (so tiny
        islands still get a usable margin without eating the atlas),
      * the shelf width is swept and the most compact result is kept,
      * the atlas is uniformly rescaled to fill [0, 1] minus a border, so no
        texture space is wasted by an arbitrary global scale cap.

    Returns (uv, island_of, stats) where island_of[v] is the island index or
    -1 for vertices that belong to no island.
    """
    uv = np.zeros((n_verts, 2))
    island_of = np.full(n_verts, -1, dtype=np.int32)
    if not patch_meta:
        return uv, island_of, {"islands": 0, "packing_efficiency": 0.0}

    boxes = [(w, h) for _, _, w, h in patch_meta]
    inner_area = float(sum(max(w, 0.0) * max(h, 0.0) for w, h in boxes))
    n = len(boxes)

    # gutter: 6% of each island's short side, bounded relative to the mean
    # island size so that neither giants nor single triangles dominate
    mean_size = float(np.sqrt(max(inner_area, 1e-18) / max(n, 1)))
    pads = []
    for w, h in boxes:
        p = 0.06 * min(w, h)
        p = min(max(p, 0.04 * mean_size), 0.22 * mean_size)
        pads.append(p)

    base_w = max(1e-6, np.sqrt(max(inner_area, 1e-18)))
    outer = [(boxes[i][0] + 2.0 * pads[i], boxes[i][1] + 2.0 * pads[i]) for i in range(n)]
    best = None          # (max_extent, placements, built_from_padded_boxes)

    # skyline packing is noticeably tighter, but O(n * skyline_segments); on
    # huge island counts fall back to the (linear) shelf packer.
    if n <= 2500:
        for factor in (0.8, 0.95, 1.1, 1.35, 1.7):
            pl, ext = _skyline_pack(outer, target_w=base_w * factor)
            if best is None or max(ext) < best[0] - 1e-12:
                best = (max(ext), pl, True)
    for factor in (0.55, 0.7, 0.85, 1.0, 1.12, 1.28, 1.5, 1.75, 2.0):
        pl, ext = _pack_patches(boxes, target_w=base_w * factor, pads=pads)
        if best is None or max(ext) < best[0] - 1e-12:
            best = (max(ext), pl, False)

    if best is None:
        placements = [(0.0, 0.0)] * n
    elif best[2]:
        placements = [(p[0] + pads[i], p[1] + pads[i]) for i, p in enumerate(best[1])]
    else:
        placements = best[1]

    for i, ((gidx, uv_raw, _, _), (px, py)) in enumerate(zip(patch_meta, placements)):
        uv[gidx] = uv_raw + np.array([px, py])
        island_of[gidx] = i

    mask = island_of >= 0
    efficiency = 0.0
    if mask.any():
        pts = uv[mask]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        span = hi - lo
        extent = float(span.max())
        if extent > 1e-12:
            s = (1.0 - 2.0 * border) / extent
            uv[mask] = (uv[mask] - lo) * s + border
            efficiency = inner_area / max(float(span[0] * span[1]), 1e-18)

    return uv, island_of, {
        "islands": int(n),
        "packing_efficiency": round(float(min(efficiency, 1.0)), 3),
    }


def _solve_or_none(M: sp.csc_matrix, rhs: np.ndarray) -> Optional[np.ndarray]:
    """spsolve wrapper that turns 'Matrix is exactly singular' into None."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            sol = spla.spsolve(M, rhs)
        except Exception:
            return None
    if sol is None or not np.all(np.isfinite(sol)):
        return None
    return sol


def _lscm_solve_component(
    Vc: np.ndarray, Lc: sp.csr_matrix, Kc: sp.csr_matrix, loops: list
) -> Tuple[np.ndarray, bool]:
    """
    LSCM for one open component: pin the two most separated vertices of the
    longest boundary loop, then solve the symmetric conformal block system
        [[L, K], [-K, L]] [u; v] = 0
    with the pinned dofs eliminated. Returns (uv, ok).
    """
    m = len(Vc)
    loop = max(loops, key=len)
    if len(loop) < 2:
        return _latlong_uv(Vc), False

    d = np.linalg.norm(Vc[loop] - Vc[loop[0]], axis=1)
    far = loop[int(np.argmax(d))]
    pinned = np.array([loop[0], far], dtype=np.int64)
    target = np.array([[0.0, 0.0], [1.0, 0.0]])

    free = np.setdiff1d(np.arange(m), pinned)
    if len(free) == 0:
        uv = np.zeros((m, 2))
        uv[pinned] = target
        return uv, True

    Lff = Lc[free][:, free]
    Lfp = Lc[free][:, pinned]
    Kff = Kc[free][:, free]
    Kfp = Kc[free][:, pinned]

    M = sp.bmat([[Lff, Kff], [-Kff, Lff]], format="csc")
    u_p, v_p = target[:, 0], target[:, 1]
    rhs = np.concatenate([
        -(Lfp @ u_p + Kfp @ v_p),
        Kfp @ u_p - Lfp @ v_p,
    ])

    sol = _solve_or_none(M, rhs)
    if sol is None:
        return _latlong_uv(Vc), False

    uv = np.zeros((m, 2))
    uv[pinned] = target
    uv[free, 0] = sol[:len(free)]
    uv[free, 1] = sol[len(free):]
    return uv, True


def _harmonic_solve_component(
    Vc: np.ndarray, Lc: sp.csr_matrix, Kc: sp.csr_matrix, loops: list
) -> Tuple[np.ndarray, bool]:
    """
    Harmonic map for one open component: the boundary layout comes from
    LSCM (proven fold-free here), then a Laplace solve harmonically
    smooths the interior with all boundary vertices as Dirichlet data.
    Classic circle pinning is not used: pinning interior hole loops to
    fixed circles folds heavily textured, multi-hole patches.
    Returns (uv, ok).
    """
    m = len(Vc)
    if not loops:
        return _latlong_uv(Vc), True

    # 1) boundary positions from LSCM
    uv_lscm, ok_lscm = _lscm_solve_component(Vc, Lc, Kc, loops)
    if not np.all(np.isfinite(uv_lscm)):
        uv_lscm = _latlong_uv(Vc)

    bnd_set = sorted({v for lp in loops for v in lp})
    pinned = np.array(bnd_set, dtype=np.int64)
    target = uv_lscm[pinned]

    free = np.setdiff1d(np.arange(m), pinned)
    if len(free) == 0:
        return uv_lscm, ok_lscm

    # 2) harmonic interior over the LSCM boundary
    A = Lc[free][:, free].tocsc()
    rhs = -(Lc[free][:, pinned] @ target)          # (Nf, 2): u and v at once
    sol = _solve_or_none(A, rhs)
    if sol is None:
        return uv_lscm, ok_lscm

    uv = np.zeros((m, 2))
    uv[pinned] = target
    uv[free] = sol
    return uv, ok_lscm


def _unwrap_components(
    V: np.ndarray, F: np.ndarray, method: str
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Shared pipeline for both methods: assemble matrices once, then solve
    each connected component independently and pack the patches into a
    non-overlapping UV atlas inside [0, 1]^2.
    """
    t0 = time.time()
    N = len(V)
    method_name = "LSCM (Pure Python)" if method == "LSCM" else "Harmonic (Pure Python)"
    status: Dict[str, Any] = {
        "method": method_name,
        "valid": True,
        "message": "",
        "components": 0,
        "boundary_loops": 0,
        "patches_fallback": 0,
        "time_seconds": 0.0,
    }

    try:
        if N < 3 or len(F) == 0:
            raise ValueError("mesh has too few vertices or faces")

        L, K = _cotan_matrices(V, F)
        n_comp, labels = _vertex_components(N, F)
        _, loops = _boundary_loops(F)

        loop_by_comp: Dict[int, list] = defaultdict(list)
        for lp in loops:
            loop_by_comp[int(labels[lp[0]])].append(lp)

        uv = np.zeros((N, 2))
        patch_meta: list = []       # (gidx, uv_shifted, w, h) per solved patch
        n_fallback = 0

        face_comp = labels[F[:, 0]]
        faces_sorted = F[np.argsort(face_comp, kind="stable")]
        counts = np.bincount(face_comp, minlength=n_comp)
        starts = np.concatenate([[0], np.cumsum(counts)])

        # 3D area per component -> atlas cell size is proportional to it
        e1v = V[F[:, 1]] - V[F[:, 0]]
        e2v = V[F[:, 2]] - V[F[:, 0]]
        face_area = 0.5 * np.linalg.norm(np.cross(e1v, e2v), axis=1)
        comp_area = np.bincount(face_comp, weights=face_area, minlength=n_comp)
        area_total = float(comp_area.sum())

        local_of = np.empty(N, dtype=np.int64)
        for c in np.unique(face_comp):
            c = int(c)
            gidx = np.flatnonzero(labels == c)
            local_of[gidx] = np.arange(len(gidx))
            Vc = V[gidx]
            Fc = local_of[faces_sorted[starts[c]:starts[c + 1]]]
            loops_c = loop_by_comp.get(c, [])
            # loops carry GLOBAL vertex ids; the solvers index local arrays
            loops_c = [[int(local_of[v]) for v in lp] for lp in loops_c]

            ok = True
            try:
                if len(gidx) == 3:
                    uv_c = _triangle_uv(Vc)      # exact, no solve needed
                elif loops_c:
                    Lc = L[gidx][:, gidx]
                    Kc = K[gidx][:, gidx]
                    if method == "LSCM":
                        uv_c, ok = _lscm_solve_component(Vc, Lc, Kc, loops_c)
                    else:
                        uv_c, ok = _harmonic_solve_component(Vc, Lc, Kc, loops_c)
                else:
                    uv_c = _latlong_uv(Vc)     # closed component
            except Exception:
                uv_c, ok = _latlong_uv(Vc), False

            if not np.all(np.isfinite(uv_c)):
                uv_c, ok = _latlong_uv(Vc), False
            if not ok:
                n_fallback += 1

            # Rotate the island to its minimal bounding box, normalize its shape
            # to max dimension 1 (aspect preserved) and scale by sqrt of its 3D
            # area share so atlas space stays proportional to surface area.
            f_area = (float(comp_area[c]) / area_total) if area_total > 1e-12 else 0.0
            uv_box, w_i, h_i = _prepare_island(uv_c, f_area)
            patch_meta.append((gidx, uv_box, w_i, h_i))

        uv, island_of, pack_stats = _finalize_atlas(patch_meta, N)

        status.update({
            "components": int(n_comp),
            "boundary_loops": int(len(loops)),
            "patches_fallback": int(n_fallback),
            "islands": pack_stats.get("islands", 0),
            "packing_efficiency": pack_stats.get("packing_efficiency", 0.0),
            "message": "UV computation completed successfully",
            "time_seconds": round(time.time() - t0, 2),
        })
        return uv, status, island_of

    except Exception as e:
        status["valid"] = False
        status["message"] = f"{method_name} failed ({e}); used box projection fallback"
        return _cube_projection(V), status, np.zeros(N, dtype=np.int32)


def _artist_unwrap_components(
    V: np.ndarray, F: np.ndarray, face_groups: Optional[np.ndarray] = None,
    unclassified_group_id: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    """
    "Artist" unwrap: reduce the excessive island count produced by the
    default per-component LSCM by (1) dropping degenerate triangles,
    (2) merging small connected components into their nearest neighbour
    and (3) stitching merged groups with a few synthetic bridge triangles
    so each group unfolds as one coherent patch instead of many tiny
    islands. Then LSCM is run per group and patches are packed. If semantic
    face groups are supplied, disconnected pieces can merge only within the
    same detected part.
    """
    t0 = time.time()
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    N = len(V)
    method_name = "Artist (merged LSCM)"
    status: Dict[str, Any] = {
        "method": method_name,
        "valid": True,
        "message": "",
        "components": 0,
        "boundary_loops": 0,
        "patches_fallback": 0,
        "time_seconds": 0.0,
    }

    try:
        if N < 3 or len(F) == 0:
            raise ValueError("mesh has too few vertices or faces")

        # 1) drop degenerate triangles
        e1 = V[F[:, 1]] - V[F[:, 0]]
        e2 = V[F[:, 2]] - V[F[:, 0]]
        areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        keep = areas > 1e-12
        Fc = F[keep]
        semantic_face_groups = None
        if face_groups is not None:
            face_groups = np.asarray(face_groups).reshape(-1)
            if len(face_groups) != len(F):
                raise ValueError("face_groups must have one label per input face")
            semantic_face_groups = face_groups[keep]
        if len(Fc) == 0:
            raise ValueError("no non-degenerate faces")

        L, K = _cotan_matrices(V, Fc)
        n_comp, labels = _vertex_components(N, Fc)

        # face -> component (vertex-component label of first vertex)
        face_comp = labels[Fc[:, 0]]
        comp_ids = sorted(int(c) for c in np.unique(face_comp))
        comp_index = {c: i for i, c in enumerate(comp_ids)}
        n_parts = len(comp_ids)

        # per-part vertex sets / faces / area / centroid
        part_verts: list = []
        part_faces: list = []   # indices into Fc
        part_area: list = []
        part_centroid: list = []
        part_group: list = []
        for ci, c in enumerate(comp_ids):
            gidx = np.flatnonzero(labels == c)
            part_verts.append(gidx)
            mask = face_comp == c
            part_faces.append(np.flatnonzero(mask))
            fa = areas[keep][mask]
            part_area.append(float(fa.sum()))
            part_centroid.append(V[gidx].mean(axis=0) if len(gidx) else np.zeros(3))
            if semantic_face_groups is None:
                part_group.append(None)
            else:
                values, counts = np.unique(semantic_face_groups[mask], return_counts=True)
                part_group.append(int(values[np.argmax(counts)]))

        total_area = float(sum(part_area)) or 1.0
        diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0))) or 1.0
        merge_dist = 0.04 * diag
        small_threshold = 0.005 * total_area   # 0.5% of surface area

        # 2) union-find: merge small parts into nearest (larger) neighbour
        parent = list(range(n_parts))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # process parts from smallest to largest
        order = sorted(range(n_parts), key=lambda i: part_area[i])
        for i in order:
            if part_area[i] > small_threshold:
                continue
            # find nearest part with larger-or-equal area within merge_dist
            best, bestd = None, merge_dist
            ci = part_centroid[i]
            for j in range(n_parts):
                if j == i or part_area[j] < part_area[i]:
                    continue
                if part_group[i] != part_group[j]:
                    continue
                d = float(np.linalg.norm(part_centroid[j] - ci))
                if d < bestd:
                    bestd = d
                    best = j
            if best is not None:
                union(i, best)

        # also merge any two parts (regardless of size) that are very close
        for i in range(n_parts):
            for j in range(i + 1, n_parts):
                if part_group[i] != part_group[j]:
                    continue
                d = float(np.linalg.norm(part_centroid[i] - part_centroid[j]))
                if d <= 0.015 * diag:
                    union(i, j)

        # group parts by root
        groups: dict = {}
        for i in range(n_parts):
            groups.setdefault(find(i), []).append(i)

        uv = np.zeros((N, 2))
        patch_meta: list = []
        semantic_patch_meta: dict[int, list] = {}
        semantic_group_area: dict[int, float] = {}
        n_fallback = 0
        n_loops_total = 0

        for g_parts in groups.values():
            # global vertex set for this group
            gidx_set = set()
            for pi in g_parts:
                gidx_set.update(int(v) for v in part_verts[pi])
            gidx = np.array(sorted(gidx_set), dtype=np.int64)
            local_of = np.full(N, -1, dtype=np.int64)
            local_of[gidx] = np.arange(len(gidx))

            # original faces of group (into Fc)
            orig_face_idx = np.concatenate([part_faces[pi] for pi in g_parts]) if g_parts else np.array([], dtype=np.int64)
            Fg_orig = Fc[orig_face_idx]

            # augmented faces: original + bridge triangles
            bridge_tris = []
            # for each pair of parts in group, add a bridge triangle
            for a in range(len(g_parts)):
                for b in range(a + 1, len(g_parts)):
                    pa, pb = g_parts[a], g_parts[b]
                    va = part_verts[pa]
                    vb = part_verts[pb]
                    if len(va) == 0 or len(vb) == 0:
                        continue
                    # closest vertex pair
                    # brute force (parts usually small after merge; but guard)
                    VA = V[va]; VB = V[vb]
                    # compute distances via broadcasting in chunks
                    best_d = 1e18; best_i = 0; best_j = 0
                    # chunk to avoid huge memory
                    for chunk_start in range(0, len(va), 256):
                        chunk = va[chunk_start:chunk_start+256]
                        D = np.linalg.norm(V[chunk][:, None, :] - V[vb][None, :, :], axis=2)
                        flat = int(np.argmin(D))
                        li, lj = divmod(flat, len(vb))
                        d = D[li, lj]
                        if d < best_d:
                            best_d = d
                            best_i = chunk[li]
                            best_j = vb[lj]
                    if best_d > merge_dist * 2:
                        continue
                    # second vertex in pa near best_i to form triangle
                    # pick any other vertex in pa
                    others = [v for v in va if v != best_i]
                    if not others:
                        continue
                    v_third = others[0]
                    bridge_tris.append((int(best_i), int(best_j), int(v_third)))

            if bridge_tris:
                bridge_arr = np.array(bridge_tris, dtype=np.int64)
                Faug_global = np.vstack([Fg_orig, bridge_arr])
            else:
                Faug_global = Fg_orig

            Fg_local = local_of[Faug_global]
            # drop any face with -1 (shouldn't happen)
            valid = (Fg_local >= 0).all(axis=1)
            Fg_local = Fg_local[valid]

            Vg = V[gidx]
            Lg, Kg = _cotan_matrices(Vg, Fg_local) if len(Fg_local) else (L[gidx][:, gidx], K[gidx][:, gidx])

            # boundary loops on augmented faces
            _, loops_g = _boundary_loops(Fg_local)
            n_loops_total += len(loops_g)
            loops_g_local = loops_g  # already local vertex ids (Fg_local uses local)

            ok = True
            try:
                if len(gidx) == 3:
                    uv_g = _triangle_uv(Vg)
                elif loops_g_local:
                    if True:
                        uv_g, ok = _lscm_solve_component(Vg, Lg, Kg, loops_g_local)
                    else:
                        uv_g, ok = _harmonic_solve_component(Vg, Lg, Kg, loops_g_local)
                else:
                    uv_g = _latlong_uv(Vg)
            except Exception:
                uv_g, ok = _latlong_uv(Vg), False

            if not np.all(np.isfinite(uv_g)):
                uv_g, ok = _latlong_uv(Vg), False
            if not ok:
                n_fallback += 1

            # area of group (original faces only)
            g_area = sum(part_area[pi] for pi in g_parts)
            f_area = (g_area / total_area) if total_area > 1e-12 else 0.0

            uv_box, w_i, h_i = _prepare_island(uv_g, f_area)
            meta = (gidx, uv_box, w_i, h_i)
            patch_meta.append(meta)
            if semantic_face_groups is not None:
                semantic_id = int(part_group[g_parts[0]])
                semantic_patch_meta.setdefault(semantic_id, []).append(meta)
                semantic_group_area[semantic_id] = semantic_group_area.get(semantic_id, 0.0) + g_area

        if semantic_face_groups is None:
            uv, island_of, pack_stats = _finalize_atlas(patch_meta, N)
        else:
            # Pack the small charts inside each semantic part first, then pack
            # those part atlases together. This keeps face/ear charts together,
            # hair charts together, etc., instead of scattering every chart
            # across the full texture square.
            semantic_layouts = []
            unclassified_chart_base = (
                int(unclassified_group_id)
                if unclassified_group_id is not None
                else (max(semantic_patch_meta) + 1 if semantic_patch_meta else 0)
            )
            for semantic_id, metas in sorted(semantic_patch_meta.items()):
                local_uv, local_chart_ids, local_stats = _finalize_atlas(metas, N)
                vertices_in_group = np.unique(np.concatenate([m[0] for m in metas]))
                points = local_uv[vertices_in_group]
                lo = points.min(axis=0)
                span = points.max(axis=0) - lo
                span = np.maximum(span, 1e-12)
                area_share = semantic_group_area[semantic_id] / total_area
                aspect = float(np.clip(span[0] / span[1], 0.2, 5.0))
                width = float(np.sqrt(max(area_share, 1e-12) * aspect))
                height = float(np.sqrt(max(area_share, 1e-12) / aspect))
                semantic_layouts.append((semantic_id, vertices_in_group, local_uv, local_chart_ids, lo, span, width, height, local_stats))

            group_boxes = [(item[6], item[7]) for item in semantic_layouts]
            group_padding = [0.012 * min(w, h) for w, h in group_boxes]
            group_outer = [
                (w + 2.0 * group_padding[i], h + 2.0 * group_padding[i])
                for i, (w, h) in enumerate(group_boxes)
            ]
            base_width = max(1e-6, np.sqrt(sum(w * h for w, h in group_boxes)))
            best_layout = None
            for factor in (0.75, 0.9, 1.05, 1.25, 1.55):
                placements, extent = _skyline_pack(group_outer, target_w=base_width * factor)
                score = max(extent)
                if best_layout is None or score < best_layout[0]:
                    best_layout = (score, placements, extent)
            _, placements, extent = best_layout
            atlas_scale = (1.0 - 2.0 * 0.008) / max(max(extent), 1e-12)
            island_of = np.full(N, -1, dtype=np.int32)
            for i, item in enumerate(semantic_layouts):
                semantic_id, group_vertices, local_uv, local_chart_ids, lo, span, width, height, _ = item
                px, py = placements[i]
                px += group_padding[i]
                py += group_padding[i]
                packed_origin = np.array([px, py]) * atlas_scale + 0.008
                local = (local_uv[group_vertices] - lo) / span
                uv[group_vertices] = local * np.array([width, height]) * atlas_scale + packed_origin
                if semantic_id == unclassified_group_id:
                    island_of[group_vertices] = unclassified_chart_base + local_chart_ids[group_vertices]
                else:
                    island_of[group_vertices] = semantic_id
            top_level_efficiency = sum(w * h for w, h in group_boxes) / max(extent[0] * extent[1], 1e-12)
            inner_efficiency = float(np.mean([item[8].get("packing_efficiency", 0.0) for item in semantic_layouts]))
            pack_stats = {
                "islands": int(len(np.unique(island_of[island_of >= 0]))),
                "packing_efficiency": round(float(min(top_level_efficiency * inner_efficiency, 1.0)), 3),
            }
            status["uv_charts"] = len(patch_meta)

        status.update({
            "components": int(len(groups)),
            "semantic_part_groups": int(len(set(part_group))) if semantic_face_groups is not None else None,
            "boundary_loops": int(n_loops_total),
            "patches_fallback": int(n_fallback),
            "islands": pack_stats.get("islands", 0),
            "packing_efficiency": pack_stats.get("packing_efficiency", 0.0),
            "message": "UV computation completed successfully",
            "time_seconds": round(time.time() - t0, 2),
        })
        return uv, status, island_of

    except Exception as e:
        status["valid"] = False
        status["message"] = f"{method_name} failed ({e}); used box projection fallback"
        return _cube_projection(V), status, np.zeros(len(V), dtype=np.int32)


def _xatlas_unwrap_semantic_groups(
    V: np.ndarray,
    F: np.ndarray,
    face_groups: np.ndarray,
    unclassified_group_id: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Unwrap each semantic part independently with xatlas, then pack parts.

    The former semantic path stitched distant shells together with synthetic
    triangles before LSCM. Closed pieces have no boundary loops, so those
    artificial bridges could collapse to enormous triangular charts. xatlas
    cuts each part into valid surface charts without adding geometry, while a
    second packing step reserves atlas space according to each part's surface
    area.
    """
    import xatlas

    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    face_groups = np.asarray(face_groups, dtype=np.int32).reshape(-1)
    if len(face_groups) != len(F):
        raise ValueError("face_groups must have one label per input face")

    groups = [int(group_id) for group_id in np.unique(face_groups)]
    points = V[F]
    face_areas = 0.5 * np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
    )
    area_total = max(float(face_areas.sum()), 1e-12)
    layouts: list[dict[str, Any]] = []
    output_vertices: list[np.ndarray] = []
    output_faces: list[np.ndarray] = []
    output_uv: list[np.ndarray] = []
    output_islands: list[np.ndarray] = []
    output_semantic_ids: list[np.ndarray] = []
    vertex_offset = 0
    next_unclassified_chart = int(unclassified_group_id or 0)
    chart_count = 0

    for group_id in groups:
        face_ids = np.flatnonzero(face_groups == group_id)
        group_faces = F[face_ids]
        source_vertices, inverse = np.unique(group_faces.reshape(-1), return_inverse=True)
        local_faces = inverse.reshape(-1, 3).astype(np.uint32, copy=False)
        atlas = xatlas.Atlas()
        atlas.add_mesh(
            np.ascontiguousarray(V[source_vertices], dtype=np.float32),
            np.ascontiguousarray(local_faces, dtype=np.uint32),
        )
        chart_options = xatlas.ChartOptions()
        # The upstream defaults stop chart growth early and strongly penalize
        # crossing normal changes. On faceted, triangulated game meshes this
        # creates thousands of postage-stamp charts. Permit broader chart
        # growth within each semantic part while still keeping xatlas's
        # distortion and packing safeguards.
        chart_options.max_cost = 4.0
        chart_options.max_iterations = 4
        chart_options.normal_seam_weight = 0.5
        pack_options = xatlas.PackOptions()
        pack_options.resolution = 1024
        pack_options.padding = 2
        atlas.generate(chart_options, pack_options)
        vmapping, atlas_faces, atlas_uv = atlas.get_mesh(0)
        atlas_faces = np.asarray(atlas_faces, dtype=np.int64)
        group_vertices = np.asarray(V[source_vertices[np.asarray(vmapping, dtype=np.int64)]], dtype=np.float64)
        group_uv = np.asarray(atlas_uv, dtype=np.float64)
        _, chart_ids = _vertex_components(len(group_vertices), atlas_faces)
        chart_ids = chart_ids.astype(np.int32)
        if group_id == unclassified_group_id:
            group_islands = chart_ids + next_unclassified_chart
            next_unclassified_chart += int(chart_ids.max()) + 1 if len(chart_ids) else 0
            group_semantic_ids = np.full(len(group_vertices), group_id, dtype=np.int32)
        else:
            # Islands in a named part share one semantic selection/color,
            # even though xatlas may make several UV charts inside the part.
            group_islands = np.full(len(group_vertices), group_id, dtype=np.int32)
            group_semantic_ids = np.full(len(group_vertices), group_id, dtype=np.int32)

        atlas_faces += vertex_offset
        tri = group_vertices[atlas_faces - vertex_offset]
        group_area = float(0.5 * np.linalg.norm(
            np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1
        ).sum())
        lo = group_uv.min(axis=0)
        span = np.maximum(group_uv.max(axis=0) - lo, 1e-12)
        aspect = float(np.clip(span[0] / span[1], 0.2, 5.0))
        area_share = group_area / area_total
        width = float(np.sqrt(max(area_share, 1e-12) * aspect))
        height = float(np.sqrt(max(area_share, 1e-12) / aspect))
        layouts.append({
            "group_id": group_id,
            "vertex_start": vertex_offset,
            "vertex_end": vertex_offset + len(group_vertices),
            "uv": group_uv,
            "lo": lo,
            "span": span,
            "width": width,
            "height": height,
            "area": group_area,
        })
        output_vertices.append(group_vertices)
        output_faces.append(atlas_faces)
        output_uv.append(group_uv.copy())
        output_islands.append(group_islands)
        output_semantic_ids.append(group_semantic_ids)
        vertex_offset += len(group_vertices)
        chart_count += int(atlas.get_mesh_chart_count(0))

    if not output_vertices:
        raise ValueError("no semantic groups to unwrap")

    group_sizes = [(entry["width"], entry["height"]) for entry in layouts]
    padding = [0.014 * min(width, height) for width, height in group_sizes]
    outer_sizes = [
        (width + 2.0 * padding[i], height + 2.0 * padding[i])
        for i, (width, height) in enumerate(group_sizes)
    ]
    base_width = max(1e-6, np.sqrt(sum(width * height for width, height in group_sizes)))
    best_layout = None
    for factor in (0.75, 0.9, 1.05, 1.25, 1.55):
        placements, extent = _skyline_pack(outer_sizes, target_w=base_width * factor)
        score = max(extent)
        if best_layout is None or score < best_layout[0]:
            best_layout = (score, placements, extent)
    _, placements, extent = best_layout
    atlas_scale = (1.0 - 2.0 * 0.008) / max(max(extent), 1e-12)

    packed_uv: list[np.ndarray] = []
    for i, entry in enumerate(layouts):
        px, py = placements[i]
        px += padding[i]
        py += padding[i]
        origin = np.array([px, py]) * atlas_scale + 0.008
        local = (entry["uv"] - entry["lo"]) / entry["span"]
        packed_uv.append(
            local * np.array([entry["width"], entry["height"]]) * atlas_scale + origin
        )

    V_out = np.vstack(output_vertices)
    F_out = np.vstack(output_faces).astype(np.int32, copy=False)
    UV_out = np.vstack(packed_uv).astype(np.float64, copy=False)
    islands = np.concatenate(output_islands).astype(np.int32, copy=False)
    semantic_ids = np.concatenate(output_semantic_ids).astype(np.int32, copy=False)
    occupied = sum(width * height for width, height in group_sizes)
    efficiency = occupied / max(extent[0] * extent[1], 1e-12)
    status = {
        "method": "Grounding DINO parts + SAM2 masks + semantic xatlas charts",
        "valid": True,
        "xatlas_available": True,
        "uv_charts": int(chart_count),
        "islands": int(len(np.unique(islands))),
        "semantic_part_groups": int(len(groups)),
        "packing_efficiency": round(float(min(efficiency, 1.0)), 3),
        "message": "Each semantic part was charted with xatlas and packed by surface area.",
    }
    return V_out, F_out, UV_out, islands, semantic_ids, status


def compute_artist_uv(
    V: np.ndarray, F: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    """Artist-style unwrap: merge small parts and stitch into coherent patches."""
    return _artist_unwrap_components(V, F)


def compute_sam2_uv(
    V: np.ndarray,
    F: np.ndarray,
    part_detection_thresh: float = 0.22,
    pred_iou_thresh: float = 0.50,
    stability_score_thresh: float = 0.75,
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Create SAM2-informed seams, split the mesh, then apply LSCM.

    Splitting is essential: one shared vertex cannot hold two UV positions on
    opposite sides of a seam.  Therefore this method returns the split mesh in
    addition to its UVs.
    """
    component_count, _ = _vertex_components(len(V), F)
    semantic_error = None
    try:
        regions, sam_status = _semantic_part_regions(
            V,
            F,
            detection_threshold=part_detection_thresh,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )
    except Exception as exc:
        semantic_error = str(exc)
        regions = None
        sam_status = {"sam2_available": False, "sam2_mode": "semantic detector unavailable"}

    if regions is None:
        # Do not pretend proximity is semantic classification. This fallback
        # is kept only when the text detector/checkpoints are unavailable.
        fragmented_threshold = max(64, len(F) // 200)
        if component_count >= fragmented_threshold:
            UV, status, islands = _artist_unwrap_components(V, F)
            status.update({
                "method": "Topology-guided part grouping + LSCM (fallback)",
                "sam2_mode": "topology fallback; semantic detector unavailable",
                "sam2_used": False,
                "semantic_detector_note": semantic_error,
                "input_components": int(component_count),
                "fragmented_mesh_threshold": int(fragmented_threshold),
                "message": (
                    "Semantic detection was unavailable; loose shells were grouped geometrically. "
                    "This fallback does not classify anatomy or materials."
                ),
            })
            return UV, status, islands, V, F
        regions, fallback_status = discover_sam2_regions(
            V, F, pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )
        sam_status.update(fallback_status)
        sam_status["semantic_detector_note"] = semantic_error
    V_cut, F_cut, duplicated = split_on_region_seams(V, F, regions)
    semantic_islands = None
    semantic_vertex_ids = None
    if sam_status.get("semantic_model"):
        # Each semantic class gets its own xatlas charting pass. This prevents
        # the closed-shell LSCM fallback from collapsing closed armor/clothing
        # pieces into the large triangular charts visible in the old atlas.
        region_names = sam_status.get("semantic_region_names", {})
        unclassified_group_id = next(
            (int(key) for key, name in region_names.items() if name == "unclassified"),
            None,
        )
        try:
            V_out, F_out, UV, islands, semantic_vertex_ids, status = _xatlas_unwrap_semantic_groups(
                V_cut, F_cut, face_groups=regions,
            unclassified_group_id=unclassified_group_id,
        )
            V_cut, F_cut = V_out, F_out
        except Exception as exc:
            UV, status, islands = _artist_unwrap_components(
                V_cut, F_cut, face_groups=regions,
                unclassified_group_id=unclassified_group_id,
            )
            semantic_vertex_ids = np.full(len(V_cut), int(unclassified_group_id or 0), dtype=np.int32)
            for face, group_id in zip(F_cut, regions):
                semantic_vertex_ids[face] = int(group_id)
            status["xatlas_available"] = False
            status["xatlas_note"] = str(exc)
        status.update({
            "method": "Grounding DINO parts + SAM2 masks + semantic xatlas charts",
            "message": "Semantic boundaries were cut; xatlas charted each part and packed the charts inside per-part atlas groups.",
            "uv_charts": int(status.get("uv_charts", status.get("islands", 0))),
        })
        # Color and select mesh areas by semantic class. A semantic group may
        # contain several disconnected UV charts, but they remain one editable
        # piece (e.g. face plus ears). Unclassified charts stay individually
        # selectable so detector misses cannot merge the whole body together.
        status["semantic_part_groups"] = int(len(np.unique(regions)))
        if unclassified_group_id is not None:
            names = {
                str(key): name
                for key, name in region_names.items()
                if name != "unclassified"
            }
            for group_id in np.unique(islands):
                if int(group_id) >= unclassified_group_id:
                    names[str(int(group_id))] = "unclassified"
            sam_status["semantic_region_names"] = names
    else:
        try:
            # For the non-semantic fallback, xatlas gives robust automatic charts.
            import xatlas
            vmapping, F_atlas, UV = xatlas.parametrize(
                np.asarray(V_cut, dtype=np.float32),
                np.asarray(F_cut, dtype=np.uint32),
            )
            V_out = np.asarray(V_cut[vmapping], dtype=np.float64)
            F_out = np.asarray(F_atlas, dtype=np.int32)
            _, islands = _vertex_components(len(V_out), F_out)
            islands = islands.astype(np.int32)
            status = {
                "method": "SAM2 multiview regions + xatlas charting",
                "valid": True,
                "message": "Mesh regions segmented by multiview edge consensus; xatlas generated and packed UV charts",
                "islands": int(len(np.unique(islands))),
                "xatlas_available": True,
            }
            V_cut, F_cut = V_out, F_out
        except Exception as exc:
            UV, status, islands = _unwrap_components(V_cut, F_cut, "LSCM")
            status["xatlas_available"] = False
            status["xatlas_note"] = str(exc)
    status.update(sam_status)
    if semantic_vertex_ids is not None:
        # Keep semantic color identity separate from UV chart IDs. Unclassified
        # charts remain independently selectable, but use one consistent color
        # that matches the legend instead of looking like false semantic parts.
        status["semantic_vertex_ids"] = semantic_vertex_ids.tolist()
    status.update({
        "method": status.get("method", "SAM2 Smart Seams + LSCM"),
        "semantic_regions": int(len(np.unique(regions))),
        "seam_vertex_duplicates": int(duplicated),
    })
    return UV, status, islands, V_cut, F_cut


def compute_lscm_uv(
    V: np.ndarray, F: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    """
    Compute LSCM (least-squares conformal) UVs, one patch per connected
    component, packed into a non-overlapping atlas.

    Returns:
        UV: Nx2 array of UV coordinates
        status: Status dict with info
        islands: N-length array with the island index each vertex belongs to
                 (-1 for vertices that are not part of any island)
    """
    return _unwrap_components(V, F, "LSCM")


def compute_harmonic_uv(
    V: np.ndarray, F: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    """
    Compute harmonic UV coordinates.

    The boundary of each connected component is laid out by LSCM, then the
    interior is relaxed with a Laplace solve (harmonic map).

    Args:
        V: Nx3 array of vertex positions
        F: Mx3 array of face indices

    Returns:
        UV: Nx2 array of UV coordinates
        status: Status dict with info
        islands: N-length array with the island index of each vertex
    """
    return _unwrap_components(V, F, "HARMONIC")


def normalize_uv_coordinates(UV: np.ndarray) -> np.ndarray:
    """Normalize UV coordinates to [0, 1] range."""
    min_vals = np.min(UV, axis=0)
    max_vals = np.max(UV, axis=0)
    span = max_vals - min_vals

    if span.max() > 0:
        UV_normalized = (UV - min_vals) / span
    else:
        UV_normalized = np.zeros_like(UV)

    return UV_normalized


def compute_automatic_uv(
    V: np.ndarray, F: np.ndarray, method: str = "SAM2"
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    """
    Automatically compute UV coordinates for a mesh using the specified method.

    Args:
        V: Nx3 array of vertex positions
        F: Mx3 array of face indices
        method: Unwrapping method ("SAM2", "LSCM", "HARMONIC" or "ARTIST")

    Returns:
        UV: Nx2 array of UV coordinates
        status: Status dict with method info
        islands: N-length array with the island index of each vertex
    """
    m = method.upper()
    if m == "SAM2":
        UV, status, islands, _, _ = compute_sam2_uv(V, F)
    elif m == "LSCM":
        UV, status, islands = compute_lscm_uv(V, F)
    elif m == "HARMONIC":
        UV, status, islands = compute_harmonic_uv(V, F)
    elif m in ("ARTIST", "SMART"):
        UV, status, islands = compute_artist_uv(V, F)
    else:
        raise ValueError(f"Unsupported UV method: {method}")

    return UV, status, islands


def build_unwrapped_mesh(
    V: np.ndarray, F: np.ndarray, UV: np.ndarray,
    status: Optional[Dict[str, Any]] = None,
    islands: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Build unwrapped mesh data structure."""
    st = status or {}
    return {
        "vertices": V.tolist(),
        "faces": F.tolist(),
        "uv": UV.tolist(),
        "islands": None if islands is None else np.asarray(islands).astype(int).tolist(),
        "status": {
            "uv_method": st.get("method", "LSCM (Pure Python)"),
            "uv_valid": bool(st.get("valid", True)),
        }
    }
