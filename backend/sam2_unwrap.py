"""SAM2-guided seam discovery for SmartUnwrapper.

SAM2 is an image segmenter, rather than a UV parameterizer.  This adapter
renders face ids from several orthographic views, asks SAM2 for masks, maps
the masks back to mesh faces, and uses their borders as *semantic seam*
candidates.  The actual flattening remains the robust LSCM implementation in
``unwrap.py``.

The checkpoint and minimal vendored SAM2 inference package live together in
``third_party/SAM2``. Keeping imports lazy makes the web server usable on
CPU-only installations without the optional SAM2 dependencies as well.
"""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any
import sys

import numpy as np


_SAM_ROOT = Path(__file__).with_name("third_party") / "SAM2"
_CHECKPOINT = _SAM_ROOT / "sam2.1_hiera_base_plus.pt"
_GROUNDING_DINO_ROOT = Path(__file__).with_name("third_party") / "GroundingDINO"

_SEMANTIC_PARTS = (
    "hair", "face", "scarf", "torso", "arm", "hand", "leg", "shoe",
    "belt", "pouch",
)


@lru_cache(maxsize=1)
def _load_semantic_models(device: str):
    """Load Apache-2.0 Grounding DINO and SAM2 once per process/device."""
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    required_model_files = (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.txt",
    )
    missing = [name for name in required_model_files if not (_GROUNDING_DINO_ROOT / name).is_file()]
    if missing:
        raise RuntimeError(
            "Bundled Grounding DINO files are missing from "
            f"{_GROUNDING_DINO_ROOT}: {', '.join(missing)}"
        )
    model_reference = str(_GROUNDING_DINO_ROOT)
    detector_processor = AutoProcessor.from_pretrained(model_reference, local_files_only=True)
    detector = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_reference, local_files_only=True
    )
    detector.to(device).eval()
    sam_model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_b+.yaml", str(_CHECKPOINT), device=device
    )
    source = "repository third_party/GroundingDINO"
    return torch, detector_processor, detector, SAM2ImagePredictor(sam_model), source


def _semantic_part_regions(
    vertices: np.ndarray,
    faces: np.ndarray,
    detection_threshold: float,
    pred_iou_thresh: float,
    stability_score_thresh: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Detect named parts in multi-view renders, refine with SAM2, lift to faces.

    Grounding DINO provides language-conditioned boxes; SAM2 turns each box
    into a mask. Mesh-edge geodesics then propagate visible labels onto hidden
    faces without allowing nearby but disconnected shells to merge.
    """
    local_source = str(_SAM_ROOT)
    if local_source not in sys.path:
        sys.path.insert(0, local_source)
    try:
        import torch
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components, dijkstra
        from scipy.spatial import cKDTree
        from PIL import Image, ImageDraw
        torch_module, processor, detector, predictor, detector_source = _load_semantic_models(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    except ImportError as exc:
        raise RuntimeError(
            "Semantic part detection needs transformers, Grounding DINO, and SAM2"
        ) from exc

    n_faces = len(faces)
    p = vertices[faces]
    normals = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    centers = p.mean(axis=1)
    scores = np.zeros((n_faces, len(_SEMANTIC_PARTS)), dtype=np.float32)
    part_index = {name: i for i, name in enumerate(_SEMANTIC_PARTS)}
    # Keep each prompted phrase tied to its own token span. Transformers'
    # Grounding DINO post-processor joins every token over text_threshold into
    # one string (for example "hair scarf torso shoe"). Treating that string
    # as one class made the first matching alias win, which mislabeled shoes,
    # the face, and other parts. Score phrase spans from model logits instead.
    # Grounding DINO responds much better to visible garment names than to
    # anatomy alone on stylized/game characters. Keep synonyms inside the
    # same semantic class, then take their strongest token score below.
    prompt_aliases = [
        ("hair",),
        ("human face", "face"),
        ("scarf",),
        ("torso", "chest armor", "body armor"),
        ("arm", "shoulder armor", "shoulder pad", "forearm armor"),
        ("hand", "glove"),
        ("leg", "pants", "thigh armor", "shin guard"),
        ("shoe", "boot"),
        ("belt",),
        ("pouch", "utility pouch", "bag"),
    ]
    prompts = [phrase for aliases in prompt_aliases for phrase in aliases]
    prompt_text = " . ".join(prompts) + "."
    token_offsets = processor.tokenizer(
        prompt_text, return_offsets_mapping=True, add_special_tokens=True
    )["offset_mapping"]
    if token_offsets and isinstance(token_offsets[0], list):
        token_offsets = token_offsets[0]
    phrase_token_ids: list[list[int]] = []
    char_cursor = 0
    for phrase in prompts:
        phrase_start = prompt_text.index(phrase, char_cursor)
        phrase_end = phrase_start + len(phrase)
        phrase_token_ids.append([
            token_id
            for token_id, (start, end) in enumerate(token_offsets)
            if start < phrase_end and end > phrase_start
        ])
        char_cursor = phrase_end
    if any(not ids for ids in phrase_token_ids):
        raise RuntimeError("Grounding DINO tokenizer did not preserve semantic prompt spans")
    class_phrase_indices: list[list[int]] = []
    next_phrase = 0
    for aliases in prompt_aliases:
        class_phrase_indices.append(list(range(next_phrase, next_phrase + len(aliases))))
        next_phrase += len(aliases)
    detection_counts: dict[str, int] = defaultdict(int)
    mask_counts: dict[str, int] = defaultdict(int)
    rejected_mask_counts: dict[str, int] = defaultdict(int)
    size = 768
    light = np.array([0.35, 0.45, 0.82], dtype=np.float64)
    light /= np.linalg.norm(light)

    # Use upright front/back and left/right views. The previous XY projection
    # was upside down and the remaining projections were top-down/sideways,
    # making DINO confuse shoes, hair, and clothing.
    for u, v, depth_axis in ((0, 1, 2), (2, 1, 0)):
        lo, hi = vertices[:, [u, v]].min(axis=0), vertices[:, [u, v]].max(axis=0)
        xy = (vertices[:, [u, v]] - lo) * ((size - 2) / np.maximum(hi - lo, 1e-12)) + 1
        xy[:, 1] = (size - 1) - xy[:, 1]
        for reverse in (False, True):
            rgb = Image.new("RGB", (size, size), (20, 20, 20))
            ids = Image.new("I", (size, size), 0)
            color_draw, id_draw = ImageDraw.Draw(rgb), ImageDraw.Draw(ids)
            order = np.argsort(centers[:, depth_axis])
            if reverse:
                order = order[::-1]
            for fi in order:
                polygon = [tuple(point) for point in xy[faces[fi]]]
                shade = 55 + int(175 * abs(float(normals[fi] @ light)))
                color_draw.polygon(polygon, fill=(shade, shade, shade))
                id_draw.polygon(polygon, fill=int(fi + 1))
            image = np.asarray(rgb)
            face_map = np.asarray(ids, dtype=np.int64) - 1
            visible = face_map >= 0
            view_face_counts = np.bincount(face_map[visible], minlength=n_faces)

            inputs = processor(images=rgb, text=[prompt_text], return_tensors="pt").to(detector.device)
            with torch_module.inference_mode():
                outputs = detector(**inputs)
            token_logits = outputs.logits[0].sigmoid()
            phrase_scores = torch_module.stack([
                token_logits[:, token_ids].amax(dim=1)
                for token_ids in phrase_token_ids
            ], dim=1)
            class_scores = torch_module.stack([
                phrase_scores[:, phrase_ids].amax(dim=1)
                for phrase_ids in class_phrase_indices
            ], dim=1)
            confidence, class_ids = class_scores.max(dim=1)
            keep = confidence >= detection_threshold
            if not bool(keep.any()):
                predictor.set_image(image)
                continue
            normalized_boxes = outputs.pred_boxes[0][keep]
            confidence = confidence[keep]
            class_ids = class_ids[keep]
            box_min = normalized_boxes[:, :2] - normalized_boxes[:, 2:] * 0.5
            box_max = normalized_boxes[:, :2] + normalized_boxes[:, 2:] * 0.5
            boxes = torch_module.cat((box_min, box_max), dim=1)
            boxes *= torch_module.tensor(
                [size, size, size, size], device=boxes.device, dtype=boxes.dtype
            )
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, size - 1)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, size - 1)
            predictor.set_image(image)
            view_scores = np.zeros_like(scores)
            for box, confidence, class_id in zip(boxes, confidence, class_ids):
                part = _SEMANTIC_PARTS[int(class_id)]
                box_values = box.detach().cpu().numpy().astype(np.float32)
                box_area = max(float(box_values[2] - box_values[0]), 0.0) * max(
                    float(box_values[3] - box_values[1]), 0.0
                )
                # Full-frame person boxes are generic detections, not useful
                # part localizations; SAM2 otherwise returns most of the mesh.
                if box_area > size * size * 0.72:
                    continue
                detection_counts[part] += 1
                masks, mask_scores, mask_logits = predictor.predict(
                    box=box_values,
                    multimask_output=True,
                )
                valid_candidates = []
                for candidate, mask_score in enumerate(mask_scores):
                    logits = np.asarray(mask_logits[candidate])
                    loose = logits > -1.0
                    stable = logits > 1.0
                    stability = float(stable.sum() / max(int(loose.sum()), 1))
                    if float(mask_score) >= pred_iou_thresh and stability >= stability_score_thresh:
                        valid_candidates.append((float(mask_score), stability, candidate))
                if not valid_candidates:
                    rejected_mask_counts[part] += 1
                    continue
                # Prefer the best SAM confidence among candidates that also
                # pass stability; the single top-scoring mask may be unstable.
                _, _, candidate = max(valid_candidates)
                mask_counts[part] += 1
                mask = np.asarray(masks[candidate], dtype=bool).copy()
                # Box prompts occasionally make SAM return a neighboring
                # garment or an entire limb. Clip a small margin outside the
                # detector box so that class evidence stays spatially local.
                bx0, by0, bx1, by1 = box_values
                pad_x = max((bx1 - bx0) * 0.12, 3.0)
                pad_y = max((by1 - by0) * 0.12, 3.0)
                x0 = max(0, int(np.floor(bx0 - pad_x)))
                y0 = max(0, int(np.floor(by0 - pad_y)))
                x1 = min(size, int(np.ceil(bx1 + pad_x)))
                y1 = min(size, int(np.ceil(by1 + pad_y)))
                box_clip = np.zeros_like(mask)
                box_clip[y0:y1, x0:x1] = True
                mask &= box_clip
                if not mask.any():
                    continue
                hit = face_map[mask]
                hit = hit[hit >= 0]
                if not len(hit):
                    continue
                per_face = np.bincount(hit, minlength=n_faces).astype(np.float32)
                per_face /= np.maximum(view_face_counts, 1.0).astype(np.float32)
                evidence = float(confidence.detach().cpu()) * per_face
                # Several overlapping boxes for the same class in one view are
                # duplicate evidence, not independent votes. Keep the strongest
                # mask per face/class/view, then combine evidence across views.
                view_scores[:, part_index[part]] = np.maximum(
                    view_scores[:, part_index[part]], evidence
                )
            scores += view_scores

    if not np.any(scores):
        raise RuntimeError("Grounding DINO did not detect any supported body parts")

    # Seed only confident visible detections. Dijkstra on the face-dual graph
    # expands each class over the surface, constrained by broad anatomical
    # plausibility ranges. These ranges prevent impossible propagation; they
    # never create labels or overwrite detector/SAM evidence.
    seed_labels = np.argmax(scores, axis=1).astype(np.int32)
    seed_strength = scores.max(axis=1)
    seed_labels[seed_strength < 0.12] = -1
    y_min, y_max = float(vertices[:, 1].min()), float(vertices[:, 1].max())
    x_min, x_max = float(vertices[:, 0].min()), float(vertices[:, 0].max())
    height = (centers[:, 1] - y_min) / max(y_max - y_min, 1e-9)
    lateral = np.abs(centers[:, 0] - 0.5 * (x_min + x_max)) / max(0.5 * (x_max - x_min), 1e-9)
    # (minimum height, maximum height, minimum lateral distance, maximum
    # lateral distance). The asymmetric arm/hand bounds keep those labels from
    # geodesically flooding into the chest; these only constrain propagation.
    part_bands = {
        "hair": (0.89, 1.02, 0.00, 0.58), "face": (0.78, 0.98, 0.00, 0.46),
        "scarf": (0.69, 0.91, 0.00, 0.54), "torso": (0.42, 0.84, 0.00, 0.72),
        "arm": (0.49, 0.89, 0.28, 1.05), "hand": (0.39, 0.79, 0.70, 1.05),
        "belt": (0.52, 0.69, 0.00, 0.66), "pouch": (0.25, 0.70, 0.16, 1.05),
        "leg": (0.06, 0.53, 0.00, 0.90), "shoe": (0.00, 0.16, 0.00, 0.88),
    }
    allowed_by_part = {}
    for class_id, part in enumerate(_SEMANTIC_PARTS):
        low, high, min_lateral, max_lateral = part_bands[part]
        allowed = (
            (height >= low) & (height <= high)
            & (lateral >= min_lateral) & (lateral <= max_lateral)
        )
        allowed_by_part[class_id] = allowed
        seed_labels[(seed_labels == class_id) & ~allowed] = -1
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        for x, y in ((a, b), (b, c), (c, a)):
            edge_faces[(min(int(x), int(y)), max(int(x), int(y)))].append(fi)
    rows, cols, costs = [], [], []
    for incident in edge_faces.values():
        if len(incident) != 2:
            continue
        a, b = incident
        length = float(np.linalg.norm(centers[a] - centers[b]))
        normal_cost = 1.0 + 3.0 * (1.0 - abs(float(normals[a] @ normals[b])))
        weight = max(length, 1e-8) * normal_cost
        rows.extend((a, b)); cols.extend((b, a)); costs.extend((weight, weight))
    graph = coo_matrix((costs, (rows, cols)), shape=(n_faces, n_faces)).tocsr()
    labels = np.full(n_faces, -1, dtype=np.int32)
    best_distance = np.full(n_faces, np.inf, dtype=np.float64)
    for class_id, part in enumerate(_SEMANTIC_PARTS):
        allowed = allowed_by_part[class_id]
        seeds = np.flatnonzero((seed_labels == class_id) & allowed)
        if not len(seeds):
            continue
        active_faces = np.flatnonzero(allowed)
        active_graph = graph[active_faces][:, active_faces]
        local_seeds = np.searchsorted(active_faces, seeds)
        local_distances = dijkstra(
            active_graph, directed=False, indices=local_seeds, min_only=True
        )
        distances = np.full(n_faces, np.inf, dtype=np.float64)
        distances[active_faces] = local_distances
        better = distances < best_distance
        labels[better] = class_id
        best_distance[better] = distances[better]

    # Symmetric hard-surface pieces such as shoulder pads are often missed by
    # the text detector on grayscale mesh renders. If two unlabelled shells
    # form a close bilateral pair in the upper-arm band, classify the pair as
    # arm armor together instead of leaving them as unrelated color fragments.
    unknown_faces = np.flatnonzero(labels < 0)
    symmetric_shells_assigned = 0
    if len(unknown_faces):
        unknown_graph = graph[unknown_faces][:, unknown_faces]
        n_unknown_components, unknown_component_ids = connected_components(
            unknown_graph, directed=False
        )
        part_centers = []
        part_areas = []
        part_extents = []
        part_face_ids = []
        y_span = max(float(vertices[:, 1].max() - vertices[:, 1].min()), 1e-9)
        x_mid = 0.5 * (x_min + x_max)
        diag = max(float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))), 1e-9)
        total_area = max(float(0.5 * np.linalg.norm(
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1
        ).sum()), 1e-12)
        for component_id in range(n_unknown_components):
            face_ids = unknown_faces[unknown_component_ids == component_id]
            if len(face_ids) < max(12, int(n_faces * 0.0004)):
                continue
            component_centers = centers[face_ids]
            centroid = component_centers.mean(axis=0)
            component_height = (centroid[1] - y_min) / y_span
            component_lateral = abs(centroid[0] - x_mid) / max(0.5 * (x_max - x_min), 1e-9)
            area = float(0.5 * np.linalg.norm(
                np.cross(p[face_ids, 1] - p[face_ids, 0], p[face_ids, 2] - p[face_ids, 0]),
                axis=1,
            ).sum())
            # Restrict this correction to substantial upper-limb shells. It
            # must not relabel the head, scarf, chest, or belt accessories.
            if not (0.66 <= component_height <= 0.93 and 0.10 <= component_lateral <= 0.78):
                continue
            if area < total_area * 0.0005:
                continue
            component_points = vertices[faces[face_ids]].reshape(-1, 3)
            part_centers.append(centroid)
            part_areas.append(area)
            part_extents.append(component_points.max(axis=0) - component_points.min(axis=0))
            part_face_ids.append(face_ids)

        if len(part_centers) >= 2:
            part_centers_array = np.asarray(part_centers)
            tree = cKDTree(part_centers_array)
            paired: set[int] = set()
            max_pair_distance = 0.035 * diag
            arm_id = part_index["arm"]
            for i, centroid in enumerate(part_centers_array):
                if i in paired:
                    continue
                mirrored = centroid.copy()
                mirrored[0] = 2.0 * x_mid - centroid[0]
                candidates = tree.query_ball_point(mirrored, max_pair_distance)
                best = None
                best_cost = np.inf
                for j in candidates:
                    if j == i or j in paired:
                        continue
                    other = part_centers_array[j]
                    if (centroid[0] - x_mid) * (other[0] - x_mid) >= 0:
                        continue
                    if abs(centroid[1] - other[1]) > 0.045 * y_span:
                        continue
                    ratio = part_areas[i] / max(part_areas[j], 1e-12)
                    if not (0.20 <= ratio <= 5.0):
                        continue
                    extent_error = np.linalg.norm(part_extents[i] - part_extents[j])
                    if extent_error > 0.10 * diag:
                        continue
                    mirror_error = float(np.linalg.norm(mirrored - other))
                    cost = mirror_error + abs(np.log(ratio)) * 0.01 * diag
                    if cost < best_cost:
                        best, best_cost = j, cost
                if best is None:
                    continue
                labels[part_face_ids[i]] = arm_id
                labels[part_face_ids[best]] = arm_id
                paired.update((i, best))
                symmetric_shells_assigned += 2

    # Fill detector gaps around the upper arm using the nearby arm-labeled
    # surface as an anchor. This catches shoulder plates that are attached to
    # the torso shell rather than represented as a clean mirrored component.
    nearby_arm_faces_assigned = 0
    arm_faces = np.flatnonzero(labels == part_index["arm"])
    remaining_unknown = np.flatnonzero(labels < 0)
    if len(arm_faces) and len(remaining_unknown):
        y_span = max(float(vertices[:, 1].max() - vertices[:, 1].min()), 1e-9)
        x_mid = 0.5 * (x_min + x_max)
        arm_zone = (
            (height[remaining_unknown] >= 0.62)
            & (height[remaining_unknown] <= 0.93)
            & (np.abs(centers[remaining_unknown, 0] - x_mid) / max(0.5 * (x_max - x_min), 1e-9) >= 0.12)
            & (np.abs(centers[remaining_unknown, 0] - x_mid) / max(0.5 * (x_max - x_min), 1e-9) <= 0.78)
        )
        candidate_faces = remaining_unknown[arm_zone]
        if len(candidate_faces):
            arm_tree = cKDTree(centers[arm_faces])
            distances, _ = arm_tree.query(centers[candidate_faces], k=1)
            near_arm = distances <= 0.05 * diag
            assigned_faces = candidate_faces[near_arm]
            labels[assigned_faces] = part_index["arm"]
            nearby_arm_faces_assigned = int(len(assigned_faces))

    # A mesh assembled from many FBX objects often has small detached shells
    # that are invisible to the rendered masks. Transfer nearby semantic
    # evidence onto those faces, but only inside each class's anatomical band
    # and only when both position and surface orientation agree.
    spatially_assigned_faces = 0
    unknown_faces = np.flatnonzero(labels < 0)
    if len(unknown_faces):
        nearest_cost = np.full(len(unknown_faces), np.inf, dtype=np.float64)
        nearest_class = np.full(len(unknown_faces), -1, dtype=np.int32)
        max_transfer_distance = 0.014 * diag
        normal_penalty = 0.003 * diag
        for class_id in range(len(_SEMANTIC_PARTS)):
            allowed_unknown = allowed_by_part[class_id][unknown_faces]
            query_faces = unknown_faces[allowed_unknown]
            anchor_faces = np.flatnonzero(labels == class_id)
            if not len(query_faces) or not len(anchor_faces):
                continue
            tree = cKDTree(centers[anchor_faces])
            k = min(4, len(anchor_faces))
            distances, nearest = tree.query(centers[query_faces], k=k)
            if k == 1:
                distances = distances[:, None]
                nearest = nearest[:, None]
            anchor_normals = normals[anchor_faces[nearest]]
            normal_agreement = np.abs(np.einsum(
                "ij,ikj->ik", normals[query_faces], anchor_normals
            ))
            costs = distances + normal_penalty * (1.0 - normal_agreement)
            local_choice = np.argmin(costs, axis=1)
            local_cost = costs[np.arange(len(query_faces)), local_choice]
            improve = local_cost < nearest_cost[allowed_unknown]
            if np.any(improve):
                target_faces = query_faces[improve]
                target_cost = local_cost[improve]
                global_positions = np.flatnonzero(allowed_unknown)[improve]
                nearest_cost[global_positions] = target_cost
                nearest_class[global_positions] = class_id
        accepted = nearest_cost <= max_transfer_distance
        labels[unknown_faces[accepted]] = nearest_class[accepted]
        spatially_assigned_faces = int(accepted.sum())

    unknown = labels < 0
    if np.any(unknown):
        # Keep uncertain geometry in one neutral class. Unique geometry labels
        # here would reintroduce hundreds of artificial cuts into the atlas.
        labels[unknown] = len(_SEMANTIC_PARTS)
    labels, cleanup = _refine_face_regions(
        vertices,
        faces,
        labels,
        min_fragment_faces=max(24, int(np.ceil(n_faces * 0.001))),
        protected_labels={
            class_id for class_id in range(len(_SEMANTIC_PARTS))
            if np.sum(seed_labels == class_id) >= max(4, int(n_faces * 0.0001))
        },
        compact=False,
    )
    raw_region_ids = np.unique(labels)
    region_names = {
        int(new_id): (
            _SEMANTIC_PARTS[int(old_id)]
            if int(old_id) < len(_SEMANTIC_PARTS)
            else "unclassified"
        )
        for new_id, old_id in enumerate(raw_region_ids)
    }
    labels = np.searchsorted(raw_region_ids, labels).astype(np.int32)
    detected_parts = [
        name for i, name in enumerate(_SEMANTIC_PARTS)
        if np.any(seed_labels == i)
    ]
    return labels.astype(np.int32), {
        "semantic_model": "Grounding DINO tiny + SAM2",
        "semantic_model_source": detector_source,
        "semantic_parts_detected": detected_parts,
        "semantic_region_names": region_names,
        "semantic_detections": dict(detection_counts),
        "semantic_masks_accepted": dict(mask_counts),
        "semantic_masks_rejected": dict(rejected_mask_counts),
        "semantic_face_seeds": int(np.sum(seed_labels >= 0)),
        "semantic_seeds_by_part": {
            part: int(np.sum(seed_labels == class_id))
            for class_id, part in enumerate(_SEMANTIC_PARTS)
            if np.any(seed_labels == class_id)
        },
        "semantic_fragments_absorbed": cleanup["fragments_absorbed"],
        "semantic_faces_reassigned": cleanup["faces_reassigned"],
        "semantic_symmetric_shells_assigned": symmetric_shells_assigned,
        "semantic_nearby_arm_faces_assigned": nearby_arm_faces_assigned,
        "semantic_spatial_faces_assigned": spatially_assigned_faces,
        "semantic_multiviews": 4,
        "semantic_parameters": {
            "part_detection_threshold": float(detection_threshold),
            "mask_pred_iou_threshold": float(pred_iou_thresh),
            "mask_stability_threshold": float(stability_score_thresh),
        },
    }


def _geometry_regions(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Conservative normal-aware regions used only when SAM2 is unavailable."""
    n_faces = len(faces)
    if n_faces == 0:
        return np.zeros(0, dtype=np.int32)
    p = vertices[faces]
    normals = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals /= np.maximum(lengths[:, None], 1e-12)

    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces[(min(int(u), int(v)), max(int(u), int(v)))].append(fi)

    parent = np.arange(n_faces, dtype=np.int32)
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x
    def join(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    # A 60 degree threshold keeps smoothly curved character surfaces together.
    # ``abs`` also handles imported assets whose winding is inconsistent;
    # opposite normals alone must not turn every triangle into a UV island.
    for incident in edge_faces.values():
        if len(incident) == 2:
            a, b = incident
            if abs(float(normals[a] @ normals[b])) >= 0.50:
                join(a, b)
    roots = np.array([find(i) for i in range(n_faces)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int32)


def _refine_face_regions(
    vertices: np.ndarray,
    faces: np.ndarray,
    labels: np.ndarray,
    min_fragment_faces: int | None = None,
    compact: bool = True,
    protected_labels: set[int] | None = None,
) -> tuple[np.ndarray, dict[str, int]]:
    """Absorb tiny SAM labels into the most compatible edge-neighbour region.

    SAM masks are projected to visible faces, so faces missed by rasterization
    can become singleton labels. This topology-aware pass removes those specks
    without joining faces that do not share a mesh edge.
    """
    n_faces = len(faces)
    labels = np.asarray(labels, dtype=np.int32).copy()
    if n_faces == 0:
        return labels, {"components_merged": 0, "faces_reassigned": 0}
    if min_fragment_faces is None:
        min_fragment_faces = max(12, int(np.ceil(n_faces * 0.001)))

    points = vertices[faces]
    normals = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces[(min(int(u), int(v)), max(int(u), int(v)))].append(fi)
    adjacency: list[list[int]] = [[] for _ in range(n_faces)]
    for incident in edge_faces.values():
        if len(incident) == 2:
            a, b = incident
            adjacency[a].append(b)
            adjacency[b].append(a)

    merged = 0
    reassigned_faces = np.zeros(n_faces, dtype=bool)
    # Recompute components after each pass so a chain of small specks can be
    # absorbed progressively, while larger coherent regions remain anchors.
    for _ in range(4):
        seen = np.zeros(n_faces, dtype=bool)
        fragments: list[list[int]] = []
        for start in range(n_faces):
            if seen[start]:
                continue
            region = labels[start]
            stack = [start]
            seen[start] = True
            component = []
            while stack:
                face = stack.pop()
                component.append(face)
                for neighbour in adjacency[face]:
                    if not seen[neighbour] and labels[neighbour] == region:
                        seen[neighbour] = True
                        stack.append(neighbour)
            if len(component) <= min_fragment_faces and region not in (protected_labels or set()):
                fragments.append(component)
        if not fragments:
            break

        changed = False
        # Smallest components first: this reduces arbitrary dependence on face
        # ordering when several fragments touch each other.
        fragments.sort(key=len)
        for component in fragments:
            comp_set = set(component)
            boundary: dict[int, list[float]] = defaultdict(list)
            for face in component:
                for neighbour in adjacency[face]:
                    target = int(labels[neighbour])
                    if neighbour in comp_set or target == int(labels[face]):
                        continue
                    # Prefer a shared boundary and locally smooth continuation.
                    similarity = abs(float(normals[face] @ normals[neighbour]))
                    boundary[target].append(0.15 + 0.85 * similarity)
            if not boundary:
                continue
            target = max(boundary, key=lambda region: (sum(boundary[region]), len(boundary[region])))
            for face in component:
                labels[face] = target
                reassigned_faces[face] = True
            merged += 1
            changed = True
        if not changed:
            break

    if compact:
        _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int32), {
        "fragments_absorbed": merged,
        "faces_reassigned": int(reassigned_faces.sum()),
    }


def _sam2_regions(
    vertices: np.ndarray,
    faces: np.ndarray,
    points_per_side: int = 40,
    pred_iou_thresh: float = 0.50,
    stability_score_thresh: float = 0.75,
) -> np.ndarray:
    """Run SAM2 when its package is present.

    Full mesh rasterisation is intentionally isolated behind this function so
    installations with only the (large) checkpoint do not fail at import time.
    The current package-independent fallback remains useful for headless CPU
    deployments; a full SAM2 checkout enables this path.
    """
    local_source = _SAM_ROOT
    if local_source.exists() and str(local_source) not in sys.path:
        sys.path.insert(0, str(local_source))
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as exc:
        raise RuntimeError("SAM2 Python package or PyTorch is not installed") from exc
    if not _CHECKPOINT.is_file():
        raise RuntimeError(f"SAM2 checkpoint not found: {_CHECKPOINT}")

    # SAM2 needs a colour image.  Face-normal colour is a stable proxy for a
    # material-free mesh.  We deliberately sample face centroids instead of
    # using a GPU renderer: every pixel carries the face id that generated it.
    # This keeps the adapter portable and gives SAM2 enough visual boundaries
    # to identify coherent surface regions.
    from PIL import Image, ImageDraw
    p = vertices[faces]
    normals = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    centers = p.mean(axis=1)
    extent = np.maximum(vertices.max(axis=0) - vertices.min(axis=0), 1e-12)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_b+.yaml", str(_CHECKPOINT), device=device
    )
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )

    # Project on three principal planes. Masks are evidence for local face
    # adjacency, not permission to union every face visible inside a 2D mask.
    face_masks: list[set[int]] = [set() for _ in range(len(faces))]
    mask_id = 0
    for axes in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        u, v, depth_axis = axes
        size = 512
        lo, hi = vertices[:, [u, v]].min(axis=0), vertices[:, [u, v]].max(axis=0)
        scale = (size - 2) / np.maximum(hi - lo, 1e-12)
        xy = (vertices[:, [u, v]] - lo) * scale + 1
        rgb = Image.new("RGB", (size, size), (0, 0, 0))
        ids = Image.new("I", (size, size), 0)
        cr, ci = ImageDraw.Draw(rgb), ImageDraw.Draw(ids)
        order = np.argsort(centers[:, depth_axis])
        for fi in order:
            poly = [tuple(x) for x in xy[faces[fi]]]
            color = tuple(np.clip((normals[fi] + 1) * 110 + 20, 0, 255).astype(np.uint8))
            cr.polygon(poly, fill=color)
            ci.polygon(poly, fill=int(fi + 1))
        face_map = np.asarray(ids, dtype=np.int64) - 1
        for mask in generator.generate(np.asarray(rgb)):
            hit = face_map[np.asarray(mask["segmentation"], dtype=bool)]
            hit = hit[hit >= 0]
            if len(hit) >= 2:
                # Ignore masks that cover almost the complete visible model;
                # those do not describe a useful seam region.
                unique = set(np.unique(hit).tolist())
                if len(unique) < max(2, int(len(faces) * 0.85)):
                    for face in unique:
                        face_masks[face].add(mask_id)
                    mask_id += 1

    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for fi, (a, b, c) in enumerate(faces):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_faces[(min(int(u), int(v)), max(int(u), int(v)))].append(fi)
    parent = np.arange(len(faces), dtype=np.int32)
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x
    for incident in edge_faces.values():
        if len(incident) != 2:
            continue
        a, b = incident
        normal_similarity = abs(float(normals[a] @ normals[b]))
        shared_mask = not face_masks[a].isdisjoint(face_masks[b])
        # Only join topologically adjacent faces. A shared SAM region is a
        # useful cue, while unseen faces use a stricter smooth-surface prior.
        if (shared_mask and normal_similarity >= 0.72) or (
            not face_masks[a] and not face_masks[b] and normal_similarity >= 0.94
        ):
            parent[find(a)] = find(b)
    roots = np.array([find(i) for i in range(len(faces))])
    _, raw_labels = np.unique(roots, return_inverse=True)
    refined, diagnostics = _refine_face_regions(vertices, faces, raw_labels)
    _sam2_regions.last_refinement = diagnostics
    return refined


def discover_sam2_regions(
    vertices: np.ndarray,
    faces: np.ndarray,
    points_per_side: int = 40,
    pred_iou_thresh: float = 0.50,
    stability_score_thresh: float = 0.75,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a per-face region id and transparent inference diagnostics."""
    try:
        labels = _sam2_regions(
            vertices,
            faces,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )
        return labels, {
            "sam2_available": True,
            "sam2_mode": "multiview edge-consensus",
            "sam2_refinement": getattr(
                _sam2_regions,
                "last_refinement",
                {"fragments_absorbed": 0, "faces_reassigned": 0},
            ),
            "sam2_parameters": {
                "points_per_side": points_per_side,
                "pred_iou_thresh": pred_iou_thresh,
                "stability_score_thresh": stability_score_thresh,
            },
        }
    except Exception as exc:
        labels = _geometry_regions(vertices, faces)
        return labels, {
            "sam2_available": False,
            "sam2_mode": "geometry fallback",
            "sam2_note": str(exc),
        }


def split_on_region_seams(vertices: np.ndarray, faces: np.ndarray, face_regions: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Duplicate vertices across region boundaries, creating real UV seams."""
    # A vertex is shared only by faces with the same region.  This turns
    # semantic boundaries into topological boundaries that LSCM can flatten.
    remap: dict[tuple[int, int], int] = {}
    new_vertices: list[np.ndarray] = []
    new_faces = np.empty_like(faces, dtype=np.int32)
    for fi, face in enumerate(faces):
        region = int(face_regions[fi])
        for ci, vertex in enumerate(face):
            key = (int(vertex), region)
            if key not in remap:
                remap[key] = len(new_vertices)
                new_vertices.append(vertices[int(vertex)])
            new_faces[fi, ci] = remap[key]
    return np.asarray(new_vertices, dtype=np.float64), new_faces, len(new_vertices) - len(vertices)
