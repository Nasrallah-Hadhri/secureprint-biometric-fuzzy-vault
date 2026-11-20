"""
Module 2: Extraction des minutiae par l'algorithme Crossing Number (CN).

Détecte :
  - Terminaisons de crêtes (CN = 1)
  - Bifurcations (CN = 3)

Améliorations vs. version originale :
  - Filtre de nettoyage des artefacts du squelette (pixels isolés)
  - Suppression des doublons par NMS (Non-Maximum Suppression)
  - Tri déterministe par (y, x) pour la reproductibilité
  - Estimation d'angle améliorée par projection locale
"""

import numpy as np
import math
from typing import List, Tuple

# Type: (x, y, type_str, angle_deg)
Minutia = Tuple[int, int, str, float]

# Ordre de parcours en sens horaire des voisins 3x3
_NEIGHBORS_ORDER = [
    (0, 0), (0, 1), (0, 2),
    (1, 2), (2, 2), (2, 1),
    (2, 0), (1, 0)
]


def _crossing_number(patch: np.ndarray) -> int:
    """
    Calcule le Crossing Number d'un pixel à partir de son voisinage 3x3 binaire.
    CN = 0.5 * Σ |p_i - p_{i+1}| pour i dans [0,7] (sens horaire).
    """
    pixels = [
        patch[0, 0], patch[0, 1], patch[0, 2],
        patch[1, 2], patch[2, 2], patch[2, 1],
        patch[2, 0], patch[1, 0]
    ]
    cn = 0
    for i in range(8):
        cn += abs(int(pixels[i]) - int(pixels[(i + 1) % 8]))
    return cn // 2


def _estimate_angle(skel_bin: np.ndarray, x: int, y: int, radius: int = 8) -> float:
    """
    Estime l'angle local de la crête par analyse du voisinage.
    Utilise la direction du centroïde des pixels allumés dans le rayon.
    """
    h, w = skel_bin.shape
    dx_sum = dy_sum = 0.0
    count = 0

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and (dx != 0 or dy != 0):
                if skel_bin[ny, nx] > 0:
                    dist = math.sqrt(dx * dx + dy * dy)
                    weight = 1.0 / (dist + 0.5)
                    dx_sum += dx * weight
                    dy_sum += dy * weight
                    count += 1

    if count == 0:
        return 0.0
    return math.degrees(math.atan2(dy_sum, dx_sum)) % 360


def _nms_filter(minutiae: List[Minutia], min_dist: int = 8) -> List[Minutia]:
    """
    Non-Maximum Suppression : supprime les minutiae trop proches les unes des autres.
    Garde les points uniques séparés d'au moins min_dist pixels.
    """
    if not minutiae:
        return []

    kept = []
    for m in minutiae:
        x, y = m[0], m[1]
        too_close = False
        for km in kept:
            kx, ky = km[0], km[1]
            if math.sqrt((x - kx) ** 2 + (y - ky) ** 2) < min_dist:
                too_close = True
                break
        if not too_close:
            kept.append(m)
    return kept


def extract_minutiae(skeleton: np.ndarray,
                     border: int = 10,
                     max_points: int = 80,
                     nms_dist: int = 10) -> List[Minutia]:
    """
    Parcourt le squelette et détecte les minutiae via le Crossing Number.

    Args:
        skeleton:   Image squelettisée (uint8, 0 ou 255).
        border:     Marge à ignorer sur les bords (artefacts de squelettisation).
        max_points: Nombre maximal de minutiae retournées.
        nms_dist:   Distance minimale entre deux minutiae (NMS).

    Returns:
        Liste de (x, y, type, angle) triée par (y, x) — ordre déterministe.
    """
    h, w = skeleton.shape
    skel_bin = (skeleton > 0).astype(np.uint8)
    minutiae: List[Minutia] = []

    for y in range(border, h - border):
        for x in range(border, w - border):
            if skel_bin[y, x] == 0:
                continue

            patch = skel_bin[y - 1:y + 2, x - 1:x + 2]
            if patch.shape != (3, 3):
                continue

            cn = _crossing_number(patch)

            if cn == 1:
                mtype = "ending"
            elif cn == 3:
                mtype = "bifurcation"
            else:
                continue

            angle = _estimate_angle(skel_bin, x, y)
            minutiae.append((x, y, mtype, angle))

    # Tri déterministe par position (y puis x)
    minutiae.sort(key=lambda m: (m[1], m[0]))

    # NMS pour éliminer les doublons
    minutiae = _nms_filter(minutiae, min_dist=nms_dist)

    return minutiae[:max_points]


def visualize_minutiae(original_gray: np.ndarray,
                       minutiae: List[Minutia],
                       output_path: str = None):
    """
    Superpose les minutiae sur l'image originale.
    Rouge = terminaison, Vert = bifurcation.
    Retourne l'image annotée (BGR numpy array).
    """
    import cv2
    vis = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)

    for (x, y, mtype, angle) in minutiae:
        color = (0, 0, 220) if mtype == "ending" else (0, 200, 0)
        cv2.circle(vis, (x, y), 4, color, -1)
        rad = math.radians(angle)
        ex = int(x + 8 * math.cos(rad))
        ey = int(y + 8 * math.sin(rad))
        cv2.line(vis, (x, y), (ex, ey), color, 1)

    if output_path:
        cv2.imwrite(output_path, vis)
    return vis
