"""
Module 7 : Matching biométrique.

Métrique : Similarité de Hellinger + Recherche multi-rotations
Idéal pour la robustesse aux distorsions non linéaires de la peau
tout en garantissant l'invariance à la rotation.
"""

import cv2
import numpy as np
from typing import Tuple, List

from modules.preprocessor import preprocess, get_original
from modules.minutiae import extract_minutiae
from modules.template import generate_template, template_from_bytes

DEFAULT_THRESHOLD = 0.40   # Calibré pour HOG 256-dim avec rotation


def hellinger_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    a = np.sqrt(np.abs(vec_a))
    b = np.sqrt(np.abs(vec_b))
    dist = np.linalg.norm(a - b) / np.sqrt(2.0)
    return float(max(0.0, 1.0 - dist))


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Effectue une rotation de l'image (squelette) autour de son centre."""
    h, w = image.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated


def full_pipeline(image_path: str) -> Tuple[bytes, list]:
    """
    Extrait le template de référence (sans rotation).
    """
    skeleton = preprocess(image_path)
    minutiae = extract_minutiae(skeleton)
    template_bytes = generate_template(skeleton)
    return template_bytes, minutiae


def generate_rotated_candidates(skeleton: np.ndarray) -> List[bytes]:
    """
    Génère des templates pour plusieurs angles afin d'assurer l'invariance à la rotation.
    Angles testés : de -30° à +30° par pas de 10°.
    """
    angles = [0, -10, 10, -20, 20, -30, 30]
    candidates = []
    for ang in angles:
        rot_skel = rotate_image(skeleton, ang)
        candidates.append(generate_template(rot_skel))
    return candidates


def compare(template_a_bytes: bytes,
            template_b_bytes: bytes,
            threshold: float = DEFAULT_THRESHOLD) -> Tuple[float, bool]:
    """
    Compare deux templates (bytes) via la similarité de Hellinger.
    """
    vec_a = template_from_bytes(template_a_bytes)
    vec_b = template_from_bytes(template_b_bytes)

    min_len = min(len(vec_a), len(vec_b))
    if min_len == 0:
        return 0.0, False

    score = hellinger_similarity(vec_a[:min_len], vec_b[:min_len])
    return score, score >= threshold


def authenticate(image_path: str,
                 stored_encrypted_blob: bytes,
                 threshold: float = DEFAULT_THRESHOLD) -> Tuple[float, bool]:
    """
    Authentifie une empreinte. Teste plusieurs rotations de l'image candidate
    pour trouver le meilleur alignement géométrique avec l'empreinte stockée.
    """
    from modules.encryption import decrypt_template
    stored_bytes = decrypt_template(stored_encrypted_blob)
    
    # Prétraitement de l'image candidate
    skeleton = preprocess(image_path)
    
    # Génère les templates pour plusieurs angles de rotation
    candidates_bytes = generate_rotated_candidates(skeleton)
    
    max_score = 0.0
    for cand_bytes in candidates_bytes:
        score, _ = compare(stored_bytes, cand_bytes, threshold)
        if score > max_score:
            max_score = score
            
    return max_score, max_score >= threshold
