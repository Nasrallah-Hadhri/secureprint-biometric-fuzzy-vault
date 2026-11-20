"""
Module 1 : Prétraitement des images d'empreintes digitales.

Pipeline :
  1. Normalisation globale (moyenne=128, variance=64)
  2. CLAHE pour le contraste local
  3. Seuillage adaptatif gaussien (17×17) → binaire robuste
  4. Morphologie : fermeture (combler les crêtes) + ouverture (bruit)
  5. Squelettisation (scikit-image)
  6. Élagage (pruning) des branches courtes (< 6 px) → supprime les faux minutiae
"""

import cv2
import numpy as np
from skimage.morphology import skeletonize as ski_skeletonize


# ─── Normalisation ───────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Charge une image en niveaux de gris."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Impossible de charger l'image : {path}")
    return img


def normalize_contrast(img: np.ndarray) -> np.ndarray:
    """
    Normalisation globale (µ=128, σ=64) puis CLAHE local.
    Compense les variations de pression du doigt sur le capteur.
    """
    img_f = img.astype(np.float32)
    mean, std = img_f.mean(), img_f.std()
    if std < 1e-6:
        std = 1.0
    normalized = (img_f - mean) / std * 64.0 + 128.0
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(normalized)


# ─── Binarisation ────────────────────────────────────────────────────────────

def binarize(img: np.ndarray) -> np.ndarray:
    """
    Binarisation adaptative Gaussienne (17×17) + morphologie.
    Superieur à Otsu pour les empreintes avec éclairage non uniforme.
    """
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Seuillage adaptatif Gaussien (voisinage 17×17, constante C=4)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 17, 4
    )

    # Fermeture (Close) : comble les micro-trous dans les crêtes
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

    # Ouverture (Open) : supprime le bruit isolé
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)

    return binary


# ─── Squelettisation + élagage ───────────────────────────────────────────────

def skeletonize(binary: np.ndarray) -> np.ndarray:
    """Squelettisation : réduit les crêtes à 1 pixel d'épaisseur."""
    binary_norm = (binary > 0).astype(np.uint8)
    skeleton = ski_skeletonize(binary_norm)
    return skeleton.astype(np.uint8) * 255


def _prune_skeleton(skeleton: np.ndarray, iterations: int = 6) -> np.ndarray:
    """
    Élagage itératif des pixels terminaux (branches courtes < iterations px).
    Supprime les artefacts de binarisation qui génèrent de faux minutiae.

    Approche vectorisée : à chaque passe, on retire les pixels à 1 seul voisin.
    """
    skel = (skeleton > 0).astype(np.uint8)

    for _ in range(iterations):
        # Somme des 8 voisins (connexité-8)
        nbrs = (
            np.roll(skel,  1, axis=0) + np.roll(skel, -1, axis=0) +
            np.roll(skel,  1, axis=1) + np.roll(skel, -1, axis=1) +
            np.roll(np.roll(skel,  1, axis=0),  1, axis=1) +
            np.roll(np.roll(skel,  1, axis=0), -1, axis=1) +
            np.roll(np.roll(skel, -1, axis=0),  1, axis=1) +
            np.roll(np.roll(skel, -1, axis=0), -1, axis=1)
        )
        # Pixels terminaux : allumés ET exactement 1 voisin
        terminals = (skel == 1) & (nbrs == 1)
        if not terminals.any():
            break
        skel = skel.copy()
        skel[terminals] = 0

    return skel * 255


# ─── Pipeline principal ───────────────────────────────────────────────────────

def preprocess(image_path: str) -> np.ndarray:
    """
    Pipeline complet : image brute → squelette élagué (uint8, 0 ou 255).
    Prêt pour l'extraction de features HOG/densité.
    """
    img  = load_image(image_path)
    img  = normalize_contrast(img)
    bin_ = binarize(img)
    skel = skeletonize(bin_)
    skel = _prune_skeleton(skel, iterations=3)
    return skel


def get_original(image_path: str) -> np.ndarray:
    """Retourne l'image originale en niveaux de gris (pour affichage)."""
    return load_image(image_path)
