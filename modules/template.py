"""
Module 3 : Génération du descripteur biométrique (256 dimensions).
"""

import cv2
import numpy as np
from typing import List, Tuple, Union

Minutia = Tuple[int, int, str, float]

TARGET_SIZE = 200
GRID_D      = 8
GRID_O      = 8
GRID_H      = 4
HOG_BINS    = 8
FEATURE_DIM = GRID_D**2 + GRID_O**2 + GRID_H**2 * HOG_BINS

def _align_skeleton(skel: np.ndarray) -> np.ndarray:
    coords = np.argwhere(skel > 0)
    if len(coords) == 0:
        return np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    margin = 4
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(skel.shape[0] - 1, y1 + margin)
    x1 = min(skel.shape[1] - 1, x1 + margin)

    cropped = skel[y0:y1 + 1, x0:x1 + 1]
    resized = cv2.resize(cropped, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8) * 255

def _density_field(skel: np.ndarray, grid: int = GRID_D) -> np.ndarray:
    h, w = skel.shape
    ch, cw = h // grid, w // grid
    features = np.zeros(grid * grid, dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            cell = skel[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
            features[i * grid + j] = (cell > 0).sum() / max(cell.size, 1)
    return features

def _orientation_field(skel: np.ndarray, grid: int = GRID_O) -> np.ndarray:
    gx = cv2.Sobel(skel.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(skel.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    angles = np.arctan2(np.abs(gy), np.abs(gx))
    
    h, w = skel.shape
    ch, cw = h // grid, w // grid
    features = np.zeros(grid * grid, dtype=np.float32)
    for i in range(grid):
        for j in range(grid):
            cell_mag = mag[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
            cell_ang = angles[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
            total = cell_mag.sum()
            if total > 1e-8:
                features[i * grid + j] = (cell_ang * cell_mag).sum() / total / (np.pi / 2)
    return features

def _hog_descriptor(skel: np.ndarray, grid: int = GRID_H, n_bins: int = HOG_BINS) -> np.ndarray:
    gx = cv2.Sobel(skel.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(skel.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    angles = np.arctan2(np.abs(gy), np.abs(gx))
    
    h, w = skel.shape
    ch, cw = h // grid, w // grid
    features = []
    for i in range(grid):
        for j in range(grid):
            cell_mag = mag[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw].ravel()
            cell_ang = angles[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw].ravel()
            hist, _ = np.histogram(cell_ang, bins=n_bins, range=(0.0, np.pi / 2), weights=cell_mag)
            norm = np.linalg.norm(hist)
            features.append(hist / norm if norm > 1e-8 else hist)
    return np.concatenate(features).astype(np.float32)

def generate_template(skeleton_input: Union[np.ndarray, List]) -> bytes:
    if not isinstance(skeleton_input, np.ndarray):
        return np.zeros(FEATURE_DIM, dtype=np.float32).tobytes()
    aligned = _align_skeleton(skeleton_input)
    density = _density_field(aligned)
    ori = _orientation_field(aligned)
    hog = _hog_descriptor(aligned)
    vec = np.concatenate([density, ori, hog]).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm
    return vec.tobytes()

def template_from_bytes(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float32).copy()
