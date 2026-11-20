import cv2
import numpy as np

def align_skeleton(skel: np.ndarray, target_size: int = 200) -> np.ndarray:
    """Aligne et met à l'échelle le squelette sur sa bounding box."""
    coords = np.argwhere(skel > 0)
    if len(coords) == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)

    # Ajouter une marge pour ne pas couper les bords
    margin = 4
    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(skel.shape[0] - 1, y1 + margin)
    x1 = min(skel.shape[1] - 1, x1 + margin)

    cropped = skel[y0:y1 + 1, x0:x1 + 1]
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return (resized > 0).astype(np.uint8) * 255
