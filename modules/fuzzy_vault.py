"""
Module 4 : Fuzzy Vault Scheme avec Invariants Topologiques Locaux.

Remplace les coordonnées absolues (instables) par des descripteurs locaux 
(distances et angles relatifs des 3 plus proches voisins) pour garantir
une invariance parfaite à la translation et à la rotation, tout en 
assurant une unicité cryptographique.
"""

import numpy as np
import hashlib
import json
import math
from typing import List, Dict, Any, Tuple

DEGREE = 9          # Nécessite 10 voisinages identiques pour déverrouiller
CHAFF_RATIO = 8     # 8 leurres par minutie
PRIME = 1048573     # Grand nombre premier pour le polynôme

Minutia = Tuple[int, int, str, float]

def _poly_eval(coeffs: List[int], x: int, prime: int = PRIME) -> int:
    result = 0
    for i, c in enumerate(coeffs):
        result = (result + c * pow(x, i, prime)) % prime
    return result

def _key_to_coeffs(secret_key: bytes, degree: int, prime: int = PRIME) -> List[int]:
    coeffs = []
    for i in range(degree + 1):
        digest = hashlib.sha256(secret_key + i.to_bytes(4, "big")).digest()
        c = int.from_bytes(digest[:4], "big") % prime
        coeffs.append(c)
    return coeffs

def _lagrange_interpolation(points: List[tuple], prime: int = PRIME) -> int:
    n = len(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    def mod_inv(a, m):
        return pow(a, m - 2, m)

    result = 0
    for i in range(n):
        num = ys[i]
        den = 1
        for j in range(n):
            if i != j:
                num = (num * (-xs[j] % prime)) % prime
                den = (den * ((xs[i] - xs[j]) % prime)) % prime
        result = (result + num * mod_inv(den, prime)) % prime
    return result

def _angle_diff(a1: float, a2: float) -> float:
    """Retourne la différence d'angle entre 0 et 360."""
    diff = abs(a1 - a2)
    return min(diff, 360 - diff)

def extract_features(minutiae: List[Minutia]) -> List[int]:
    """
    Crée un identifiant unique (hash topologique) pour chaque minutie
    en se basant sur ses 3 plus proches voisins.
    Totalement invariant à la rotation et à la translation.
    """
    n = len(minutiae)
    if n < 4:
        return []
    
    features = []
    for i in range(n):
        x0, y0, t0_str, a0 = minutiae[i]
        t0 = 1 if t0_str == "Bifurcation" else 0
        
        neighbors = []
        for j in range(n):
            if i == j: continue
            x1, y1, t1_str, a1 = minutiae[j]
            t1 = 1 if t1_str == "Bifurcation" else 0
            
            d = math.hypot(x1 - x0, y1 - y0)
            ang_rel = _angle_diff(a0, a1)
            neighbors.append((d, ang_rel, t1))
            
        # Trier par distance
        neighbors.sort(key=lambda x: x[0])
        
        # Prendre les 3 plus proches
        n1, n2, n3 = neighbors[:3]
        
        # Quantification pour absorber la distorsion élastique de la peau
        # Distance par pas de 15px, Angle par pas de 30°
        def quantize(n):
            qd = min(31, int(n[0] / 15.0))
            qa = min(15, int(n[1] / 30.0))
            return qd, qa, n[2]
            
        qd1, qa1, qt1 = quantize(n1)
        qd2, qa2, qt2 = quantize(n2)
        qd3, qa3, qt3 = quantize(n3)
        
        # Packing bit à bit pour créer un entier de 30 bits (~1 milliard)
        # Chaque voisin prend 10 bits: 5 pour d, 4 pour a, 1 pour t
        val = 0
        val = (val << 5) | qd1; val = (val << 4) | qa1; val = (val << 1) | qt1
        val = (val << 5) | qd2; val = (val << 4) | qa2; val = (val << 1) | qt2
        val = (val << 5) | qd3; val = (val << 4) | qa3; val = (val << 1) | qt3
        
        # On ajoute le type de la minutie centrale (1 bit)
        val = (val << 1) | t0
        
        features.append(val)
        
    return list(set(features))

def create_vault(minutiae: List[Minutia], secret_key: bytes) -> Dict[str, Any]:
    minutiae_ints = extract_features(minutiae)
    coeffs = _key_to_coeffs(secret_key, DEGREE)

    genuine = []
    used_xs = set()
    for x in minutiae_ints:
        x_mod = x % PRIME
        if x_mod not in used_xs:
            y = _poly_eval(coeffs, x_mod)
            genuine.append((x_mod, y))
            used_xs.add(x_mod)

    chaff = []
    rng = np.random.default_rng()
    n_chaff = max(100, len(genuine) * CHAFF_RATIO)
    attempts = 0
    # On tire les leurres dans un grand espace (2^31)
    while len(chaff) < n_chaff and attempts < 100000:
        xc = int(rng.integers(0, 2000000000))
        xc_mod = xc % PRIME
        if xc_mod in used_xs:
            attempts += 1
            continue
        yc_true = _poly_eval(coeffs, xc_mod)
        yc_fake = int(rng.integers(0, PRIME))
        if yc_fake == yc_true:
            yc_fake = (yc_fake + 1) % PRIME
        chaff.append((xc_mod, yc_fake))
        used_xs.add(xc_mod)
        attempts += 1

    all_points = genuine + chaff
    rng.shuffle(all_points)
    key_hash = hashlib.sha256(secret_key).hexdigest()

    return {
        "all_points": [(int(x), int(y)) for x, y in all_points],
        "key_hash": key_hash,
        "degree": DEGREE,
        "n_genuine": len(genuine),
    }

def unlock_vault(vault: Dict[str, Any], candidate_minutiae: List[Minutia], secret_key_guess: bytes) -> bool:
    candidate_ints = extract_features(candidate_minutiae)
    all_points = vault["all_points"]
    points_dict = {x: y for x, y in all_points}
    degree = vault.get("degree", DEGREE)

    matched = []
    for x in candidate_ints:
        x_mod = x % PRIME
        if x_mod in points_dict:
            matched.append((x_mod, points_dict[x_mod]))

    if len(matched) < degree + 1:
        return False

    import itertools
    max_combos = 500
    combos = itertools.islice(itertools.combinations(matched, degree + 1), max_combos)
    expected_p0 = _key_to_coeffs(secret_key_guess, degree)[0]
    
    for subset in combos:
        try:
            p0 = _lagrange_interpolation(list(subset))
            if p0 == expected_p0:
                return True
        except Exception:
            continue
            
    return False
