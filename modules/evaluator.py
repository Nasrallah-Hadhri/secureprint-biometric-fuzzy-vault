"""
Module d'évaluation biométrique — Protocole FVC standard.

Protocole officiel FVC2000 :
  - Paires genuines  : toutes les C(N_imp, 2) paires d'impressions du même doigt
  - Paires imposteurs: impression 1 du doigt i vs impression 1 du doigt j (i≠j)

Métriques : FAR, FRR, EER (Equal Error Rate).
"""

import os
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable

Fingers = Dict[int, List[str]]  # {finger_id: [path_imp1, path_imp2, ...]}


# ──────────────────────────────────────────────────────────────────────────────
# Découverte du dataset
# ──────────────────────────────────────────────────────────────────────────────

def discover_fvc_images(dataset_dir: str) -> Fingers:
    """
    Parcourt un dossier FVC et retourne {finger_id: [sorted paths]}.
    Accepte les noms de fichiers du type : {id}_{impression}.{ext}
    Exemples : 1_1.tif, 12_3.bmp, 110_8.tif
    """
    pattern = re.compile(
        r'^(\d+)_(\d+)\.(tif|tiff|bmp|png|jpg|jpeg)$', re.IGNORECASE
    )
    fingers: Fingers = {}

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dossier introuvable : {dataset_dir}")

    for fname in os.listdir(dataset_dir):
        m = pattern.match(fname)
        if m:
            fid = int(m.group(1))
            imp = int(m.group(2))
            path = os.path.join(dataset_dir, fname)
            fingers.setdefault(fid, []).append((imp, path))

    for fid in fingers:
        fingers[fid].sort(key=lambda x: x[0])
        fingers[fid] = [p for (_, p) in fingers[fid]]

    return fingers


# ──────────────────────────────────────────────────────────────────────────────
# Extraction de templates
# ──────────────────────────────────────────────────────────────────────────────

def _safe_extract(path: str) -> Optional[bytes]:
    """Extrait le template bytes depuis une image. Retourne None si erreur."""
    try:
        from modules.matcher import full_pipeline
        tpl, _ = full_pipeline(path)
        return tpl
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Calcul des scores
# ──────────────────────────────────────────────────────────────────────────────

def compute_all_scores(
    fingers: Fingers,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Tuple[List[float], List[float]]:
    """
    Calcule tous les scores genuins et imposteurs (protocole FVC standard).

    Returns:
        (genuine_scores, impostor_scores) — listes de scores cosinus ∈ [0,1].
    """
    from modules.template import template_from_bytes
    from modules.matcher import hellinger_similarity

    finger_ids = sorted(fingers.keys())
    n = len(finger_ids)

    # ── 1. Précalcul de tous les templates ──
    templates: Dict[int, List] = {}
    for idx, fid in enumerate(finger_ids):
        if progress_callback:
            progress_callback(idx, n * 2, f"Extraction doigt {fid} ({idx+1}/{n})…")
        templates[fid] = [_safe_extract(p) for p in fingers[fid]]

    # ── 2. Scores genuins (toutes paires du même doigt) ──
    genuine_scores: List[float] = []
    for fid in finger_ids:
        valid = [t for t in templates[fid] if t is not None]
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                va = template_from_bytes(valid[i])
                vb = template_from_bytes(valid[j])
                genuine_scores.append(hellinger_similarity(va, vb))

    # ── 3. Scores imposteurs (impression 1 de chaque doigt vs les autres) ──
    impostor_scores: List[float] = []
    ref = []
    for fid in finger_ids:
        valid = [t for t in templates[fid] if t is not None]
        ref.append(valid[0] if valid else None)

    for i in range(n):
        if progress_callback:
            progress_callback(n + i, n * 2, f"Paires imposteurs doigt {finger_ids[i]}…")
        if ref[i] is None:
            continue
        va = template_from_bytes(ref[i])
        for j in range(i + 1, n):
            if ref[j] is None:
                continue
            vb = template_from_bytes(ref[j])
            impostor_scores.append(hellinger_similarity(va, vb))

    return genuine_scores, impostor_scores


# ──────────────────────────────────────────────────────────────────────────────
# Métriques
# ──────────────────────────────────────────────────────────────────────────────

def compute_far_frr(
    genuine_scores: List[float],
    impostor_scores: List[float],
    thresholds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    FAR(t) = P(score ≥ t | imposteur)  — taux de faux acceptés
    FRR(t) = P(score <  t | genuine)   — taux de faux rejetés
    """
    gen = np.array(genuine_scores)
    imp = np.array(impostor_scores)
    far = np.array([np.mean(imp >= t) if len(imp) else 0.0 for t in thresholds])
    frr = np.array([np.mean(gen < t)  if len(gen) else 1.0 for t in thresholds])
    return far, frr


def find_eer(
    far: np.ndarray,
    frr: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[float, float]:
    """Trouve l'EER par interpolation linéaire. Retourne (eer, threshold_at_eer)."""
    diff = far - frr
    for i in range(len(diff) - 1):
        if diff[i] * diff[i + 1] <= 0:
            t0, t1 = thresholds[i], thresholds[i + 1]
            d0, d1 = diff[i], diff[i + 1]
            t_eer = t0 - d0 * (t1 - t0) / (d1 - d0) if (d1 - d0) != 0 else (t0 + t1) / 2
            eer = (float(np.interp(t_eer, thresholds, far)) +
                   float(np.interp(t_eer, thresholds, frr))) / 2
            return eer, float(t_eer)
    idx = np.argmin(np.abs(diff))
    return float((far[idx] + frr[idx]) / 2), float(thresholds[idx])


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ──────────────────────────────────────────────────────────────────────────────

def run_fvc_evaluation(
    dataset_dir: str,
    n_thresholds: int = 200,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """
    Pipeline complet d'évaluation FVC sur un dossier de dataset.

    Returns:
        dict avec genuine_scores, impostor_scores, far, frr, eer,
        eer_threshold, genuine_mean, impostor_mean, n_fingers, etc.
    """
    fingers = discover_fvc_images(dataset_dir)
    if len(fingers) < 2:
        raise ValueError(
            f"Pas assez de doigts détectés dans '{dataset_dir}' "
            f"(trouvé : {len(fingers)}). "
            "Vérifiez le chemin et le format des fichiers (ex: 1_1.tif)."
        )

    genuine_scores, impostor_scores = compute_all_scores(fingers, progress_callback)

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    far, frr = compute_far_frr(genuine_scores, impostor_scores, thresholds)
    eer, eer_threshold = find_eer(far, frr, thresholds)

    return {
        "dataset_dir":      dataset_dir,
        "n_fingers":        len(fingers),
        "n_impressions":    {fid: len(fingers[fid]) for fid in fingers},
        "n_genuine_pairs":  len(genuine_scores),
        "n_impostor_pairs": len(impostor_scores),
        "genuine_scores":   genuine_scores,
        "impostor_scores":  impostor_scores,
        "thresholds":       thresholds.tolist(),
        "far":              far.tolist(),
        "frr":              frr.tolist(),
        "eer":              eer,
        "eer_threshold":    eer_threshold,
        "genuine_mean":     float(np.mean(genuine_scores))   if genuine_scores   else 0.0,
        "impostor_mean":    float(np.mean(impostor_scores))  if impostor_scores  else 0.0,
        "genuine_std":      float(np.std(genuine_scores))    if genuine_scores   else 0.0,
        "impostor_std":     float(np.std(impostor_scores))   if impostor_scores  else 0.0,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Export de graphique
# ──────────────────────────────────────────────────────────────────────────────

def save_evaluation_plot(results: dict, output_path: str) -> None:
    """
    Sauvegarde un graphique à deux panneaux :
      - Gauche : distributions des scores genuins / imposteurs
      - Droite : courbes FAR / FRR avec point EER
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gen  = results["genuine_scores"]
    imp  = results["impostor_scores"]
    far  = np.array(results["far"])
    frr  = np.array(results["frr"])
    thr  = np.array(results["thresholds"])
    eer  = results["eer"]
    eer_t = results["eer_threshold"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0F172A")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1E293B")
        for spine in ax.spines.values():
            spine.set_edgecolor("#334155")
        ax.tick_params(colors="#94A3B8")
        ax.xaxis.label.set_color("#94A3B8")
        ax.yaxis.label.set_color("#94A3B8")
        ax.title.set_color("white")

    # ── Panneau 1 : distributions ──
    bins = np.linspace(0, 1, 40)
    ax1.hist(gen, bins=bins, alpha=0.7, color="#16A34A", label=f"Genuins (n={len(gen)})")
    ax1.hist(imp, bins=bins, alpha=0.7, color="#DC2626", label=f"Imposteurs (n={len(imp)})")
    ax1.axvline(eer_t, color="#FBBF24", linestyle="--", linewidth=1.5,
                label=f"Seuil EER = {eer_t:.3f}")
    ax1.set_xlabel("Score de similarité Hellinger")
    ax1.set_ylabel("Nombre de paires")
    ax1.set_title("Distribution des scores")
    ax1.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="white")

    # ── Panneau 2 : FAR / FRR ──
    ax2.plot(thr, far * 100, color="#DC2626", linewidth=2, label="FAR (%)")
    ax2.plot(thr, frr * 100, color="#2563EB", linewidth=2, label="FRR (%)")
    ax2.axvline(eer_t, color="#FBBF24", linestyle="--", linewidth=1.5,
                label=f"EER = {eer*100:.2f}% @ {eer_t:.3f}")
    ax2.scatter([eer_t], [eer * 100], color="#FBBF24", s=80, zorder=5)
    ax2.set_xlabel("Seuil de décision")
    ax2.set_ylabel("Taux d'erreur (%)")
    ax2.set_title("Courbes FAR / FRR")
    ax2.legend(facecolor="#0F172A", edgecolor="#334155", labelcolor="white")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 100)

    fig.suptitle(
        f"SecurePrint — Évaluation FVC  |  {results['n_fingers']} doigts  |  EER = {eer*100:.2f}%",
        color="white", fontsize=13, y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
