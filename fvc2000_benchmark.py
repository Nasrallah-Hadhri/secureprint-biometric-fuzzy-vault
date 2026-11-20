#!/usr/bin/env python3
"""
SecurePrint — Script d'évaluation FVC2000 (CLI).

Usage:
    python fvc2000_benchmark.py --db data/fvc2000/DB1_B
    python fvc2000_benchmark.py --db data/fvc2000/DB1_B --plot results/eval.png
    python fvc2000_benchmark.py --db data/fvc2000/DB1_B --enroll-db

Protocole FVC standard :
  - Genuins   : toutes les C(N,2) paires d'impressions du même doigt
  - Imposteurs: impression 1 de chaque doigt vs les autres doigts
"""

import argparse
import os
import sys
import time
import numpy as np

# ── S'assurer que le répertoire du projet est dans le path ──────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ────────────────────────────────────────────────────────────────────────────
# Helpers d'affichage
# ────────────────────────────────────────────────────────────────────────────

W = 62  # largeur de la boîte


def _box(lines, color_code=""):
    reset = "\033[0m" if color_code else ""
    print(color_code + "┌" + "─" * W + "┐" + reset)
    for line in lines:
        pad = W - len(line)
        print(color_code + "│ " + line + " " * (pad - 1) + "│" + reset)
    print(color_code + "└" + "─" * W + "┘" + reset)


def _progress(current, total, msg):
    pct = int(100 * current / max(total, 1))
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print(f"\r  [{bar}] {pct:3d}%  {msg[:38]:<38}", end="", flush=True)


# ────────────────────────────────────────────────────────────────────────────
# Enrôlement dans la base SecurePrint (optionnel)
# ────────────────────────────────────────────────────────────────────────────

def enroll_fvc_into_db(fingers: dict, db_prefix: str = "FVC") -> None:
    """Enrôle l'impression 1 de chaque doigt dans la base SQLite SecurePrint."""
    from modules.storage import init_db, enroll_user, get_user_by_name, delete_user
    from modules.encryption import generate_key, encrypt_template
    from modules.matcher import full_pipeline

    init_db()
    generate_key()

    ok = fail = skip = 0
    for fid, paths in sorted(fingers.items()):
        name = f"{db_prefix}_finger_{fid:03d}"
        # Supprimer si déjà existant
        existing = get_user_by_name(name)
        if existing:
            delete_user(existing[0])

        try:
            tpl, minutiae = full_pipeline(paths[0])
            encrypted = encrypt_template(tpl)
            enroll_user(name, encrypted)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"\n  ⚠ Enrôlement échoué pour {name}: {e}")

    print(f"\n  ✓ {ok} doigts enrôlés  |  {fail} échecs")


# ────────────────────────────────────────────────────────────────────────────
# Évaluation via la base SecurePrint (vérification complète)
# ────────────────────────────────────────────────────────────────────────────

def test_via_db(fingers: dict, db_prefix: str = "FVC", threshold: float = 0.60) -> dict:
    """
    Teste les impressions 2..N de chaque doigt contre le template stocké.
    Retourne un dictionnaire de métriques système.
    """
    from modules.storage import get_user_by_name, get_template
    from modules.matcher import full_pipeline, compare
    from modules.encryption import decrypt_template

    tp = fp = tn = fn = 0
    finger_ids = sorted(fingers.keys())

    for i, fid in enumerate(finger_ids):
        name = f"{db_prefix}_finger_{fid:03d}"
        user = get_user_by_name(name)
        if not user:
            continue
        blob = get_template(user[0])
        if not blob:
            continue
        stored = decrypt_template(blob)

        # ── Genuine tests: impressions 2..N du même doigt ──
        for path in fingers[fid][1:]:
            try:
                cand, _ = full_pipeline(path)
                _, is_match = compare(stored, cand, threshold)
                if is_match:
                    tp += 1
                else:
                    fn += 1
            except Exception:
                pass

        # ── Impostor test: impression 1 d'un autre doigt ──
        other_ids = [x for x in finger_ids if x != fid]
        for other_id in other_ids[:5]:  # limiter pour la rapidité
            other_name = f"{db_prefix}_finger_{other_id:03d}"
            other_user = get_user_by_name(other_name)
            if not other_user:
                continue
            try:
                cand, _ = full_pipeline(fingers[other_id][0])
                _, is_match = compare(stored, cand, threshold)
                if is_match:
                    fp += 1
                else:
                    tn += 1
            except Exception:
                pass

    total_gen = tp + fn
    total_imp = fp + tn
    tar = tp / total_gen if total_gen else 0
    far_sys = fp / total_imp if total_imp else 0

    return {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "TAR": tar, "FAR_sys": far_sys,
        "total_genuine": total_gen,
        "total_impostor": total_imp,
    }


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SecurePrint — Évaluation FVC2000",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db", required=True,
        help="Chemin vers le dossier FVC (ex: data/fvc2000/DB1_B)"
    )
    parser.add_argument(
        "--plot", default=None,
        help="Chemin de sortie pour le graphique PNG (optionnel)"
    )
    parser.add_argument(
        "--enroll-db", action="store_true",
        help="Enrôle les doigts dans la base SQLite SecurePrint avant l'éval"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.15,
        help="Seuil d'authentification pour les tests système (défaut: 0.15)"
    )
    parser.add_argument(
        "--thresholds", type=int, default=200,
        help="Nombre de seuils pour les courbes FAR/FRR (défaut: 200)"
    )
    args = parser.parse_args()

    print()
    _box([
        "SecurePrint — Évaluation Biométrique FVC2000",
        "",
        f"  Dataset : {os.path.basename(args.db)}",
        f"  Chemin  : {args.db}",
    ], "\033[94m")
    print()

    # ── Découverte des images ──────────────────────────────────────────────
    from modules.evaluator import (
        discover_fvc_images, run_fvc_evaluation, save_evaluation_plot
    )

    print("  [1/4] Découverte des images…", flush=True)
    try:
        fingers = discover_fvc_images(args.db)
    except FileNotFoundError as e:
        print(f"\n  \033[91m✗ {e}\033[0m")
        sys.exit(1)

    n_fingers = len(fingers)
    n_images  = sum(len(v) for v in fingers.values())
    print(f"        ✓ {n_fingers} doigts  |  {n_images} images")

    # ── Enrôlement dans la base (optionnel) ───────────────────────────────
    if args.enroll_db:
        print(f"\n  [2/4] Enrôlement dans la base SQLite…")
        enroll_fvc_into_db(fingers)
    else:
        print(f"\n  [2/4] (--enroll-db non activé, enrôlement ignoré)")

    # ── Évaluation FAR/FRR/EER ────────────────────────────────────────────
    print(f"\n  [3/4] Calcul des scores (protocole FVC)…\n")
    t0 = time.time()

    try:
        results = run_fvc_evaluation(
            args.db,
            n_thresholds=args.thresholds,
            progress_callback=_progress,
        )
    except Exception as e:
        print(f"\n  \033[91m✗ Évaluation échouée : {e}\033[0m")
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"\n        ✓ Terminé en {elapsed:.1f}s\n")

    # ── Résultats ─────────────────────────────────────────────────────────
    eer     = results["eer"] * 100
    eer_t   = results["eer_threshold"]
    g_mean  = results["genuine_mean"]
    i_mean  = results["impostor_mean"]
    n_gen   = results["n_genuine_pairs"]
    n_imp   = results["n_impostor_pairs"]

    eer_color = "\033[92m" if eer < 10 else ("\033[93m" if eer < 25 else "\033[91m")
    print()
    _box([
        "RÉSULTATS — Protocole FVC Standard",
        "",
        f"  Doigts analysés      : {results['n_fingers']}",
        f"  Paires genuines      : {n_gen}",
        f"  Paires imposteurs    : {n_imp}",
        "",
        f"  Score moyen genuins  : {g_mean:.4f}",
        f"  Score moyen impostor : {i_mean:.4f}",
        "",
        f"  EER (Equal Error Rate): {eer:.2f}%  @ seuil={eer_t:.4f}",
    ], eer_color)

    # ── Test système via base SQLite ───────────────────────────────────────
    if args.enroll_db:
        print(f"\n  [4/4] Test système (seuil={args.threshold})…")
        sys_res = test_via_db(fingers, threshold=args.threshold)
        tar = sys_res["TAR"] * 100
        far_sys = sys_res["FAR_sys"] * 100
        print()
        _box([
            "RÉSULTATS — Test Système (Base SQLite SecurePrint)",
            "",
            f"  Vrais positifs (TP)  : {sys_res['TP']}",
            f"  Faux négatifs (FN)   : {sys_res['FN']}",
            f"  Faux positifs (FP)   : {sys_res['FP']}",
            f"  Vrais négatifs (TN)  : {sys_res['TN']}",
            "",
            f"  TAR (True Accept)    : {tar:.2f}%",
            f"  FAR système          : {far_sys:.2f}%",
        ], "\033[96m")
    else:
        print(f"\n  [4/4] (Saut du test système — utilisez --enroll-db)")

    # ── Graphique ─────────────────────────────────────────────────────────
    if args.plot:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(args.plot)), exist_ok=True)
            save_evaluation_plot(results, args.plot)
            print(f"\n  📊 Graphique sauvegardé : {args.plot}")
        except Exception as e:
            print(f"\n  ⚠ Impossible de sauvegarder le graphique : {e}")

    print()
    _box(["✅  Évaluation terminée."], "\033[92m")
    print()


if __name__ == "__main__":
    main()
