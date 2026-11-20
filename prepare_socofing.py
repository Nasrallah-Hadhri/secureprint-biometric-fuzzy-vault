#!/usr/bin/env python3
"""
prepare_socofing.py — Convertit le dataset SOCOFing au format FVC.

SOCOFing naming : {id}__{G}_{Hand}_{finger}_finger[_variant].BMP
FVC naming      : {fid}_{impression}.BMP

Stratégie :
  impression 1 → Real/{id}__{G}_{Hand}_{finger}_finger.BMP
  impression 2 → Altered-Easy/ ...finger_CR.BMP
  impression 3 → Altered-Easy/ ...finger_Obl.BMP
  impression 4 → Altered-Easy/ ...finger_Zcut.BMP

Usage :
  python prepare_socofing.py
  python prepare_socofing.py --src data/samples/SOCOFing --out data/fvc_ready --max-subjects 30
"""

import os
import re
import shutil
import argparse

SUFFIXES_ALT = {
    "CR":   2,
    "Obl":  3,
    "Zcut": 4,
}

FINGER_KEYS = [
    "Left_index", "Left_little", "Left_middle", "Left_ring", "Left_thumb",
    "Right_index", "Right_little", "Right_middle", "Right_ring", "Right_thumb",
]


def build_finger_map(socofing_root: str, max_subjects: int = 0):
    """
    Construit un dict :
      {(subject_id, finger_key): {"real": path, "alt_easy": {suffix: path}}}
    """
    real_dir  = os.path.join(socofing_root, "Real")
    easy_dir  = os.path.join(socofing_root, "Altered", "Altered-Easy")

    pattern_real = re.compile(
        r'^(\d+)__[MF]_(Left|Right)_(index|little|middle|ring|thumb)_finger\.BMP$',
        re.IGNORECASE
    )
    pattern_alt = re.compile(
        r'^(\d+)__[MF]_(Left|Right)_(index|little|middle|ring|thumb)_finger_(CR|Obl|Zcut)\.BMP$',
        re.IGNORECASE
    )

    entries = {}

    # ── Real ──────────────────────────────────────────────────────────────
    if not os.path.isdir(real_dir):
        raise FileNotFoundError(f"Dossier Real introuvable : {real_dir}")

    for fname in os.listdir(real_dir):
        m = pattern_real.match(fname)
        if not m:
            continue
        sid  = int(m.group(1))
        hand = m.group(2)
        fing = m.group(3)
        key  = (sid, f"{hand}_{fing}")
        entries.setdefault(key, {"real": None, "alt": {}})
        entries[key]["real"] = os.path.join(real_dir, fname)

    # ── Altered-Easy ──────────────────────────────────────────────────────
    if os.path.isdir(easy_dir):
        for fname in os.listdir(easy_dir):
            m = pattern_alt.match(fname)
            if not m:
                continue
            sid    = int(m.group(1))
            hand   = m.group(2)
            fing   = m.group(3)
            suffix = m.group(4)
            key    = (sid, f"{hand}_{fing}")
            entries.setdefault(key, {"real": None, "alt": {}})
            entries[key]["alt"][suffix] = os.path.join(easy_dir, fname)

    # ── Filtrer les entrées complètes ──────────────────────────────────────
    complete = {k: v for k, v in entries.items()
                if v["real"] and len(v["alt"]) >= 1}

    # ── Limiter les sujets ────────────────────────────────────────────────
    if max_subjects > 0:
        subjects = sorted({k[0] for k in complete.keys()})[:max_subjects]
        complete = {k: v for k, v in complete.items() if k[0] in subjects}

    return complete


def prepare(src: str, out: str, max_subjects: int = 0) -> str:
    """
    Copie les images SOCOFing dans `out/` avec la convention FVC :
      {fid}_{impression}.BMP
    Retourne le chemin du dossier de sortie.
    """
    os.makedirs(out, exist_ok=True)

    finger_map = build_finger_map(src, max_subjects)
    if not finger_map:
        raise ValueError("Aucune paire Real+Altered trouvée dans le dataset SOCOFing.")

    # Assigner un fid séquentiel à chaque clé
    sorted_keys = sorted(finger_map.keys())  # (sid, finger_key)
    fid_map = {k: i + 1 for i, k in enumerate(sorted_keys)}

    copied = skipped = 0
    for key, data in finger_map.items():
        fid = fid_map[key]
        ext = ".BMP"

        # Impression 1 : Real
        dst = os.path.join(out, f"{fid}_1{ext}")
        if not os.path.exists(dst):
            shutil.copy2(data["real"], dst)
        copied += 1

        # Impressions 2-4 : Altered variants
        for suffix, imp_num in SUFFIXES_ALT.items():
            src_path = data["alt"].get(suffix)
            if src_path:
                dst = os.path.join(out, f"{fid}_{imp_num}{ext}")
                if not os.path.exists(dst):
                    shutil.copy2(src_path, dst)
                copied += 1
            else:
                skipped += 1

    print(f"  ✓ {copied} fichiers copiés dans '{out}'  ({skipped} variants manquants ignorés)")
    print(f"  ✓ {len(sorted_keys)} identités (doigts uniques) — impressions 1..4 par identité")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Convertit SOCOFing au format FVC pour l'évaluation SecurePrint"
    )
    parser.add_argument(
        "--src", default="data/samples/SOCOFing",
        help="Racine du dataset SOCOFing (défaut: data/samples/SOCOFing)"
    )
    parser.add_argument(
        "--out", default="data/fvc_ready",
        help="Dossier de sortie au format FVC (défaut: data/fvc_ready)"
    )
    parser.add_argument(
        "--max-subjects", type=int, default=0,
        help="Limiter à N sujets (0 = tous, défaut: 0)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  SecurePrint — Préparation SOCOFing → FVC")
    print("=" * 55)
    print(f"\n  Source : {args.src}")
    print(f"  Sortie : {args.out}")
    if args.max_subjects:
        print(f"  Limite : {args.max_subjects} sujets\n")
    else:
        print("  Limite : tous les sujets\n")

    try:
        out_dir = prepare(args.src, args.out, args.max_subjects)
        n_files = len([f for f in os.listdir(out_dir)
                       if f.lower().endswith(".bmp")])
        print(f"\n  Total fichiers dans '{out_dir}' : {n_files}")
        print("\n  Prêt pour l'évaluation :")
        print(f"  python fvc2000_benchmark.py --db {out_dir} --plot results/eval_socofing.png --enroll-db")
    except Exception as e:
        print(f"\n  ERREUR : {e}")
        raise

    print("\n" + "=" * 55 + "\n")


if __name__ == "__main__":
    main()
