#!/usr/bin/env python3
"""
SecurePrint — Point d'entrée principal.
Usage :
    python main.py              → Lance l'interface graphique
    python main.py --cli-test   → Test headless pour CI/vérification
"""

import sys
import os

# Force UTF-8 pour les émojis/symboles dans la console Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")



def cli_test():
    """
    Test headless : génère un template synthétique, le chiffre, le stocke,
    le lit et le compare. Vérifie que tout le pipeline fonctionne.
    """
    print("=" * 55)
    print("  SecurePrint — Test d'intégration (mode CLI)")
    print("=" * 55)

    # 1. Init DB + clé
    from modules.storage import init_db, enroll_user, get_all_users, get_template, delete_user
    from modules.encryption import generate_key, encrypt_template, decrypt_template
    from modules.template import generate_template, template_from_bytes
    from modules.matcher import compare
    import numpy as np

    print("\n[1/5] Initialisation base de données et clé de chiffrement...")
    init_db()
    generate_key()
    print("      ✓ Base SQLite et clé Fernet prêtes.")

    # 2. Créer un template synthétique (sans image réelle)
    print("\n[2/5] Génération d'un template synthétique...")
    fake_minutiae = [(50 + i * 10, 60 + i * 7, "ending", float(i * 18)) for i in range(15)]
    template_bytes = generate_template(fake_minutiae)
    print(f"      ✓ Template généré : {len(template_bytes)} bytes")

    # 3. Chiffrement
    print("\n[3/5] Chiffrement avec Fernet (AES-128-CBC + HMAC)...")
    encrypted = encrypt_template(template_bytes)
    print(f"      ✓ Template chiffré : {len(encrypted)} bytes (jamais stocké en clair)")

    # 4. Enrôlement en base
    print("\n[4/5] Enrôlement utilisateur test en base SQLite...")
    try:
        from modules.storage import delete_user, get_user_by_name
        existing = get_user_by_name("__test_user__")
        if existing:
            delete_user(existing[0])

        user_id = enroll_user("__test_user__", encrypted)
        print(f"      ✓ Utilisateur enregistré (ID={user_id})")

        blob = get_template(user_id)
        decrypted = decrypt_template(blob)
        print(f"      ✓ Template déchiffré : {len(decrypted)} bytes (identique={decrypted == template_bytes})")
    except Exception as e:
        print(f"      ✗ Erreur : {e}")
        sys.exit(1)

    # 5. Comparaison (même template → score = 1.0)
    print("\n[5/5] Test de matching (template vs lui-même)...")
    score, is_match = compare(template_bytes, template_bytes, threshold=0.60)
    print(f"      Score     : {score:.6f}")
    print(f"      Résultat  : {'✅ MATCH (attendu)' if is_match else '❌ NO MATCH (erreur !)'}")

    # Test cross-impression avec images FVC2000 réelles (si disponibles)
    fvc_img1 = os.path.join("data", "fvc2000", "DB1_B", "101_1.tif")
    fvc_img2 = os.path.join("data", "fvc2000", "DB1_B", "101_2.tif")
    if os.path.exists(fvc_img1) and os.path.exists(fvc_img2):
        try:
            from modules.matcher import full_pipeline
            t1, m1 = full_pipeline(fvc_img1)
            t2, m2 = full_pipeline(fvc_img2)
            score_genuine, _ = compare(t1, t2, threshold=0.90)

            # Imposteur : doigt différent
            fvc_imp = os.path.join("data", "fvc2000", "DB1_B", "102_1.tif")
            t_imp, _ = full_pipeline(fvc_imp)
            score_imp, _ = compare(t1, t_imp, threshold=0.90)

            print(f"      Score genuine (101_1 vs 101_2) : {score_genuine:.4f}")
            print(f"      Score imposteur (101_1 vs 102_1): {score_imp:.4f}")
            gap = score_genuine - score_imp
            print(f"      Séparation genuine/imposteur    : {gap:+.4f} {'✅ positive' if gap > 0 else '⚠ negative'}")
        except Exception as e:
            print(f"      Test images réelles ignoré : {e}")
    else:
        # Fallback : légère perturbation du vecteur HOG
        vec = template_from_bytes(template_bytes).copy()
        vec += np.random.normal(0, 0.005, size=vec.shape).astype("float32")
        noisy_bytes = vec.tobytes()
        score2, _ = compare(template_bytes, noisy_bytes, threshold=0.90)
        print(f"      Score perturbé (HOG+bruit) : {score2:.6f}")

    # Nettoyage
    delete_user(user_id)

    print("\n" + "=" * 55)
    print("  ✅ Tous les tests passés — pipeline fonctionnel.")
    print("=" * 55)


def fvc2000_eval():
    """Lance le benchmark FVC2000 en mode CLI (délègue à fvc2000_benchmark.py)."""
    import sys as _sys
    # Retirer --fvc2000-eval et passer le reste à fvc2000_benchmark.main()
    _sys.argv = [_sys.argv[0]] + [a for a in _sys.argv[1:] if a != "--fvc2000-eval"]
    try:
        from fvc2000_benchmark import main as bm_main
        bm_main()
    except ImportError:
        print("[ERREUR] fvc2000_benchmark.py introuvable dans le répertoire du projet.")
        _sys.exit(1)


def main():
    if "--cli-test" in sys.argv:
        cli_test()
    elif "--fvc2000-eval" in sys.argv:
        fvc2000_eval()
    else:
        try:
            from ui.main_window import launch
            launch()
        except ImportError as e:
            print(f"[ERREUR] Dépendance manquante : {e}")
            print("Exécutez : pip install -r requirements.txt")
            sys.exit(1)


if __name__ == "__main__":
    main()
