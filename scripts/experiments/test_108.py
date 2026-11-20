import os
import cv2
import numpy as np
from modules.matcher import full_pipeline, compare, minutiae_match_score

def test_finger_108():
    print("=" * 60)
    print("TEST DE ROBUSTESSE : DOIGT 108")
    print("=" * 60)
    
    base_dir = os.path.join("data", "fvc2000", "DB1_B")
    ref_image = os.path.join(base_dir, "108_1.tif")
    
    if not os.path.exists(ref_image):
        print(f"Erreur : Image {ref_image} introuvable.")
        return
        
    print(f"1. Enrôlement de l'empreinte de référence : 108_1.tif")
    ref_template, ref_minutiae = full_pipeline(ref_image)
    print(f"   ✓ Extraites : {len(ref_minutiae)} minuties.\n")
    
    print("2. Tests sur les variations du doigt 108 (Genuins)")
    print("-" * 60)
    for i in range(2, 9):
        img_path = os.path.join(base_dir, f"108_{i}.tif")
        if not os.path.exists(img_path): continue
        
        tpl, minu = full_pipeline(img_path)
        score, match = compare(ref_template, tpl, threshold=0.15)
        status = "ACCÈS AUTORISÉ" if match else "REFUSÉ"
        size = os.path.getsize(img_path)
        crop_info = "(Cropped/Partial)" if size < 90000 else ""
        print(f"   [108_{i}.tif] {crop_info:18} | Minuties: {len(minu):3} | Score: {score:.3f} -> {status}")
        
    print("\n3. Tests sur d'autres doigts (Imposteurs)")
    print("-" * 60)
    for fid in [101, 102, 105, 110]:
        img_path = os.path.join(base_dir, f"{fid}_1.tif")
        if not os.path.exists(img_path): continue
        
        tpl, minu = full_pipeline(img_path)
        score, match = compare(ref_template, tpl, threshold=0.15)
        status = "ACCÈS (FAUX)" if match else "REFUSÉ (Correct)"
        print(f"   [{fid}_1.tif]                     | Minuties: {len(minu):3} | Score: {score:.3f} -> {status}")

    print("\n4. Test de rotation extrême artificielle (108_1 tourné de 90°)")
    print("-" * 60)
    img = cv2.imread(ref_image, cv2.IMREAD_GRAYSCALE)
    rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rot_path = "108_1_rotated.tif"
    cv2.imwrite(rot_path, rotated)
    
    tpl, minu = full_pipeline(rot_path)
    score, match = compare(ref_template, tpl, threshold=0.15)
    status = "✅ ACCÈS AUTORISÉ" if match else "❌ REFUSÉ"
    print(f"   [108_1_rotated.tif] 90°         | Minuties: {len(minu):3} | Score: {score:.3f} -> {status}")
    os.remove(rot_path)
    
    print("\n5. Test de recadrage extrême artificiel (Moitié inférieure de 108_1)")
    print("-" * 60)
    h, w = img.shape
    cropped = img[h//2:, :] # Bottom half
    crop_path = "108_1_half.tif"
    cv2.imwrite(crop_path, cropped)
    
    tpl, minu = full_pipeline(crop_path)
    score, match = compare(ref_template, tpl, threshold=0.15)
    status = "✅ ACCÈS AUTORISÉ" if match else "❌ REFUSÉ"
    print(f"   [108_1_half.tif] Moitié Inf     | Minuties: {len(minu):3} | Score: {score:.3f} -> {status}")
    os.remove(crop_path)
    print("=" * 60)

if __name__ == "__main__":
    test_finger_108()
