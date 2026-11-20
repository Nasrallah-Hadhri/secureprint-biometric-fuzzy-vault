import os
from modules.matcher import full_pipeline, authenticate
from modules.preprocessor import preprocess
from modules.alignment import align_skeleton
from modules.minutiae import extract_minutiae
from modules.fuzzy_vault import extract_features, DEGREE

base_dir = os.path.join("data", "fvc2000", "DB1_B")

def test():
    ref_image = os.path.join(base_dir, "108_1.tif")
    cand_impostor = os.path.join(base_dir, "101_1.tif")
    
    print("1. Enrolling 108_1")
    vault_bytes, _ = full_pipeline(ref_image)
    
    import json
    vault = json.loads(vault_bytes)
    print(f"Vault points: {len(vault['all_points'])}")
    print(f"Vault genuine: {vault['n_genuine']}")
    
    skel = preprocess(cand_impostor)
    aligned = align_skeleton(skel)
    minutiae = extract_minutiae(aligned)
    cand_ints = extract_features(minutiae)
    
    points_dict = {x: y for x, y in vault["all_points"]}
    matched = [x for x in cand_ints if (x % 65537) in points_dict]
    print(f"Impostor matched points: {len(matched)}")
    print(f"Degree required: {DEGREE + 1}")

if __name__ == "__main__":
    test()
