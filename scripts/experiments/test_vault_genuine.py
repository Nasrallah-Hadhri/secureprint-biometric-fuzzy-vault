import os
from modules.matcher import full_pipeline, authenticate
from modules.preprocessor import preprocess
from modules.alignment import align_skeleton
from modules.minutiae import extract_minutiae
from modules.fuzzy_vault import extract_features, DEGREE

base_dir = os.path.join("data", "fvc2000", "DB1_B")

def test():
    ref_image = os.path.join(base_dir, "108_1.tif")
    cand_genuine = os.path.join(base_dir, "108_2.tif")
    
    vault_bytes, _ = full_pipeline(ref_image)
    
    import json
    vault = json.loads(vault_bytes)
    
    skel = preprocess(cand_genuine)
    aligned = align_skeleton(skel)
    minutiae = extract_minutiae(aligned)
    cand_ints = extract_features(minutiae)
    
    points_dict = {x: y for x, y in vault["all_points"]}
    matched = [x for x in cand_ints if (x % 65537) in points_dict]
    print(f"Genuine matched points: {len(matched)}")

if __name__ == "__main__":
    test()
