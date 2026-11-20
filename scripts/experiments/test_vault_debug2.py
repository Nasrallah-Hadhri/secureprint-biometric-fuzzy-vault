import os
import json
from modules.preprocessor import preprocess
from modules.alignment import align_skeleton
from modules.minutiae import extract_minutiae
from modules.fuzzy_vault import extract_features, DEGREE, PRIME, _key_to_coeffs, _lagrange_interpolation

base_dir = os.path.join("data", "fvc2000", "DB1_B")

def test():
    ref_image = os.path.join(base_dir, "108_1.tif")
    cand_impostor = os.path.join(base_dir, "101_1.tif")
    
    from modules.matcher import full_pipeline
    vault_bytes, _ = full_pipeline(ref_image)
    vault = json.loads(vault_bytes)
    
    skel = preprocess(cand_impostor)
    aligned = align_skeleton(skel)
    minutiae = extract_minutiae(aligned)
    cand_ints = extract_features(minutiae)
    
    all_points = vault["all_points"]
    points_dict = {x: y for x, y in all_points}
    degree = vault.get("degree", DEGREE)

    matched = []
    for x in cand_ints:
        x_mod = x % PRIME
        if x_mod in points_dict:
            matched.append((x_mod, points_dict[x_mod]))

    print(f"Matched points: {len(matched)}")
    
    if len(matched) < degree + 1:
        print("Not enough matches.")
        return
        
    expected_key = bytes.fromhex(vault["_expected_key"])
    expected_p0 = _key_to_coeffs(expected_key, degree)[0]
    print(f"Expected P0: {expected_p0}")
    
    import itertools
    max_combos = 500
    combos = itertools.islice(itertools.combinations(matched, degree + 1), max_combos)
    
    success = False
    for subset in combos:
        try:
            p0 = _lagrange_interpolation(list(subset))
            if p0 == expected_p0:
                print("SUCCESSFUL INTERPOLATION FOUND!")
                success = True
                break
        except Exception as e:
            continue
            
    print(f"Success: {success}")

if __name__ == "__main__":
    test()
