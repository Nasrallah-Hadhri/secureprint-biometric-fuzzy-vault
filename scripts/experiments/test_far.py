import os
from modules.matcher import authenticate, generate_rotated_candidates, compare
from modules.preprocessor import preprocess
from modules.template import generate_template

base_dir = os.path.join("data", "fvc2000", "DB1_B")

def test_impostor():
    ref_image = os.path.join(base_dir, "108_1.tif")
    cand_image = os.path.join(base_dir, "101_1.tif")
    
    # Generate enrolled template
    ref_skel = preprocess(ref_image)
    ref_template = generate_template(ref_skel)
    
    # Generate candidate templates
    cand_skel = preprocess(cand_image)
    candidates = generate_rotated_candidates(cand_skel)
    
    print("Testing 101_1 against 108_1:")
    max_score = 0
    for i, cand in enumerate(candidates):
        score, _ = compare(ref_template, cand, threshold=0.40)
        print(f"  Rotation {i} score: {score:.4f}")
        if score > max_score: max_score = score
        
    print(f"Max score: {max_score:.4f}")

if __name__ == "__main__":
    test_impostor()
