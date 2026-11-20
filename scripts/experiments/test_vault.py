import os
from modules.matcher import full_pipeline, authenticate

base_dir = os.path.join("data", "fvc2000", "DB1_B")

def test():
    ref_image = os.path.join(base_dir, "108_1.tif")
    cand_genuine = os.path.join(base_dir, "108_2.tif")
    cand_rotated = "108_1_rotated.tif"
    cand_impostor = os.path.join(base_dir, "101_1.tif")
    
    print("1. Enrolling 108_1")
    vault_bytes, _ = full_pipeline(ref_image)
    
    # Needs encryption to simulate the full pipeline since matcher.authenticate calls decrypt
    from modules.encryption import encrypt_template
    enc_vault = encrypt_template(vault_bytes)
    
    print("\n2. Testing Genuine (108_2)")
    score, match = authenticate(cand_genuine, enc_vault)
    print(f"Result: {match} (Score: {score:.3f})")
    
    print("\n3. Testing Impostor (101_1)")
    score, match = authenticate(cand_impostor, enc_vault)
    print(f"Result: {match} (Score: {score:.3f})")

if __name__ == "__main__":
    test()
