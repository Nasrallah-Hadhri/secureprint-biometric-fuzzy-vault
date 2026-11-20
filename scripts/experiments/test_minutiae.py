import numpy as np

def minutiae_match_score(m1, m2, max_dist=15.0, max_angle=20.0):
    N = len(m1)
    M = len(m2)
    if N < 5 or M < 5: return 0.0

    t1 = m1[:, 2]
    t2 = m2[:, 2]
    valid_pairs = (t1[None, :] == t2[:, None])
    
    a1 = np.radians(m1[:, 3])
    a2 = np.radians(m2[:, 3])
    
    dtheta = a2[:, None] - a1[None, :]
    cos_t = np.cos(dtheta)
    sin_t = np.sin(dtheta)
    
    x1, y1 = m1[:, 0], m1[:, 1]
    x2, y2 = m2[:, 0], m2[:, 1]
    
    dx = x2[:, None] - (x1[None, :] * cos_t - y1[None, :] * sin_t)
    dy = y2[:, None] - (x1[None, :] * sin_t + y1[None, :] * cos_t)
    
    valid_j, valid_i = np.where(valid_pairs)
    if len(valid_j) == 0: return 0.0
    
    MAX_PAIRS = 4000
    if len(valid_j) > MAX_PAIRS:
        idx = np.random.choice(len(valid_j), MAX_PAIRS, replace=False)
        valid_j = valid_j[idx]
        valid_i = valid_i[idx]

    K = len(valid_j)
    cos_K = cos_t[valid_j, valid_i]
    sin_K = sin_t[valid_j, valid_i]
    dx_K = dx[valid_j, valid_i]
    dy_K = dy[valid_j, valid_i]
    
    x1_trans = x1[None, :] * cos_K[:, None] - y1[None, :] * sin_K[:, None] + dx_K[:, None]
    y1_trans = x1[None, :] * sin_K[:, None] + y1[None, :] * cos_K[:, None] + dy_K[:, None]
    a1_trans = (m1[None, :, 3] + np.degrees(dtheta[valid_j, valid_i])[:, None]) % 360
    
    max_score = 0
    x2_expanded = x2[None, None, :]
    y2_expanded = y2[None, None, :]
    a2_expanded = m2[:, 3][None, None, :]
    t2_expanded = t2[None, None, :]
    
    CHUNK_SIZE = 200
    for k_start in range(0, K, CHUNK_SIZE):
        k_end = min(k_start + CHUNK_SIZE, K)
        x1_c = x1_trans[k_start:k_end, :, None]
        y1_c = y1_trans[k_start:k_end, :, None]
        a1_c = a1_trans[k_start:k_end, :, None]
        t1_c = t1[None, :, None]
        
        dist_sq = (x1_c - x2_expanded)**2 + (y1_c - y2_expanded)**2
        ang_diff = np.abs(a1_c - a2_expanded)
        ang_diff = np.minimum(ang_diff, 360 - ang_diff)
        
        valid_match = (dist_sq <= max_dist**2) & (ang_diff <= max_angle) & (t1_c == t2_expanded)
        has_match = np.any(valid_match, axis=2)
        match_counts = np.sum(has_match, axis=1)
        
        chunk_max = np.max(match_counts)
        if chunk_max > max_score:
            max_score = chunk_max

    return float(max_score) / min(N, M)

if __name__ == "__main__":
    m1 = np.random.rand(150, 4) * 200
    m1[:, 2] = np.random.randint(0, 2, 150)
    m1[:, 3] *= 360/200
    m2 = m1.copy()
    
    # apply translation and rotation to m2
    theta = np.radians(45)
    c, s = np.cos(theta), np.sin(theta)
    x, y = m2[:, 0].copy(), m2[:, 1].copy()
    m2[:, 0] = x*c - y*s + 50
    m2[:, 1] = x*s + y*c - 30
    m2[:, 3] = (m2[:, 3] + 45) % 360
    
    # drop some points
    m2 = m2[20:]
    
    score = minutiae_match_score(m1, m2)
    print("Match score:", score)
