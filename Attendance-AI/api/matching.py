import numpy as np, time
from typing import List, Tuple, Dict
from collections import defaultdict

# cache: {employee_id: [embeddings...]}
GALLERY: Dict[str, List[np.ndarray]] = defaultdict(list)
LAST_HIT: Dict[str, float] = {}  # chống lặp (theo employee_id)

def load_gallery_from_db(session, FaceTemplate):
    GALLERY.clear()
    for row in session.query(FaceTemplate).all():
        emb = np.frombuffer(row.embedding, dtype=np.float32)
        GALLERY[row.employee_id].append(emb / (np.linalg.norm(emb) + 1e-9))

def match_faces(face_embeddings: List[np.ndarray], thr: float, cooldown_sec: int):
    results = []
    now = time.time()
    # Chuẩn hoá gallery
    keys = list(GALLERY.keys())
    if not keys: return results
    gallery = []
    owners = []
    for k in keys:
        for e in GALLERY[k]:
            gallery.append(e)
            owners.append(k)
    G = np.stack(gallery, axis=0)  # [M, D]

    for f in face_embeddings:
        f = f.astype(np.float32)
        f = f / (np.linalg.norm(f) + 1e-9)
        # cosine distance ~ 1 - cosine_similarity
        sims = G @ f                      # [M]
        # chuyển về "khoảng cách": d = sqrt(2*(1-sim)) hay dùng 1 - sim
        dists = 1.0 - sims
        j = int(np.argmin(dists))
        d = float(dists[j]); emp = owners[j]
        if d <= thr:
            # kiểm tra cooldown
            if emp in LAST_HIT and now - LAST_HIT[emp] < cooldown_sec:
                continue
            LAST_HIT[emp] = now
            results.append((emp, d))
    return results
