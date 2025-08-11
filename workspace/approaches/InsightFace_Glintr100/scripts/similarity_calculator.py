import numpy as np
def cosine_similarity(emb1, emb2):
   emb1_norm = emb1 / np.linalg.norm(emb1)
   emb2_norm = emb2 / np.linalg.norm(emb2)
   return np.dot(emb1_norm, emb2_norm)