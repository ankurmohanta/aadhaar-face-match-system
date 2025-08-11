import os
from preprocessing import preprocess_image
from embedding_extractor import Glintr100Embedder
from similarity_calculator import cosine_similarity
# Paths
model_path = r"/workspace/approaches/InsightFace_Glintr100/models/glintr100/model.onnx"
img1_path =  r"/workspace/test_images/Image (16).jpg"
img2_path =  r"/workspace/test_images/Image (17).jpg"
# Load model
extractor = Glintr100Embedder(model_path)
# Preprocess both images
img1 = preprocess_image(img1_path)
img2 = preprocess_image(img2_path)
# Extract embeddings
emb1 = extractor.get_embedding(img1)
emb2 = extractor.get_embedding(img2)
# Calculate similarity
similarity_score = cosine_similarity(emb1, emb2)
print(f"Cosine Similarity: {similarity_score:.4f}")