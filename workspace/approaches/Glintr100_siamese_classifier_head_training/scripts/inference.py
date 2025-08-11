import torch
from embedding_extractor import Glintr100Embedder
from train import SiameseHead
def infer(img1_path, img2_path, model_path):
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   # Load embedder & model
   embedder = Glintr100Embedder("/workspace/approaches/InsightFace_Glintr100/models/glintr100/model.onnx")
   model = SiameseHead().to(device)
   model.load_state_dict(torch.load(model_path, map_location=device))
   model.eval()
   # Get embeddings
   emb1 = embedder.get_embedding(img1_path)
   emb2 = embedder.get_embedding(img2_path)
   emb1 = torch.tensor(emb1, dtype=torch.float32).unsqueeze(0).to(device)
   emb2 = torch.tensor(emb2, dtype=torch.float32).unsqueeze(0).to(device)
   # Predict similarity
   with torch.no_grad():
       score = model(emb1, emb2).item()
   print(f"🔍 Similarity Score: {score:.4f}")
   return score
# Example
if __name__ == "__main__":
   img1 = "/workspace/approaches/Glintr100_retrain_siamese_classifier/cropped_test_cases/4CD83696CB414BF0B7A6E66F0CEF7F4B/4CD83696CB414BF0B7A6E66F0CEF7F4B.jpg"
   img2 ="/workspace/approaches/Glintr100_retrain_siamese_classifier/cropped_test_cases/4CD83696CB414BF0B7A6E66F0CEF7F4B/6CCD305F5B784A1AAF2EB8977B9CAD46.jpg"
   infer(img1, img2, "/workspace/approaches/Glintr100_retrain_siamese_classifier/scripts/siamese_head.pth")