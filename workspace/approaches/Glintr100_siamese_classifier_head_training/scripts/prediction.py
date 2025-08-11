import torch
import pandas as pd
from embedding_extractor import Glintr100Embedder
from train import SiameseHead
def generate_scores(input_csv, output_csv, model_path, onnx_path):
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = SiameseHead().to(device)
   model.load_state_dict(torch.load(model_path, map_location=device))
   model.eval()
   embedder = Glintr100Embedder(onnx_path)
   df = pd.read_csv(input_csv)
   score_list = []
   for _, row in df.iterrows():
       img1_path = row['img1']
       img2_path = row['img2']
       label = int(row['label'])
       emb1 = embedder.get_embedding(img1_path)
       emb2 = embedder.get_embedding(img2_path)
       emb1 = torch.tensor(emb1, dtype=torch.float32).unsqueeze(0).to(device)
       emb2 = torch.tensor(emb2, dtype=torch.float32).unsqueeze(0).to(device)
       with torch.no_grad():
           score = model(emb1, emb2).item()
       score_list.append([img1_path, img2_path, label, round(score, 4)])
   # Save to CSV
   scored_df = pd.DataFrame(score_list, columns=['img1', 'img2', 'label', 'score'])
   scored_df.to_csv(output_csv, index=False)
   print(f"✅ Scores written to: {output_csv}")
if __name__ == "__main__":
   generate_scores(
       input_csv= "/workspace/approaches/Glintr100_retrain_siamese_classifier/negative_15-19.06.25_testpairs.csv",
       output_csv="/workspace/approaches/Glintr100_retrain_siamese_classifier/neg_pred_15-19.06.25_testpairs.csv",
       model_path="/workspace/approaches/Glintr100_retrain_siamese_classifier/scripts/siamese_head_v2.pth",
       onnx_path= "/workspace/approaches/InsightFace_Glintr100/models/glintr100/model.onnx"
   )

