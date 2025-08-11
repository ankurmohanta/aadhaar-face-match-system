import torch
from torch.utils.data import Dataset
import pandas as pd
from embedding_extractor import Glintr100Embedder
class SiameseDataset(Dataset):
   def __init__(self, csv_file, embedder):
       self.data = pd.read_csv(csv_file)
       self.embedder = embedder
   def __len__(self):
       return len(self.data)
   def __getitem__(self, idx):
       row = self.data.iloc[idx]
       img1_path = row['img1']
       img2_path = row['img2']
       label = int(row['label'])
       emb1 = self.embedder.get_embedding(img1_path)
       emb2 = self.embedder.get_embedding(img2_path)
       return (
           torch.tensor(emb1, dtype=torch.float32),
           torch.tensor(emb2, dtype=torch.float32),
           torch.tensor(label, dtype=torch.float32)
       )