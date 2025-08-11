import torch

import torch.nn as nn

from torch.utils.data import DataLoader

from data_loader import SiameseDataset

from embedding_extractor import Glintr100Embedder

# Model: simple 2-layer MLP on embedding difference

class SiameseHead(nn.Module):

    def __init__(self):

        super(SiameseHead, self).__init__()

        self.classifier = nn.Sequential(

            nn.Linear(512, 256),

            nn.ReLU(),

            nn.Linear(256, 1),

            nn.Sigmoid()

        )

    def forward(self, emb1, emb2):

        diff = torch.abs(emb1 - emb2)  # element-wise absolute difference

        return self.classifier(diff)

# Training loop

def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedder = Glintr100Embedder("/workspace/approaches/InsightFace_Glintr100/models/glintr100/model.onnx")

    dataset = SiameseDataset("/workspace/approaches/Glintr100_retrain_siamese_classifier/pairs_v2.csv", embedder)

    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = SiameseHead().to(device)

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()

    for epoch in range(100):

        total_loss = 0

        for emb1, emb2, label in loader:

            emb1, emb2, label = emb1.to(device), emb2.to(device), label.to(device).unsqueeze(1)

            output = model(emb1, emb2)

            loss = criterion(output, label)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

    # Save trained model

    torch.save(model.state_dict(), "siamese_head_v2.pth")

    print("✅ Model saved: siamese_head.pth")

if __name__ == "__main__":

    train()
 