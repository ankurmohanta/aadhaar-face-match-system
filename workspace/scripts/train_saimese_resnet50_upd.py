import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms, models
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# === CUDA Check ===
if torch.cuda.is_available():
    num_devices = torch.cuda.device_count()
    print(f"Number of CUDA devices available: {num_devices}")
    for i in range(num_devices):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available on this system.")

# === Dataset Class ===
class SiamesePairsDataset(Dataset):
    def __init__(self, folder_path, label, transform=None, debug=False):
        self.folder_path = folder_path
        self.label = label
        self.transform = transform
        self.debug = debug
        self.pairs = sorted([f for f in os.listdir(folder_path) if f.endswith("_img1.jpg")])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_name = self.pairs[idx]
        img2_name = img1_name.replace("_img1.jpg", "_img2.jpg")

        img1_path = os.path.join(self.folder_path, img1_name)
        img2_path = os.path.join(self.folder_path, img2_name)

        if self.debug and idx < 5:
            print(f"[DEBUG] Loading pair: {img1_path}, {img2_path} | Label: {self.label}")

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(self.label, dtype=torch.float32)

# === Siamese Network ===
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, 256)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward_once(self, x):
        return self.cnn(x)

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2, self.fc(torch.abs(output1 - output2))

# === Contrastive Loss ===
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# === Main Training Script ===
def main():
    pos_path = r"/workspace/datasets/positive_pairs/"
    neg_path = r"/workspace/datasets/negative_pairs/"
    debug_mode = True

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pos_dataset = SiamesePairsDataset(pos_path, label=1, transform=transform, debug=debug_mode)
    neg_dataset = SiamesePairsDataset(neg_path, label=0, transform=transform, debug=debug_mode)

    print("\n[INFO] Sample image pairs from positive and negative datasets:")
    for i in range(3):
        _, _, label = pos_dataset[i]
        print(f"Positive Pair {i}: Label={label.item()}")
    for i in range(3):
        _, _, label = neg_dataset[i]
        print(f"Negative Pair {i}: Label={label.item()}")

    full_dataset = ConcatDataset([pos_dataset, neg_dataset])
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    num_epochs = 100
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        all_labels = []
        all_preds = []

        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            output1, output2, similarity = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            distances = nn.functional.pairwise_distance(output1, output2)
            pred = (distances < 1.0).float()
            all_preds.extend(pred.detach().cpu().numpy())
            all_labels.extend(label.cpu().numpy())

        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        scheduler.step(total_loss)

    # === Validation ===
    model.eval()
    val_labels = []
    val_preds = []
    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output1, output2, similarity = model(img1, img2)
            distances = nn.functional.pairwise_distance(output1, output2)
            val_preds.extend(distances.cpu().numpy())
            val_labels.extend(label.cpu().numpy())

    precisions, recalls, thresholds = precision_recall_curve(val_labels, val_preds)
    plt.plot(recalls, precisions, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.show()

    roc_auc = roc_auc_score(val_labels, val_preds)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # === t-SNE Visualization ===
    embeddings = []
    labels = []
    with torch.no_grad():
        for img1, img2, label in val_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            output1, output2, _ = model(img1, img2)
            embeddings.append(output1.cpu().numpy())
            embeddings.append(output2.cpu().numpy())
            labels.extend(label.cpu().numpy())
            labels.extend(label.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title('t-SNE Embedding Space')
    plt.show()

    torch.save(model.state_dict(), "siamese_resnet50_model_v2.pth")
    print("Model saved as siamese_resnet50_model_v2.pth")

if __name__ == "__main__":
    main()
