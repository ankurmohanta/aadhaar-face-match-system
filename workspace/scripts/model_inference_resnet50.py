import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import warnings
warnings.filterwarnings("ignore")


# === Siamese Network Definition ===
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = models.resnet50(pretrained=False)
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

# === Image Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === Load Model ===
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(r"/workspace/scripts/siamese_resnet50_model_v2.pth", map_location=device))
model.eval()

# === Inference Function ===
def infer_similarity(img1_path, img2_path):
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, similarity = model(img1, img2)
        similarity_score = torch.sigmoid(similarity).item()

    print(f"Similarity Score: {similarity_score:.4f}")

    if similarity_score > 0.7:
        print("Images are likely similar.")
    elif similarity_score < 0.3:
        print("Images are likely different.")
    else:
        print("Images are moderately similar.")
    return similarity_score

# === Example Usage ===
if __name__ == "__main__":
    img1_path = r"/workspace/test_images/Passport.jpeg"
    img2_path = r"/workspace/test_images/FB03BB87F0F54FD08783BB129ED3EF7A.jpg"
    infer_similarity(img1_path, img2_path)
