import torch
from PIL import Image
from torchvision import transforms
from siamese_model import SiameseNetwork
import torch.nn.functional as F

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load(r"/workspace/scripts/siamese_resnet50_model.pth", map_location=device))
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def infer_similarity(img1_path, img2_path, threshold=0.5):
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)

    with torch.no_grad():
        output1, output2 = model(img1, img2)
        distance = F.pairwise_distance(output1, output2).item()
        print(f"Euclidean Distance: {distance:.4f}")
        if distance < threshold:
            print
            print("Prediction: Similar")
        else:
            print("Prediction: Dissimilar")


# Example usage
infer_similarity(r"/workspace/test_images/Passport.jpeg", r"/workspace/test_images/FB03BB87F0F54FD08783BB129ED3EF7A.jpg")

