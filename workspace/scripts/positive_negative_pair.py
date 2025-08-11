import os
import random
import shutil
from glob import glob

# === CONFIGURATION ===
input_folder = r"/workspace/processed_face_data"
output_positive_folder = r"/workspace/datasets/positive_pairs"
output_negative_folder = r"/workspace/datasets/negative_pairs"

# === CREATE OUTPUT FOLDERS ===
os.makedirs(output_positive_folder, exist_ok=True)
os.makedirs(output_negative_folder, exist_ok=True)

# === LOAD IMAGES RECURSIVELY AND GROUP BY IDENTITY ===
print(f"Looking for images in: {input_folder}")
images = sorted(
    glob(os.path.join(input_folder, "**", "*.jpg"), recursive=True) +
    glob(os.path.join(input_folder, "**", "*.jpeg"), recursive=True) +
    glob(os.path.join(input_folder, "**", "*.png"), recursive=True)
)

print(f"Found {len(images)} images.")
if len(images) > 0:
    print("Sample image paths:")
    for img in images[:5]:
        print(f" - {img}")

# === GROUP IMAGES BY IDENTITY (using folder name as identity) ===
identity_groups = {}
for img in images:
    # Use the parent folder name as the identity
    identity = os.path.basename(os.path.dirname(img))
    print(f"Image: {os.path.basename(img)} -> Identity: {identity}")
    identity_groups.setdefault(identity, []).append(img)

print(f"Detected {len(identity_groups)} unique identities.")

# === CREATE POSITIVE PAIRS ===
positive_pairs = []
for identity, imgs in identity_groups.items():
    if len(imgs) < 2:
        print(f"Skipping identity '{identity}' with only {len(imgs)} image(s).")
        continue
    for i in range(0, len(imgs) - 1, 2):
        positive_pairs.append((imgs[i], imgs[i + 1]))

print(f"Created {len(positive_pairs)} positive pairs.")

# === COPY POSITIVE PAIRS ===
for i, (img1, img2) in enumerate(positive_pairs):
    shutil.copy(img1, os.path.join(output_positive_folder, f"pair_{i:04d}_img1.jpg"))
    shutil.copy(img2, os.path.join(output_positive_folder, f"pair_{i:04d}_img2.jpg"))

# === CREATE NEGATIVE PAIRS ===
negative_pairs = []
identities = list(identity_groups.keys())
while len(negative_pairs) < len(positive_pairs):
    id1, id2 = random.sample(identities, 2)
    img1 = random.choice(identity_groups[id1])
    img2 = random.choice(identity_groups[id2])
    negative_pairs.append((img1, img2))

print(f"Created {len(negative_pairs)} negative pairs.")

# === COPY NEGATIVE PAIRS ===
for i, (img1, img2) in enumerate(negative_pairs):
    shutil.copy(img1, os.path.join(output_negative_folder, f"pair_{i:04d}_img1.jpg"))
    shutil.copy(img2, os.path.join(output_negative_folder, f"pair_{i:04d}_img2.jpg"))

print(f"Saved {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs.")
