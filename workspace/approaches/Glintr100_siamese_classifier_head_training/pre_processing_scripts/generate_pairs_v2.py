import os
import pandas as pd
import random
from glob import glob
from itertools import combinations
# Set this to your uploaded folder (e.g., cropped_faces_v1 or last_2months_raw_data)
base_dir = "/workspace/approaches/Glintr100_retrain_siamese_classifier/cropped_faces_v2"
positive_pairs = []
negative_pairs = []
# Step 1: Get all person folders
person_folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir)
                 if os.path.isdir(os.path.join(base_dir, d))]
# Step 2: Generate all positive pairs from same person folder
for folder in person_folders:
   images = [img for img in glob(os.path.join(folder, '*'))
             if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
   if len(images) < 2:
       continue
   for img1, img2 in combinations(images, 2):
       positive_pairs.append([img1, img2, 1])
# Step 3: Generate negative pairs by pairing from different folders
num_negatives_required = int(len(positive_pairs) * 0.9)
while len(negative_pairs) < num_negatives_required:
   folder1, folder2 = random.sample(person_folders, 2)
   imgs1 = glob(os.path.join(folder1, '*'))
   imgs2 = glob(os.path.join(folder2, '*'))
   imgs1 = [img for img in imgs1 if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
   imgs2 = [img for img in imgs2 if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
   if imgs1 and imgs2:
       img1 = random.choice(imgs1)
       img2 = random.choice(imgs2)
       negative_pairs.append([img1, img2, 0])
# Step 4: Combine and shuffle
all_pairs = positive_pairs + negative_pairs
random.shuffle(all_pairs)
# Step 5: Save to CSV
df = pd.DataFrame(all_pairs, columns=["img1", "img2", "label"])
df.to_csv("pairs_v2.csv", index=False, header=False)
print(f"✅ Saved {len(df)} training pairs to final_pairs_augmented.csv")
