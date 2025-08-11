import os

import csv

import random

from itertools import combinations

input_root = "/workspace/approaches/Glintr100_retrain_siamese_classifier/cropped_faces_v1"

output_csv = "/workspace/approaches/Glintr100_retrain_siamese_classifier/pairs_v1.csv"

pairs = []

positive_pairs = []

negative_candidates = []

# Step 1: Get all folders (people)

persons = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]

# Step 2: Map person to their image paths

person_to_images = {}

for person in persons:

    folder = os.path.join(input_root, person)

    images = [os.path.join(folder, img) for img in os.listdir(folder) if img.lower().endswith((".jpg", ".jpeg", ".png"))]

    if len(images) >= 2:

        person_to_images[person] = images

# Step 3: Generate positive pairs

for person, images in person_to_images.items():

    for img1, img2 in combinations(images, 2):

        positive_pairs.append((img1, img2, 1))

# Step 4: Generate all candidate negative pairs

person_list = list(person_to_images.keys())

for i, person1 in enumerate(person_list):

    for person2 in person_list[i+1:]:

        for img1 in person_to_images[person1]:

            for img2 in person_to_images[person2]:

                negative_candidates.append((img1, img2, 0))

# Step 5: Balance the pairs

random.shuffle(negative_candidates)

negative_pairs = negative_candidates[:len(positive_pairs)]

# Step 6: Combine, shuffle, save

pairs = positive_pairs + negative_pairs

random.shuffle(pairs)

with open(output_csv, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow(["img1", "img2", "label"])

    writer.writerows(pairs)

print(f"✅ Generated {len(positive_pairs)} positive and {len(negative_pairs)} negative pairs.")

print(f"📄 Saved to {output_csv}")
 