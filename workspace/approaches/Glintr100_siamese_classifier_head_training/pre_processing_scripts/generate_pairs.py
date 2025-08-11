import os
import random
import csv
from itertools import combinations
input_root = "/workspace/approaches/Glintr100_retrain_siamese_classifier/cropped_faces"
output_csv = "/workspace/approaches/Glintr100_retrain_siamese_classifier/pre_processing_scripts/pairs.csv"
num_negative_per_person = 2  # How many negative pairs to sample per person
pairs = []
# Get all person folders
persons = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
person_to_images = {}
# Load all image paths per person
for person in persons:
   folder = os.path.join(input_root, person)
   images = [os.path.join(folder, img) for img in os.listdir(folder) if img.lower().endswith((".jpg", ".png", ".jpeg"))]
   if len(images) >= 2:
       person_to_images[person] = images
# Generate positive pairs
for person, images in person_to_images.items():
   for img1, img2 in combinations(images, 2):
       pairs.append([img1, img2, 1])
# Generate negative pairs
person_list = list(person_to_images.keys())
for person, images in person_to_images.items():
   for img in images:
       sampled_negatives = 0
       while sampled_negatives < num_negative_per_person:
           other_person = random.choice(person_list)
           if other_person == person:
               continue
           other_images = person_to_images[other_person]
           neg_img = random.choice(other_images)
           pairs.append([img, neg_img, 0])
           sampled_negatives += 1
# Shuffle and save
random.shuffle(pairs)
with open(output_csv, "w", newline='') as f:
   writer = csv.writer(f)
   writer.writerow(["img1", "img2", "label"])
   writer.writerows(pairs)
print(f"✅ Pairs CSV saved to: {output_csv} ({len(pairs)} total)")