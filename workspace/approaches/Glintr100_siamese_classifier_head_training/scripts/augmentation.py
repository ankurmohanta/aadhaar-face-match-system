import os
import cv2
import random
from glob import glob
from albumentations import Compose, Rotate, HorizontalFlip, Blur, GaussNoise, RandomBrightnessContrast
# Update this to your local path
base_dir = "/workspace/approaches/Glintr100_retrain_siamese_classifier/cropped_last_2months"
transform = Compose([
   Rotate(limit=10, p=0.5),
   HorizontalFlip(p=0.5),
   Blur(blur_limit=3, p=0.3),
   GaussNoise(var_limit=(5.0, 30.0), p=0.4),
   RandomBrightnessContrast(p=0.3)
])
augmented_count = 0
for person_folder in os.listdir(base_dir):
   person_path = os.path.join(base_dir, person_folder)
   if not os.path.isdir(person_path):
       continue
   image_paths = sorted(glob(os.path.join(person_path, "*.*")))
   image_paths = [img for img in image_paths if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
   if len(image_paths) < 2:
       continue
   candidate_to_augment = image_paths[1]  # Augment doc image
   try:
       image = cv2.imread(candidate_to_augment)
       if image is None:
           continue
       augmented = transform(image=image)['image']
       filename = os.path.splitext(os.path.basename(candidate_to_augment))[0]
       save_path = os.path.join(person_path, f"{filename}_aug.jpg")
       cv2.imwrite(save_path, augmented)
       augmented_count += 1
   except Exception as e:
       print(f"Error in {person_path}: {e}")
print(f"✅ Augmented {augmented_count} images.")