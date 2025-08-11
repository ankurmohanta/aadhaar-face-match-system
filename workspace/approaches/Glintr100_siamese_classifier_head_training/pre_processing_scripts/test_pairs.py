import os
import csv
def create_test_pairs(folder_path, output_csv):
   rows = []
   for person_dir in os.listdir(folder_path):
       person_path = os.path.join(folder_path, person_dir)
       if os.path.isdir(person_path):
           files = sorted([
               os.path.join(person_path, f)
               for f in os.listdir(person_path)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))
           ])
           if len(files) >= 2:
               rows.append([files[0], files[1], 1])  # label 1: same person
   # Write CSV
   with open(output_csv, 'w', newline='') as f:
       writer = csv.writer(f)
       writer.writerow(['img1', 'img2', 'label'])
       writer.writerows(rows)
   print(f"✅ Created {output_csv} with {len(rows)} positive pairs.")
# Example usage
if __name__ == "__main__":
   create_test_pairs(
       folder_path="/workspace/approaches/Glintr100_retrain_siamese_classifier/cropped_faces_15-19.06.25",
       output_csv= "/workspace/approaches/Glintr100_retrain_siamese_classifier/positive_15-19.06.25_testpairs.csv"
   )
   