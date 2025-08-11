import os
import cv2
from retinaface import RetinaFace
# Input/output paths
input_root =  r"/workspace/approaches/Glintr100_retrain_siamese_classifier/last_2months_raw_data"
output_root = r"/workspace/approaches/Glintr100_retrain_siamese_classifier/cropped_last_2months"
os.makedirs(output_root, exist_ok=True)
def crop_face(img_path, save_path):
   try:
       if os.path.exists(save_path):
           print(f"⚠️ Skipping (already exists): {save_path}")
           return
       faces = RetinaFace.extract_faces(img_path=img_path, align=True)
       if not faces:
           print(f"❌ No face in: {img_path}")
           return
       face = faces[0]
       face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
       cv2.imwrite(save_path, face_bgr)
       print(f"✅ Saved: {save_path}")
   except Exception as e:
       print(f"❌ Error in {img_path}: {e}")
# Loop over each subfolder/person
for folder in os.listdir(input_root):
   input_folder = os.path.join(input_root, folder)
   if not os.path.isdir(input_folder):
       continue
   output_folder = os.path.join(output_root, folder)
   os.makedirs(output_folder, exist_ok=True)
   for file in os.listdir(input_folder):
       if not file.lower().endswith((".jpg", ".jpeg", ".png")):
           continue
       img_path = os.path.join(input_folder, file)
       save_path = os.path.join(output_folder, file)
       crop_face(img_path, save_path)