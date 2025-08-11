import os
import cv2
from mtcnn import MTCNN
from tqdm import tqdm

def detect_and_save_faces(input_folder, output_folder):
    detector = MTCNN()

    for root, _, files in os.walk(input_folder):
        for file in tqdm(files, desc="Processing images"):
            if file.lower().endswith('.jpg'):
                input_path = os.path.join(root, file)

                # Determine relative path and create corresponding output path
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)

                # Skip if already processed
                base_name = os.path.splitext(file)[0]
                already_processed = any(
                    fname.startswith(base_name + "_face") and fname.endswith(".jpg")
                    for fname in os.listdir(output_subfolder)
                )
                if already_processed:
                    print(f"⏭️ Skipping already processed image: {file}")
                    continue

                # Read image
                img = cv2.imread(input_path)
                if img is None:
                    print(f"❌ Failed to read image {input_path}")
                    continue

                # Detect faces
                detections = detector.detect_faces(img)
                if not detections:
                    print(f"⚠️ No faces detected in {input_path}")
                    continue

                # Crop and save faces
                for i, detection in enumerate(detections):
                    x, y, width, height = detection['box']
                    cropped_face = img[y:y+height, x:x+width]
                    output_filename = f"{base_name}_face{i}.jpg"
                    output_path = os.path.join(output_subfolder, output_filename)
                    success = cv2.imwrite(output_path, cropped_face)
                    if success:
                        print(f"✅ Saved face {i} to {output_path}")
                    else:
                        print(f"❌ Failed to save face {i} from {file}")

# Example usage
input_folder =  r"/workspace/raw_data"
output_folder = r"/workspace/processed_face_data_1"
detect_and_save_faces(input_folder, output_folder)
