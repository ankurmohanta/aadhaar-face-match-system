import csv
import os
import requests
import json
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

def get_jwt_token():
    url = "https://accounts.godigit.com/auth/realms/ABS-21/protocol/openid-connect/token"
    payload = 'grant_type=client_credentials&client_id=digitai&client_secret=ac12aaa6-724b-430d-b0bd-8a1ad9b3370f'
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    response = requests.post(url, headers=headers, data=payload)
    return response.json()['access_token']

def convert_pdf_bytes_to_jpg(stream):
    doc = fitz.open(stream=stream, filetype="pdf")
    images = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(np.array(img))
    return images

def read_images(contents):
    np_array = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    return img_bgr

def dms_document_download(reference_id, token):
    url = "https://prod-life-dms.godigit.com/DigitDMS/v1/document"
    payload = json.dumps({
        "documentId": reference_id,
        "documentCode": None,
        "companyCode": "DL"
    })
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.post(url, headers=headers, data=payload)

    if response.status_code != 200:
        print(f"❌ Failed to download {reference_id}: {response.status_code}")
        return []

    stream = response.content
    content_disp = response.headers.get('Content-Disposition', '')
    ext = content_disp.split('.')[-1].strip('"') if '.' in content_disp else 'jpg'
    print(f"📄 {reference_id}: File extension detected - {ext}")

    if ext.lower() == 'pdf':
        images = convert_pdf_bytes_to_jpg(stream)
        if not images:
            print(f"⚠️ {reference_id}: No images extracted from PDF")
        return images
    else:
        img = read_images(stream)
        if img is None:
            print(f"⚠️ {reference_id}: Failed to decode image")
            return []
        return [img]

def save_images(images, reference_id, folder):
    os.makedirs(folder, exist_ok=True)
    for i, img in enumerate(images):
        filename = f"{reference_id}_{i}.jpg"
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            print(f"⏭️ Skipping existing file: {path}")
            continue
        success = cv2.imwrite(path, img)
        if not success:
            print(f"❌ Failed to save image {path}")
        else:
            print(f"✅ Saved image: {path}")

def download_documents_from_csv(csv_path, output_folder, max_folders=200):
    token = get_jwt_token()
    print(f"🔍 Opening CSV file: {csv_path}")
    with open(csv_path, newline='') as csvfile:
        reader = list(csv.DictReader(csvfile))  # Convert to list for tqdm
        print(f"📋 Detected headers: {reader[0].keys() if reader else 'None'}")

        folders_created = 0

        for idx, row in enumerate(tqdm(reader, desc="📦 Downloading documents", unit="row")):
            if folders_created >= max_folders:
                print(f"🚫 Reached folder creation limit of {max_folders}. Stopping.")
                break

            print(f"\n🔄 Processing row {idx + 1}: {row}")
            try:
                image_1 = row.get('image_1')
                image_2 = row.get('image_2')

                # Fallback for combined column
                if not image_1 or not image_2:
                    combined = row.get('image_1,"image_2"') or row.get('?column?', '')
                    parts = [p.strip().strip('"') for p in combined.split(',')]
                    if len(parts) >= 2:
                        image_1, image_2 = parts[0], parts[1]
                    elif len(parts) == 1:
                        image_1, image_2 = parts[0], None
                    else:
                        print(f"⚠️ Skipping row with insufficient image IDs: {row}")
                        continue

                images_downloaded = False
                subfolder = os.path.join(output_folder, image_1)

                for ref_id in [image_1, image_2]:
                    if ref_id:
                        images = dms_document_download(ref_id, token)
                        print(f"🖼️ {ref_id}: {len(images)} image(s)")
                        if images:
                            if not images_downloaded:
                                os.makedirs(subfolder, exist_ok=True)
                                images_downloaded = True
                                folders_created += 1
                            save_images(images, ref_id, subfolder)

                if not images_downloaded:
                    print(f"⚠️ No images downloaded for row {idx + 1}, skipping folder creation.")
            except Exception as e:
                print(f"❗ Error processing row {idx + 1}: {e}")

# Example usage
download_documents_from_csv(
    r"/workspace/approaches/Glintr100_retrain_siamese_classifier/Face_Match_Data_18_06.csv",
    r"/workspace/approaches/Glintr100_retrain_siamese_classifier/raw_test_data"
)
