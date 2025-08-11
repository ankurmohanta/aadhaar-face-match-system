import csv
import os
import requests
import json
import fitz  # This is correct *only* if PyMuPDF is installed
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
    payload = json.dumps({"documentId": reference_id, "documentCode": None, "companyCode": "DL"})
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
        path = os.path.join(folder, f"{reference_id}_{i}.jpg")
        success = cv2.imwrite(path, img)
        if not success:
            print(f"❌ Failed to save image {path}")
        else:
            print(f"✅ Saved image: {path}")

def download_documents_from_csv(csv_path, output_folder):
    token = get_jwt_token()
    with open(csv_path, newline='') as csvfile:
        reader = list(csv.DictReader(csvfile))
        for row in tqdm(reader, desc="Downloading documents"):
            try:
                request_body = json.loads(row['request_body'])
                ref_id_1 = request_body['reference_id_1']
                ref_id_2 = request_body['reference_id_2']
                subfolder = os.path.join(output_folder, ref_id_1)
                os.makedirs(subfolder, exist_ok=True)
                print(f"\n📥 Downloading {ref_id_1} and {ref_id_2} into {subfolder}...")

                images_1 = dms_document_download(ref_id_1, token)
                images_2 = dms_document_download(ref_id_2, token)

                print(f"🖼️ {ref_id_1}: {len(images_1)} images | {ref_id_2}: {len(images_2)} images")

                save_images(images_1, ref_id_1, subfolder)
                save_images(images_2, ref_id_2, subfolder)
            except Exception as e:
                print(f"❗ Error processing row: {e}")

# Example usage
download_documents_from_csv(
    r"/workspace/scripts/face_matching_API_data.csv",
    r"/workspace/data"
)
