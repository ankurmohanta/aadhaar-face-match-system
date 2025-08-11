# Aadhaar Face Match Pipeline - InsightFace (glintr100)

---

## ✅ Project Overview

This project uses InsightFace's glintr100 model to verify if two face images (profile photo & document photo) belong to the same person. It uses pre-trained embeddings combined with a fine-tuned MLP classifier for verification.

---


# Aadhaar Face Match Pipeline - InsightFace (glintr100)

---

## ✅ Project Overview

This project uses InsightFace's glintr100 model to verify if two face images (profile photo & document photo) belong to the same person. It uses pre-trained embeddings combined with a fine-tuned MLP classifier for verification.

---

## 📂 Project Structure

.

├── config.yaml

├── requirements.txt

├── train.py

├── evaluate.py

├── inference.py

├── data_loader.py

├── embedding_extractor.py

├── augmentations.py

├── output/  (models will be saved here)

└── datasets/

├── positive_pairs/

└── negative_pairs/

---

## ⚙ Dataset Structure

- Dataset path is defined inside `config.yaml`.

- Inside dataset folder:

datasets/

├── positive_pairs/

│     ├── pair_0000_img1.jpg

│     ├── pair_0000_img2.jpg

└── negative_pairs/

├── pair_0000_img1.jpg

├── pair_0000_img2.jpg

---

## 🖥 Installation

1️⃣ Create virtual environment (recommended):

```bash

python -m venv venv

source venv/bin/activate  # Linux/mac

venv\Scripts\activate  # Windows

2️⃣ Install dependencies:

pip install -r requirements.txt

3️⃣ InsightFace model will automatically be downloaded at first run.

⸻

🚀 Training

python train.py

✅ Model will be saved in /output/face_match_model.pth

⸻

🔬 Evaluation

python evaluate.py

Shows Accuracy & ROC AUC score on full dataset.

⸻

🔎 Inference

python inference.py --img1 path/to/image1.jpg --img2 path/to/image2.jpg

Shows match probability + prediction.

⸻

🔧 Configuration

All parameters are configurable inside config.yaml:

	•	dataset_path

	•	positive_pairs_dir

	•	negative_pairs_dir

	•	batch_size, learning_rate, num_epochs, etc.

⸻

⚠ Notes

	•	InsightFace will automatically detect and align faces.

	•	If no face is detected, pair will be skipped.

	•	Both CPU and GPU supported.

	•	Augmentations can be enabled via augmentations.py.

