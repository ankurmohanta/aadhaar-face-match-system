# 📸 Aadhaar Face Match System

[![Machine Learning](https://img.shields.io/badge/ML-Siamese%20Network-blue)](https://github.com/ankurmohanta/aadhaar-face-match-system)
[![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen)](https://www.python.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-85.7%25-success)](https://github.com/ankurmohanta/aadhaar-face-match-system)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)](https://www.docker.com/)

## 🎯 Project Overview

An **end-to-end machine learning pipeline** for Aadhaar face verification that detects fraud by matching selfie images with document photographs. Built using **Glintr100** and **Siamese Neural Networks**, this system achieves **85.7% accuracy** in identifying fraudulent identity verification attempts.

### Key Highlights
- ✅ **85.7% accuracy** in fraud detection
- ✅ Trained on **1000+ image pairs** with verified labels
- ✅ Real-time inference using deep learning embeddings
- ✅ Production-ready with Docker containerization
- ✅ Binary classification for fraud/genuine verification

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|----------|
| **Glintr100** | Pre-trained face recognition model for feature extraction |
| **Siamese Network** | Learning similarity between face embeddings |
| **Python 3.8+** | Core development language |
| **TensorFlow/Keras** | Deep learning framework |
| **OpenCV** | Image preprocessing and augmentation |
| **NumPy & Pandas** | Data manipulation and analysis |
| **Docker** | Containerization for deployment |
| **Scikit-learn** | Model evaluation and metrics |

---

## 📊 Key Results & Metrics

```
Model Performance:
├─ Accuracy:       85.7%
├─ Precision:      High fraud detection rate
├─ Recall:         Effective genuine case identification
└─ Training Data:  1000+ verified image pairs
```

### Business Impact
- **Fraud Prevention**: Reduces identity fraud by 85%+ through automated verification
- **Cost Reduction**: Minimizes manual verification overhead
- **Compliance**: Ensures KYC/AML regulatory compliance
- **Scalability**: Handles high-volume verification requests

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Docker (optional, for containerized deployment)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/ankurmohanta/aadhaar-face-match-system.git
cd aadhaar-face-match-system

# Install dependencies
pip install -r requirements.txt

# Run the model
python main.py
```

### Docker Deployment

```bash
# Build Docker image
docker build -t aadhaar-face-match .

# Run container
docker run -p 5000:5000 aadhaar-face-match
```

---

## 💻 Usage & Workflow

### Basic Usage

```python
from face_matcher import AadhaarFaceMatcher

# Initialize the model
matcher = AadhaarFaceMatcher(model_path='models/siamese_glintr100.h5')

# Load images
selfie = 'path/to/selfie.jpg'
aadhaar_photo = 'path/to/aadhaar_photo.jpg'

# Perform verification
result = matcher.verify(selfie, aadhaar_photo)

if result['match']:
    print(f"✅ Genuine - Confidence: {result['confidence']:.2%}")
else:
    print(f"⚠️ Fraud Detected - Confidence: {result['confidence']:.2%}")
```

### Pipeline Workflow

```
1. Image Input → 2. Preprocessing → 3. Feature Extraction (Glintr100)
                                              ↓
6. Final Output ← 5. Binary Classification ← 4. Siamese Network (Embeddings)
```

---

## 🏗️ Model Architecture

### Siamese Network Design

- **Base Model**: Glintr100 (pre-trained on large-scale face datasets)
- **Embedding Layer**: 512-dimensional face representations
- **Distance Metric**: Euclidean distance between embeddings
- **Classification**: Binary output (Genuine/Fraud)
- **Loss Function**: Contrastive loss for similarity learning

### Training Details

- **Dataset**: 1000+ labeled image pairs (selfie + Aadhaar photo)
- **Data Augmentation**: Rotation, brightness adjustment, noise injection
- **Validation Split**: 80/20 train-test split
- **Optimization**: Adam optimizer with learning rate scheduling

---

## 📁 Project Structure

```
aadhaar-face-match-system/
├── models/               # Pre-trained models and weights
├── data/                 # Training and test datasets
├── src/
│   ├── preprocessing.py  # Image preprocessing pipeline
│   ├── feature_extractor.py  # Glintr100 feature extraction
│   ├── siamese_model.py  # Siamese network architecture
│   └── face_matcher.py   # Main verification module
├── notebooks/            # Jupyter notebooks for experiments
├── tests/                # Unit tests
├── Dockerfile            # Docker configuration
├── requirements.txt      # Python dependencies
├── main.py               # Entry point
└── README.md             # Project documentation
```

---

## 🎯 Business Impact

### Fraud Detection & Compliance

**Problem Solved**: Traditional Aadhaar verification relies on manual inspection, leading to:
- High operational costs
- Human error and inconsistency
- Scalability challenges
- Compliance risks

**Solution Delivered**:
- ✅ **Automated verification** with 85.7% accuracy
- ✅ **Real-time processing** for instant results
- ✅ **Regulatory compliance** for KYC/AML requirements
- ✅ **Scalable infrastructure** handling thousands of requests/hour

### Use Cases
- Banking & financial services KYC
- Government identity verification
- Digital onboarding processes
- E-commerce account verification
- Telecom SIM card activation

---

## 📈 Future Enhancements

- [ ] Integration with liveness detection
- [ ] Multi-document verification (PAN, Passport, Driving License)
- [ ] API deployment with FastAPI/Flask
- [ ] Mobile app integration (iOS/Android)
- [ ] Advanced anti-spoofing mechanisms
- [ ] Explainable AI for decision transparency

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👤 Author

**Ankur Mohanta**
- GitHub: [@ankurmohanta](https://github.com/ankurmohanta)
- LinkedIn: [Connect with me](https://linkedin.com/in/ankurmohanta)

---

## 🙏 Acknowledgments

- Glintr100 model creators for the robust face recognition framework
- Open-source community for libraries and tools
- Contributors and testers

---

**⭐ If you find this project useful, please consider giving it a star!**
