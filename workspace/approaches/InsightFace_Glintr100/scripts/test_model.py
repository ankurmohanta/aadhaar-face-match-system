import onnxruntime as ort
import numpy as np
import cv2
# Load image
img_path = "/workspace/test_images/Passport.jpeg"
img = cv2.imread(img_path)
img_resized = cv2.resize(img, (112, 112))
# Preprocess (glintr100 normalization)
img_resized = img_resized[..., ::-1]  # BGR to RGB
img_resized = np.transpose(img_resized, (2, 0, 1))  # HWC to CHW
img_resized = np.expand_dims(img_resized, axis=0).astype(np.float32)
img_resized = (img_resized - 127.5) / 127.5
# Load glintr100.onnx directly
onnx_path = "/workspace/approaches/InsightFace_Glintr100/models/glintr100/model.onnx"
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
# Extract embedding
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: img_resized})
embedding = outputs[0][0]
print("✅ Embedding shape:", embedding.shape)
print("First 5 dims:", embedding[:5])