import numpy as np
import cv2
import onnxruntime as ort
class Glintr100Embedder:
   def __init__(self, model_path):
       self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
       self.input_name = self.session.get_inputs()[0].name
   def preprocess(self, img):
       img = cv2.resize(img, (112, 112))
       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
       img = img.astype(np.float32)
       img = np.transpose(img, (2, 0, 1))  # HWC to CHW
       img = np.expand_dims(img, axis=0)  # Add batch dimension
       img = (img - 127.5) / 128.0
       return img
   def get_embedding(self, img_path):
       img = cv2.imread(img_path)
       if img is None:
           raise ValueError(f"❌ Failed to read image: {img_path}")
       img = self.preprocess(img)
       embedding = self.session.run(None, {self.input_name: img})[0][0]
       return embedding / np.linalg.norm(embedding)  # L2 normalize