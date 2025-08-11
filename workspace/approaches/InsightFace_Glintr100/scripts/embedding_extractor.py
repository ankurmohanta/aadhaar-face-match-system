import onnxruntime as ort
import numpy as np
class Glintr100Embedder:
   def __init__(self, model_path):
       self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
       self.input_name = self.session.get_inputs()[0].name
   def get_embedding(self, img_rgb):
       img = np.transpose(img_rgb, (2, 0, 1))  # HWC to CHW
       img = np.expand_dims(img, axis=0).astype(np.float32)
       img = (img - 127.5) / 127.5
       outputs = self.session.run(None, {self.input_name: img})
       embedding = outputs[0][0]
       return embedding