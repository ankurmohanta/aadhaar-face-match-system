import cv2
def preprocess_image(image_path):
   img = cv2.imread(image_path)
   img_resized = cv2.resize(img, (112, 112))
   img_resized = img_resized[..., ::-1]  # BGR to RGB
   return img_resized