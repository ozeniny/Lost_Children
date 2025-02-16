import shared_file as sf
import cv2
import numpy as np

def get_image(input_img):

    print(f"Input image type: {type(input_img)}")
    if isinstance(input_img, str):  
        img = cv2.imread(input_img)
        if img is None:
            raise ValueError(f"Failed to load image from path: {input_img}")
    else: 
        img = np.frombuffer(input_img.read(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode uploaded image.")

    print(f"Image shape: {img.shape}")

    img = sf.preprocess_image(img)
    input_face = sf.detector.detect_faces(img)

    if input_face:
        input_face_region = sf.extract_face(img, input_face[0]['box'])
        input_face_embedding = sf.get_embedding_InsightFace(input_face_region)
        input_face_embedding = sf.normalize_embedding(input_face_embedding)
        print(f"Input face embedding shape: {input_face_embedding.shape}")
    else:
        raise ValueError("No face detected in the input image")

    return input_face_embedding