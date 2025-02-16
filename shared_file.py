import cv2
import os
from mtcnn import MTCNN
from insightface.app import FaceAnalysis
import numpy as np
import pickle
from skimage import exposure

detector = MTCNN()
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(320, 320))

def extract_face(img, box):
    x, y, width, height = box
    face_region = img[y:y+height, x:x+width]
    face_region = cv2.resize(face_region, (160, 160))
    face_region = face_region.astype('float32')
    return face_region

def normalize_embedding(embedding):
    max_abs = np.max(np.abs(embedding))
    if max_abs == 0:
        return embedding
    return embedding / max_abs

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    adaptive_eq = clahe.apply(equalized)
    denoised = cv2.fastNlMeansDenoising(adaptive_eq, None, h=10, searchWindowSize=21, templateWindowSize=7)
    enhanced = exposure.adjust_gamma(denoised, gamma=1.2)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return enhanced_bgr

def get_embedding_InsightFace(face_region):
    
    faces = app.get(face_region)
    if faces:
        return faces[0].embedding
    else:
        print("No face detected by InsightFace.")
        return None