import shared_file as sf
import os
import pickle

def process_image_faces(image_path, output_folder, output_file):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_file, exist_ok=True)

    embeddings_dict = {}  

    img = sf.cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    img = sf.preprocess_image(img)
    faces = sf.detector.detect_faces(img)

    print(f"Detected {len(faces)} faces in the image.")

    face = faces[0]
    input_face_region = sf.extract_face(img, face['box'])
    input_face_embedding = sf.get_embedding_InsightFace(input_face_region)
    input_face_embedding = sf.normalize_embedding(input_face_embedding)
    
    embeddings_dict[image_path] = {
        "embedding": input_face_embedding,
    }
    
    embeddings_output_path = os.path.join(output_file, "face_embeddings.pkl")
    with open(embeddings_output_path, 'wb') as file:
        pickle.dump(embeddings_dict, file)
    print(f"Saved {len(embeddings_dict)} face embeddings to {embeddings_output_path}")