from scipy.spatial.distance import cosine

def compare_faces(embedding1, embedding2):
    return cosine(embedding1, embedding2)