import streamlit as st
import shared_file as sf
from get_image import get_image
from face_compare import compare_faces
import pickle
import os
import cv2

st.title("Search for a Missing Child")
st.write("If you are searching for your missing child, please provide the following details:")

child_name = st.text_input("Child's Name")
child_age = st.number_input("Child's Age", min_value=0, max_value=40)
child_gender = st.radio("Child's Gender", ("Male", "Female"))

uploaded_photo = st.file_uploader("Upload a Photo of Your Child", type=["jpg", "jpeg", "png"])

if st.button("Search for Child"):
    if uploaded_photo is not None:

        try:
            input_face_embedding = get_image(uploaded_photo)
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.stop()

        st.success("Searching for your child...")
        st.write("Details Provided:")
        st.write(f"- Name: {child_name}")
        st.write(f"- Age: {child_age}")
        st.write(f"- Gender: {child_gender}")

        embeddings_path = "found_child_embeddings/face_embeddings.pkl"

        if not os.path.exists(embeddings_path):
            st.error("No embeddings found. Please report a found child first.")
            st.stop()

        with open(embeddings_path, 'rb') as file:
            dataset_embeddings = pickle.load(file)

        similarities = []
        for img_path, embedding in dataset_embeddings.items():
            candidate_embedding = embedding
            original_image_path = img_path
            dist = compare_faces(input_face_embedding, candidate_embedding)
            similarities.append((dist, original_image_path))
        similarities.sort(key=lambda x: x[0])

        if similarities:
            most_similar_dist, original_image_path = similarities[0]
            if os.path.exists(original_image_path):
                original_image = cv2.imread(original_image_path)
                if original_image is not None:
                    st.image(original_image, caption="Original Image", use_column_width=True, channels="BGR")
                else:
                    st.error("Failed to load the original image.")
            else:
                st.error(f"Original image not found at path: {original_image_path}")
        else:
            st.error("No similar face found.")
    else:
        st.error("Please upload a photo of your child.")