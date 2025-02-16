import streamlit as st
import shared_file as sf
import extract_faces as ef
import os

st.title("Report a Found Child")
st.write("If you have found a missing child, please provide the following details:")

name = st.text_input("Child's Name (if known)")
age = st.number_input("Child's Approximate Age", min_value=0, max_value=40)
gender = st.radio("Child's Gender", ("Male", "Female"))
location_found = st.text_input("Location Where the Child Was Found")

uploaded_photo = st.file_uploader("Upload a Photo of the Child", type=["jpg", "jpeg", "png"])

if st.button("Submit Report"):
    if uploaded_photo is not None:
        original_images_folder = "original_uploaded_images"
        os.makedirs(original_images_folder, exist_ok=True)

        original_image_path = os.path.join(original_images_folder, uploaded_photo.name)
        with open(original_image_path, "wb") as f:
            f.write(uploaded_photo.getbuffer())

        output_folder = "found_child_faces"  
        output_file = "found_child_embeddings" 
        ef.process_image_faces(original_image_path, output_folder, output_file)

        st.success("Thank you for reporting! The information has been submitted.")
        
    else:
        st.error("Please upload a photo of the child.")
