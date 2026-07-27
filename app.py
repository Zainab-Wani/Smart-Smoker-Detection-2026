import streamlit as st

st.title("🚭 Smart Vision-Based Smoking Detection")

st.write("Welcome to the Smoking Detection System")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image")