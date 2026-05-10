import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("brain_tumor_model.h5")

st.title("Brain Tumor Detection System")
st.write("Upload MRI Image")

uploaded_file = st.file_uploader("Choose Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    st.write(f"Prediction Score: {prediction:.4f}")

    if prediction > 0.5:
        st.error("Tumor Detected")
    else:
        st.success("No Tumor Detected")