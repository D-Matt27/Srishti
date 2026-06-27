
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load Trained Model
model = tf.keras.models.load_model("cat_dog_model.h5")

# App Title
st.title("Cat vs Dog Classifier")

st.write("Upload an image and the model will predict whether it is a cat or dog.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Display Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess Image
    img = image.resize((150,150))
    img_array = np.array(img) / 255.0

    # Ensure RGB
    if img_array.shape[-1] == 4:
        img_array = img_array[:,:,:3]

    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    # Show Result
    if prediction[0][0] > 0.5:
        st.success("Prediction: Dog")
    else:
        st.success("Prediction: Cat")
