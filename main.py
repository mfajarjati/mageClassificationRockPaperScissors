#pip install streamlit

import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('my_model')  # Ganti path sesuai dengan lokasi penyimpanan model Anda
    return model

def import_and_predict(image_data, model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_reshape = img[np.newaxis, ...]

    prediction = model.predict(img_reshape)
    return prediction

def main():
    st.title("Rock Paper Scissors Image Classification")

    model = load_model()

    file = st.file_uploader("Please upload an image file", type=["jpg", "png"])

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, model)
        score = tf.nn.softmax(predictions[0])

        # Menampilkan hasil prediksi
        st.subheader("Prediction Results:")
        st.write(f"Rock Probability: {score[0]:.2%}")
        st.write(f"Paper Probability: {score[1]:.2%}")
        st.write(f"Scissors Probability: {score[2]:.2%}")

if __name__ == "__main__":
    main()
