import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np


vgg16_model = load_model("vgg16_fashion_mnist.h5")
cnn_model = load_model("cnn_fashion_mnist.h5")  

classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

st.title("Класифікація зображень Fashion MNIST")
st.write("Завантажте зображення для класифікації за допомогою VGG16 або власної CNN.")

model_choice = st.selectbox("Оберіть модель для класифікації:", ["VGG16", "Custom CNN"])

uploaded_file = st.file_uploader("Завантажте зображення у форматі JPG/PNG:", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Завантажене зображення", use_column_width=True)

    img = load_img(uploaded_file, target_size=(32, 32))  
    img = img_to_array(img) / 255.0  
    img = np.expand_dims(img, axis=0)  

    if model_choice == "VGG16":
        predictions = vgg16_model.predict(img)
    elif model_choice == "Custom CNN":
        predictions = cnn_model.predict(img)

    predicted_class = np.argmax(predictions)  
 
    st.write(f"**Передбачений клас:** {classes[predicted_class]}")
    st.write("**Ймовірності для кожного класу:**")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{classes[i]}: {prob * 100:.2f}%")

    st.bar_chart(predictions[0])