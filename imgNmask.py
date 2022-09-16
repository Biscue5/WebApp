from turtle import width
import streamlit as st
from PIL import Image
import numpy as np
import tkinter as th

st.title('mask generator')

img_file = st.file_uploader('upload an image', type=['jpg', 'jpeg', 'png'])

if img_file:
    st.markdown(f'{img_file.name} uploaded')
    img = Image.open(img_file)
    img_arr = np.array(img)
    st.image(img)

    print(img_arr.shape)
    height, width = img_arr.shape[:-1]
    print(width, height)

    mask = np.zeros((height, width))
    print('mask shape', mask.shape)

    



