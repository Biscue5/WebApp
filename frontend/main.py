from email.mime import base
from importlib.metadata import files
import os
import json
from urllib import request
import requests
import base64
import io
import httpx 
import chardet

from PIL import Image
import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas

BACKEND_HOST = os.environ.get('BACKEND_HOST', '127.0.0.1:8000')

col1, col2 = st.columns(2)

drawing_mode = "freedraw"
canvas_result = None

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 11)
stroke_color = '#ffffff' #white
with col1:
    st.header('Input Image')
    bg_image = st.sidebar.file_uploader("Background image:", type=['jpg', 'jpeg', 'png'])

if bg_image:
    img = Image.open(bg_image)
    img_ = np.array(img)
    height, width = img_.shape[:-1]

realtime_update = st.sidebar.checkbox("Update in realtime", True)

if bg_image:
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        #background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=height,
        width=width,
        drawing_mode=drawing_mode,
        point_display_radius=0,
        key="canvas",
    )
if canvas_result:
    try:
        mask = canvas_result.image_data.transpose(2,0,1)[0]
        display_mask = Image.fromarray(mask)
        with col2:
            st.header('Masked Region')
            st.image(mask, use_column_width=True)
        
        if st.button('generate', key=1):
            with io.BytesIO() as output:
                img.save(output, format='PNG')
                bytes_img = output.getvalue()

            with io.BytesIO() as output:
                display_mask.save(output, format='PNG')
                bytes_mask = output.getvalue()

            files = [('files', file) for file in [bytes_img, bytes_mask]]

            with st.spinner('Generating'):
                inpainted_img = httpx.post(f'http://{BACKEND_HOST}/predict/', files=files, timeout=120)
            
            print('status:', inpainted_img)

            st.markdown('Generated Image')
            inpainted_img = base64.b64decode(inpainted_img.content)
            inpainted_img = Image.open(io.BytesIO(inpainted_img))
            inpainted_img = inpainted_img.resize((width, height))
            st.image(inpainted_img, use_column_width=True)
        
    except:
        pass

