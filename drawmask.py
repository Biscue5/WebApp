import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np

drawing_mode = "freedraw"
canvas_result = None

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
stroke_color = '#ffffff' #white
bg_image = st.file_uploader("Background image:", type=['jpg', 'jpeg', 'png'])

if bg_image:
    img = Image.open(bg_image)
    img = np.array(img)
    height, width = img.shape[:-1]

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
    st.markdown('mask')
    mask = canvas_result.image_data.transpose(2,0,1)[0]

    st.image(canvas_result.image_data.transpose(2,0,1)[0])

    print(np.sum(mask))

#if np.sum(mask):
#    pass

