import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

st.title('this is my very first app')
st.write('this is a table')

df = pd.DataFrame(np.random.randn(10, 20),
    columns = ('col %d' % i for i in range(20)))

img = Image.open('dog.jpg')

st.write(df)
st.image(img)
