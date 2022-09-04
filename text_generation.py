import torch
import pandas as pd
import streamlit as st

from transformers import AutoTokenizer, AutoModelForCausalLM

def text_generator(input_txt, n_steps):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    input_ids = tokenizer(input_txt, return_tensors='pt')['input_ids'].to(device)
    output = model.generate(input_ids, max_new_tokens=n_steps, do_sample=False)

    return tokenizer.decode(output[0])


st.title('Text Generator')

input_txt = st.text_input('type a couple of words')
sentence_l = st.number_input('the length of sentence', 0, 20, 0)

button = st.button('Generate')

st.write('Output: \n')

if button:
    st.write(text_generator(input_txt, sentence_l))