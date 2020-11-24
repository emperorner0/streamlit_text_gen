import pandas as pd
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config
import streamlit as st
import torch
import os
import random


# Device for pytorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
gpt_config = GPT2Config.from_pretrained('model_save/', output_hidden_states=False)
gpt_model = GPT2LMHeadModel.from_pretrained('model_save/', config=gpt_config)

gpt_tokenizer = GPT2Tokenizer.from_pretrained('model_save/',
                                        vocab_file='model_save/vocab.json', merges_file='model_save/merges.txt',
                                        bos_token='<|sot|>', eos_token='<|eot|>', pad_token='<|pad|>')

# Set model to eval
gpt_model.eval() 


# Title of page
st.title("Review Generator")
st.write("""# GPT2 Tuned Book Review Generator""")


# Text box to retrieve prompt
prompt = st.text_input('Please Enter A Prompt for the Review Generator', max_chars=160)


# Activate generation on button push
if st.button('Submit'):
    with st.spinner("Generating Review"):
        prompt = '<|sot|>' + prompt
        promp = torch.tensor(gpt_tokenizer.encode(prompt)).unsqueeze(0).to('cpu')
        outputs = gpt_model.generate(
                                        promp, 
                                        bos_token_id= random.randint(1, 100000),
                                        do_sample=True,   
                                        top_k=30, 
                                        min_length=20,
                                        max_length = 250,
                                        top_p=0.95,
                                        num_return_sequences=1
                                        )
        for i, output in enumerate(outputs):
            st.write("{}\n\n".format(gpt_tokenizer.decode(output, skip_special_tokens=True)))


st.sidebar.write("""
We are testing a new sidebar.
""")
st.sidebar.image('Images/gpt2arch.png', use_column_width=True)