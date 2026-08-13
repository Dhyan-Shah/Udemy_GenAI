import os
from dotenv import load_dotenv

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()



## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ('system','You are a helpful assistant. Please respond to the question asked'),
        ('user','Question:{question}')
    ]
)

## streamlit framework
st.title('Langchain Demo With Gemma Model')
input_text=st.text_input('What question you have in mind?')


## Ollama Model
llm=Ollama(model='gemma:2b')
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))
