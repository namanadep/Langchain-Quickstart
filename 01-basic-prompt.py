# pip install langchain
import streamlit as st
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

st.title('LangChain Chat with OpenAI')
st.write('**Developed by Naman Adep**')

user_input = st.text_input('Enter your prompt:')

if user_input:
    response = llm.invoke(user_input)
    st.write('Response:')
    st.write(response)