import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

st.title('Prompt Template')
st.write('**Developed by Naman Adep**')

system_prompt = st.text_input('Enter the system prompt:')
user_prompt = st.text_input('Enter your prompt:')

if st.button('Generate'):
    if user_prompt:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}")
        ])
        
        chain = prompt | llm
        response = chain.invoke({"input": user_prompt})
        
        st.write('Response:')
        st.write(response)
    else:
        st.write('Please enter a prompt to get a response.')