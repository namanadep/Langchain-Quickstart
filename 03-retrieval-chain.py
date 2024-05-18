import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain

# Load documents from a web base
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()

# Split and embed documents
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# Set up the LangChain
llm = ChatOpenAI()
prompt_template = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}
""")
chain = prompt_template | llm | StrOutputParser()

# Create retrieval chain with the correct arguments
document_chain = create_retrieval_chain(vector.as_retriever(), chain)

# Streamlit interface
st.title('Internet Access')
query = st.text_input('Enter your query:', '')

if st.button('Get Answer'):
    response = document_chain.invoke({"input": query})
    answer = response.get('answer', 'No answer available.')
    st.write('Answer:', answer)
else:
    st.write('Enter a query and press the button.')
