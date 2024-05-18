import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.messages import HumanMessage, AIMessage

# Loading and preparing the embeddings and documents
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# LangChain setup
llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}
""")

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Retrieval setup
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, chain)

# Streamlit UI setup
st.title('Conversation History')
st.write('Please enter your query and press "Send". The history of the conversation will be used to enhance the responses.')

# Chat history management
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

query = st.text_input('Enter your query:', '')

if st.button('Send'):
    # Update chat history
    st.session_state.chat_history.append(HumanMessage(content=query))
    
    # Retrieve response using the historical context
    response = retrieval_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": query
    })
    
    # Display response and update history
    answer = response.get('answer', 'No answer available.')
    st.session_state.chat_history.append(AIMessage(content=answer))
    st.write('Answer:', answer)

    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.write('You:', message.content)
        elif isinstance(message, AIMessage):
            st.write('AI:', message.content)