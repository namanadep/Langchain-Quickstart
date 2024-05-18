import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import create_openai_functions_agent, AgentExecutor

# Load and prepare embeddings and documents
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
embeddings = OpenAIEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# LangChain setup
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
prompt_template = ChatPromptTemplate.from_template("""
You are a world class technical documentation writer.
Input: {input}
Agent Scratchpad: {agent_scratchpad}
""")

# Tools and agent setup
retriever = vector.as_retriever()
retriever_tool = create_retriever_tool(retriever, "langsmith_search", "Search for information about LangSmith.")
search = TavilySearchResults()
tools = [retriever_tool, search]
agent = create_openai_functions_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Streamlit interface
st.title('Agents in LangChain')
query = st.text_input('Enter your query:', '')

if st.button('Submit Query'):
    response = agent_executor.invoke({"input": query, "agent_scratchpad": {}})
    st.write('Full Response:', response)  # Log the full response object
    if 'text' in response:
        answer = response['text']
        st.write('Response:', answer)
    else:
        st.write(' ')
