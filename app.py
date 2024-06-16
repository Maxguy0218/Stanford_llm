import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Configure the Google Generative AI
import google.generativeai as genai
genai.configure(api_key='AIzaSyCbyn2VMNwy02PerTQFWyTBcPoD2N4ZJsc')

# Streamlit app title
st.title("Stanford LLM Content Retriever")

# Fetch and clean content from URLs
lecture_urls = [
    "https://stanford-cs324.github.io/winter2022/lectures/introduction/",
    "https://stanford-cs324.github.io/winter2022/lectures/capabilities/",
    "https://stanford-cs324.github.io/winter2022/lectures/data/",
    "https://stanford-cs324.github.io/winter2022/lectures/training/",
]

def fetch_and_clean_content(url):
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')
    content_div = soup.find('div', class_='main-content')
    if not content_div:
        return None
    
    text = content_div.get_text(separator=' ', strip=True)
    return text

def load_and_split(lecture_urls):
    data = []
    for url in lecture_urls:
        content = fetch_and_clean_content(url)
        if content:
            data.append(content)
    return data

# Load and display content
data = load_and_split(lecture_urls)
context = ' '.join(data)

# Split the context into chunks for the LLM
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
texts = text_splitter.split_text(context)

# Initialize embeddings and vector index
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

# Create a prompt template
prompt_template = """
  Answer the question as detailed as possible from the provided context. If the answer is not in the provided context,
  just say, "answer is not available in the context." Do not provide a wrong answer.\n\n
  Context:\n {context}?\n
  Question: \n{question}\n
  Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# Create a language model chain
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Streamlit input for user question
question = st.text_input("Enter your question:")

if question:
    # Get relevant documents
    docs = vector_index.get_relevant_documents(question)

    # Get response from the chain
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    
    # Display the response
    st.markdown(response["output_text"])

# Display the entire context for reference (optional)
with st.expander("Show context"):
    st.write(context)
