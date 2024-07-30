import streamlit as st
import os
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# Set your Hugging Face token
HUGGINGFACE_TOKEN = "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"
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACE_TOKEN

st.title("Querying CSVs with LLMs")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)
    st.write(dataframe)
    
    temp_csv_path = "temp_uploaded_file.csv"
    dataframe.to_csv(temp_csv_path, index=False)
    
    loader = CSVLoader(file_path=temp_csv_path)
    data = loader.load() 

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(data)
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = Chroma.from_documents(text_chunks, embeddings)
    
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name,use_auth_token=HUGGINGFACE_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name,use_auth_token=HUGGINGFACE_TOKEN)
    
    qa = ConversationalRetrievalChain.from_llm(model, retriever=db.as_retriever())
    
    st.write("Enter your Query..")
    query = st.text_input("Input Prompt: ")
    
    if query:
        with st.spinner("Processing your question.."):
            inputs = tokenizer(query, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write("Response:", answer)
