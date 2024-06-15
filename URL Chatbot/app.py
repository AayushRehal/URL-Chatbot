# import os
# import logging

# # Configure the logging level based on an environment variable or default to "DEBUG" level

# logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredURLLoader

from dotenv import load_dotenv
load_dotenv()  

# Load CSS with background image
st.markdown(
    """
    <style>
    body {
        background-color: red;
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Bloomy The Super Bot📈")
st.sidebar.title("News Article URLs")
# Load CSS with background image

    # Rest of your Streamlit app code goes here
urls = []
for i in range(2):
    url = st.sidebar.text_input(f"URL {i+1}")
    print(url)
    urls.append(url)

print("URLs: ", urls)
print(type(urls[0]))

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"
main_placeholder = st.empty()
os.environ["GOOGLE_API_KEY"] = "AIzaSyCPOYxUyRCPnKnvL10wnA0H4lNnKD4LqFQ"

llm = ChatGoogleGenerativeAI(model="gemini-pro")
if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    print(urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=100
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    print(len(data))
    main_placeholder.text("Text Splitter...finished...✅✅✅")
   
    print(docs)
    # create embeddings and save it to FAISS index
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    main_placeholder.text("Creating embeddings...Started...✅✅✅")

    

    vector_store=FAISS.from_documents(docs,embedding)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vector_store, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)
                    