import os
import shutil
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import NotebookLoader, TextLoader


def vector_db(embeddings, code):
    """Creates and persists a vector database using Chroma from given embeddings and code documents.
    
    Args:
        embeddings (Embeddings): The embedding function to use for vectorizing the documents.
        code (List[Document]): A list of code documents to be added to the vector database.
    
    Returns:
        Chroma: A persisted Chroma vector database instance.
    
    Raises:
        OSError: If there are issues with file system operations during directory removal or creation.
        ChromaError: If there are issues with Chroma database operations.
    """
    collection_name = "db_collection"
    local_directory = "db_" + st.session_state["db_name"]
    persist_directory = os.path.join(os.getcwd(), local_directory)

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    vec_db = Chroma.from_documents(
        documents=code,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory
    )

    vec_db.persist()

    return vec_db


def load_to_db(repo_path):
    """Loads and processes files from a given repository path into a format suitable for database insertion.
    
    Args:
        repo_path (str): The path to the repository containing the files to be processed.
    
    Returns:
        list: A list of processed document chunks ready for database insertion.
    
    Raises:
        Exception: Catches and silently ignores any exceptions during file loading and processing.
    """
    docs = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for filename in files:
            if filename.startswith('.'):
                continue
            if filename == 'package-lock.json':
                continue
            file_path = os.path.join(root, filename)
            try:
                if file_path.endswith('.ipynb'):
                    loader = NotebookLoader(file_path)
                else:
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs.extend(loader.load_and_split())
            except Exception as e:
                pass

    code_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    code = code_splitter.split_documents(docs)
    return code