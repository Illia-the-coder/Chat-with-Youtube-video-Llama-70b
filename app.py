import os
import logging
import streamlit as st
from gradio_client import Client
from llama_index.llms import Replicate
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import set_global_service_context, ServiceContext, VectorStoreIndex, SimpleDirectoryReader

PATH = os.path.join(os.path.expanduser("~"), "Data")

def initialize_session_state():
    if "youtube_url" not in st.session_state:
        st.session_state.youtube_url = ""
    if "REPLICATE_API_TOKEN" not in st.session_state:
        st.session_state.REPLICATE_API_TOKEN = ""

def transcribe_video(youtube_url):
    logging.info(f"Transcribing video: {youtube_url}")
    client = Client("https://sanchit-gandhi-whisper-jax.hf.space/")
    try:
        result = client.predict(youtube_url, "transcribe", True, fn_index=7)
        with open(f'{PATH}/docs.txt', 'w') as f:
            f.write(result[1])
    except Exception as e:
        logging.error(f"Error transcribing video: {e}")
        raise ValueError(f"Error transcribing video: {e}")
    documents = SimpleDirectoryReader(PATH).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()

def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Enter the YouTube Video URL below🔗\n"
            "2. Input your REPLICATE_API_TOKEN🔑\n"
        )
        st.session_state.youtube_url = st.text_input("YouTube Video URL:")
        st.session_state.REPLICATE_API_TOKEN = st.text_input(
            "REPLICATE_API_TOKEN",
            type="password",
            placeholder="Paste your REPLICATE_API_TOKEN here"
        )

st.set_page_config(page_title="YouTube Video Chatbot",
                   layout="centered",
                   initial_sidebar_state="expanded")

st.title("YouTube Video Chatbot")
sidebar()
initialize_session_state()

if st.session_state.youtube_url:
    if not st.session_state.REPLICATE_API_TOKEN:
        st.error("Please enter your REPLICATE_API_TOKEN in the sidebar to continue.")
    else:
        os.environ["REPLICATE_API_TOKEN"] = st.session_state.REPLICATE_API_TOKEN
        llm = Replicate(
            model="replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b"
        )

        embeddings = LangchainEmbedding(
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        )

        service_context = ServiceContext.from_defaults(
            chunk_size=1024,
            llm=llm,
            embed_model=embeddings
        )
        set_global_service_context(service_context)

        with st.spinner("Transcribing video... Please wait."):
            query_engine = transcribe_video(st.session_state.youtube_url)
            Q = query_engine.query('Give full advanced article describing video transcription you have?').response
            st.markdown(Q)
