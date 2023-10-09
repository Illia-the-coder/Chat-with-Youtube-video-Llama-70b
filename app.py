import os
import logging
from typing import Any, List, Mapping, Optional

from gradio_client import Client
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
import streamlit as st

models = '''| Model | Llama2 | Llama2-hf | Llama2-chat | Llama2-chat-hf |
|---|---|---|---|---|
| 70B | [Link](https://huggingface.co/meta-llama/Llama-2-70b) | [Link](https://huggingface.co/meta-llama/Llama-2-70b-hf) | [Link](https://huggingface.co/meta-llama/Llama-2-70b-chat) | [Link](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) |
---'''

TITLE = "Chat-with-Youtube-video-Llama-70b"
DESCRIPTION = """
This Space demonstrates model [Llama-2-70b-chat-hf](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) by Meta, that can be used to chat with a YouTube video. 
It uses the checkpoint [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2) and 🤗 Transformers to transcribe audio files.
For embeddings, we use [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.
"""

# ... [rest of the function definitions]

def initialize_session_state():
    if "youtube_url" not in st.session_state:
        st.session_state.youtube_url = ""
    if "setup_done" not in st.session_state:  # Initialize the setup_done flag
        st.session_state.setup_done = False
    if "doneYoutubeurl" not in st.session_state:
        st.session_state.doneYoutubeurl = ""

# ... [rest of the function definitions]

st.title("YouTube Video Chatbot")
sidebar()
initialize_session_state()

# Check if a new YouTube URL is provided
if st.session_state.youtube_url != st.session_state.doneYoutubeurl:
    st.session_state.setup_done = False

if st.session_state.youtube_url and not st.session_state.setup_done:
    with st.status("Transcribing video..."):
        data = transcribe_video(st.session_state.youtube_url, PATH)
    
    with st.status("Running Embeddings..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(data)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
        docsearch = FAISS.from_documents(docs, embeddings)
        retriever = docsearch.as_retriever()
        retriever.search_kwargs['distance_metric'] = 'cos'
        retriever.search_kwargs['k'] = 4

    with st.status("Running RetrievalQA..."):
        llama_instance = LlamaLLM()
        st.session_state.qa = RetrievalQA.from_chain_type(llm=llama_instance, chain_type="stuff", retriever=retriever)

    st.session_state.doneYoutubeurl = st.session_state.youtube_url
    st.session_state.setup_done = True  # Mark the setup as done for this URL

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=("🧑‍💻" if message["role"] == 'human' else '🦙')):
        st.markdown(message["content"])

textinput = st.chat_input("Ask LLama-2-70b anything about the video...") 

if prompt := textinput:
    st.chat_message("human", avatar="🧑‍💻").markdown(prompt)
    st.session_state.messages.append({"role": "human", "content": prompt})
    with st.status("Requesting Client..."):
        response = st.session_state.qa.run(prompt)
    with st.chat_message("assistant", avatar='🦙'):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
