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

def transcribe_video(youtube_url: str, path: str) -> List[Document]:
    """
    Transcribe a video and return its content as a Document.
    """
    logging.info(f"Transcribing video: {youtube_url}")
    client = Client("https://sanchit-gandhi-whisper-jax.hf.space/")
    result = client.predict(youtube_url, "translate", True, fn_index=7)
    return [Document(page_content=result[1], metadata=dict(page=1))]


def predict(message: str, system_prompt: str = '', temperature: float = 0.7, max_new_tokens: int = 4096,
            topp: float = 0.5, repetition_penalty: float = 1.2) -> Any:
    """
    Predict a response using a client.
    """
    client = Client("https://ysharma-explore-llamav2-with-tgi.hf.space/")
    response = client.predict(
        message,
        system_prompt,
        temperature,
        max_new_tokens,
        topp,
        repetition_penalty,
        api_name="/chat_1"
    )
    return response


class LlamaLLM(LLM):
    """
    Custom LLM class.
    """

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        response = predict(prompt)
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

PATH = os.path.join(os.path.expanduser("~"), "Data")

def initialize_session_state():
    if "youtube_url" not in st.session_state:
        st.session_state.youtube_url = ""

def sidebar():
    with st.sidebar:
        st.markdown(
            "## How to use\n"
            "1. Enter the YouTube Video URL below🔗\n"
        )
        st.session_state.youtube_url = st.text_input("YouTube Video URL:")

st.set_page_config(page_title="YouTube Video Chatbot",
                   layout="centered",
                   initial_sidebar_state="expanded")

st.title("YouTube Video Chatbot")
sidebar()
initialize_session_state()

if st.session_state.youtube_url:
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
      qa = RetrievalQA.from_chain_type(llm=llama_instance, chain_type="stuff", retriever=retriever)

    if "messages" not in st.session_state:
      st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=("🧑‍💻" if message["role"] == 'human' else '🦙')):
            st.markdown(message["content"])

    textinput = st.chat_input("Ask LLama-2-70b anything about the video...")   

    if prompt := textinput:
      st.chat_message("human",avatar = "🧑‍💻").markdown(prompt)
      st.session_state.messages.append({"role": "human", "content": prompt})

      response = qa.run(message=prompt)
      with st.chat_message("assistant", avatar='🦙'):
          st.markdown(response)
      # Add assistant response to chat history
      st.session_state.messages.append({"role": "assistant", "content": response})