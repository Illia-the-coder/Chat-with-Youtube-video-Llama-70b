import streamlit as st
from gradio_client import Client
from llama_index.llms import Replicate
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import set_global_service_context, ServiceContext, VectorStoreIndex, SimpleDirectoryReader
import os
# Ensure the environment variable is set
if "REPLICATE_API_TOKEN" not in os.environ:
    raise ValueError("Please set the REPLICATE_API_TOKEN environment variable.")
else:
    os.environ["REPLICATE_API_TOKEN"] = os.environ["REPLICATE_API_TOKEN"]

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

# Transcribe function
def transcribe_video(youtube_url):
    client = Client("https://sanchit-gandhi-whisper-jax.hf.space/")
    result = client.predict(youtube_url, "transcribe", True, fn_index=7)
    with open(f'{PATH}/docs.txt','w') as f:
        f.write(result[1])

    documents = SimpleDirectoryReader(PATH).load_data()
    index = VectorStoreIndex.from_documents(documents)
    return index.as_query_engine()

# Streamlit UI
st.title("YouTube Video Chatbot")

# Input for YouTube URL
youtube_url = st.text_input("Enter YouTube Video URL:")

if youtube_url and "query_engine" not in st.session_state:
    st.write("Transcribing video... Please wait.")
    st.session_state.query_engine = transcribe_video(youtube_url)

# Chatbot UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "human":
        st.write(f"You: {message['content']}")
    else:
        st.write(f"Chatbot: {message['content']}")

# User input
prompt = st.text_input("Ask something about the video:")

# React to user input
if prompt and "query_engine" in st.session_state:
    # Add user message to chat history
    st.session_state.messages.append({"role": "human", "content": prompt})

    # Get response from the chatbot
    response = st.session_state.query_engine.query(prompt)
    response_text = response.response  # Assuming the response has a 'response' attribute with the answer

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response_text})

# Refresh the page to show the updated chat history
if prompt:
    st.experimental_rerun()
