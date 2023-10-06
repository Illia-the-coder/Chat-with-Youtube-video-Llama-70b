import streamlit as st
from gradio_client import Client
from llama_index.llms import Replicate
from llama_index.embeddings import LangchainEmbedding
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import set_global_service_context, ServiceContext, VectorStoreIndex, SimpleDirectoryReader
import os
import logging

PATH = os.path.join(os.path.expanduser("~"), "Data")

if not os.path.exists(PATH):
    os.makedirs(PATH)

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
# Transcribe function
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

# Gradio UI
def youtube_chatbot(youtube_url):
    logging.info(f"Chatbot invoked with YouTube URL: {youtube_url}")
    query_engine = transcribe_video(youtube_url)
    Q = query_engine.query('Give full advanced article describing video transcription you have?').response
    logging.info(f"Query result: {Q}")
    return Q
    
# Streamlit UI
st.title("YouTube Video Chatbot")

# Input for YouTube URL
youtube_url = st.sidebar.text_input("Enter YouTube Video URL:")

if youtube_url:
    st.write("Transcribing video... Please wait.")
    query_engine = transcribe_video(youtube_url)
    with st.status("Requesting Vicuna 13 b"):
        Q = youtube_chatbot
    st.markdown(Q)

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"], avatar=("🧑‍💻" if message["role"] == 'human' else '🦙')):
#         st.markdown(message["content"])

# # User input
# prompt = st.chat_input("Ask something about the video:")

# if prompt := prompt and  query_engine != None:
#     # Display user message in chat message container
#     st.chat_message("human",avatar = "🧑‍💻").markdown(prompt)
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "human", "content": prompt})

#     response = query_engine.query(prompt)
#     response_text = response.response
#     with st.chat_message("assistant", avatar='🦙'):
#         st.markdown(response_text)
#     # Add assistant response to chat history
#     st.session_state.messages.append({"role": "assistant", "content": response})