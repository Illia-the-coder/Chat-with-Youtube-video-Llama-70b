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
from langchain.prompts import PromptTemplate
import streamlit as st
from pytube import YouTube



models = '''| Model | Llama2 | Llama2-hf | Llama2-chat | Llama2-chat-hf |
|---|---|---|---|---|
| 70B | [Link](https://huggingface.co/meta-llama/Llama-2-70b) | [Link](https://huggingface.co/meta-llama/Llama-2-70b-hf) | [Link](https://huggingface.co/meta-llama/Llama-2-70b-chat) | [Link](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) |
---'''


DESCRIPTION = """
Welcome to the **YouTube Video Chatbot** powered by the state-of-the-art Llama-2-70b model. Here's what you can do:
- **Transcribe & Understand**: Provide any YouTube video URL, and our system will transcribe it. Our advanced NLP model will then understand the content, ready to answer your questions.
- **Ask Anything**: Based on the video's content, ask any question, and get instant, context-aware answers.
To get started, simply paste a YouTube video URL in the sidebar and start chatting with the model about the video's content. Enjoy the experience!
"""
st.title("YouTube Video Chatbot")
st.markdown(DESCRIPTION)

def get_video_title(youtube_url: str) -> str:
    yt = YouTube(youtube_url)
    embed_url = f"https://www.youtube.com/embed/{yt.video_id}"
    embed_html = f'<iframe  src="{embed_url}" frameborder="0" allowfullscreen></iframe>'
    return yt.title, embed_html


def transcribe_video(youtube_url: str, path: str) -> List[Document]:
    """
    Transcribe a video and return its content as a Document.
    """
    logging.info(f"Transcribing video: {youtube_url}")
    client = Client("https://sanchit-gandhi-whisper-jax.hf.space/")
    result = client.predict(youtube_url, "translate", True, fn_index=7)
    return [Document(page_content=result[1], metadata=dict(page=1))]





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
    if "setup_done" not in st.session_state:  # Initialize the setup_done flag
        st.session_state.setup_done = False
    if "doneYoutubeurl" not in st.session_state:
        st.session_state.doneYoutubeurl = ""

def sidebar():
    with st.sidebar:
        st.markdown("Enter the YouTube Video URL below🔗\n")
        st.session_state.youtube_url = st.text_input("YouTube Video URL:")

        if st.session_state.youtube_url:
            # Get the video title
            video_title, embed_html = get_video_title(st.session_state.youtube_url)
            st.markdown(f"### {video_title}")

            # Embed the video
            st.markdown(
                embed_html,
                unsafe_allow_html=True
            )
            
        # system_promptSide = st.text_input("Optional system prompt:")
        # temperatureSide = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.9, step=0.05)
        # max_new_tokensSide = st.slider("Max new tokens", min_value=0.0, max_value=4096.0, value=4096.0, step=64.0)
        # ToppSide = st.slider("Top-p (nucleus sampling)", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
        # RepetitionpenaltySide = st.slider("Repetition penalty", min_value=0.0, max_value=2.0, value=1.2, step=0.05)


def predict(message: str) -> Any:
    """
    Predict a response using a client.
    """
    client = Client("https://ysharma-explore-llamav2-with-tgi.hf.space/")
    response = client.predict(
        message,
        '',
        0.9,
        4096,
        0.6,
        1.2,
        api_name="/chat_1"
    )
    return response

sidebar()
initialize_session_state()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")

prompt = PromptTemplate(
    template="""Given the context about a video. Answer the user in a friendly and precise manner.
    Context: {context}
    Human: {question}
    AI:""",
    input_variables=["context", "question"]
)

# Check if a new YouTube URL is provided
if st.session_state.youtube_url != st.session_state.doneYoutubeurl:
    st.session_state.setup_done = False

if st.session_state.youtube_url and not st.session_state.setup_done:
    with st.status("Transcribing video..."):
      data = transcribe_video(st.session_state.youtube_url, PATH)
    
    with st.status("Running Embeddings..."):
      docs = text_splitter.split_documents(data)

      docsearch = FAISS.from_documents(docs, embeddings)
      retriever = docsearch.as_retriever()
      retriever.search_kwargs['distance_metric'] = 'cos'
      retriever.search_kwargs['k'] = 4
    with st.status("Running RetrievalQA..."):
      llama_instance = LlamaLLM()
      st.session_state.qa = RetrievalQA.from_chain_type(llm=llama_instance, chain_type="stuff", retriever=retriever,chain_type_kwargs={"prompt": prompt})
        
    st.session_state.doneYoutubeurl = st.session_state.youtube_url
    st.session_state.setup_done = True  # Mark the setup as done for this URL

if "messages" not in st.session_state:
  st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=("🧑‍💻" if message["role"] == 'human' else '🦙')):
        st.markdown(message["content"])

textinput = st.chat_input("Ask LLama-2-70b anything about the video...") 

if prompt := textinput:
  st.chat_message("human",avatar = "🧑‍💻").markdown(prompt)
  st.session_state.messages.append({"role": "human", "content": prompt})
  with st.status("Requesting Client..."):
      response = st.session_state.qa.run(prompt)
  with st.chat_message("assistant", avatar='🦙'):
      st.markdown(response)
  # Add assistant response to chat history
  st.session_state.messages.append({"role": "assistant", "content": response})
