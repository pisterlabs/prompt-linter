from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper, download_loader
import streamlit as st
import openai

BASE_PROMPT = [{"role": "system", "content": """
    You are a strategy researcher based in the UK.
    You use complicated examples from Wardley Mapping and the platform business model in your answers, focusing on lesser-known advice to better illustrate your arguments.
    Your language should be for an 12 year old to understand.
    If you do not know the answer to a question, do not make information up - instead, ask a follow-up question in order to gain more context.
    Use a mix of technical and colloquial uk englishlanguage to create an accessible and engaging tone.
    Provide your answers using Wardley Mapping and the Platform Business Model in a form of a sarcastic tweet.
    """}]

openai.api_key = st.secrets["OPENAI_API_KEY"]

YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")
loader = YoutubeTranscriptReader()
documents = loader.load_data(ytlinks=['https://www.youtube.com/watch?v=kUPCGrTUDB8'])

index = GPTVectorStoreIndex.from_documents(documents)

st.set_page_config(page_title="UN Global Platform & Open Source")
st.title("UN Global Platform")
st.sidebar.markdown("# Query this video using AI")
st.sidebar.markdown("Developed by Mark Craddock](https://twitter.com/mcraddock)", unsafe_allow_html=True)
st.sidebar.markdown("Current Version: 0.0.2")

st.video('https://youtu.be/kUPCGrTUDB8') 

text = st.empty()
prompt = st.text_input("Prompt", value="What is this video about?")
query_engine = index.as_query_engine()
if st.button("Send"):
    with st.spinner("Generating response..."):
        
        response = query_engine.query(prompt)
        text.text_area("Messages", response, height=250)
if st.button("Clear"):
    st.session_state["messages"] = BASE_PROMPT
    show_messages(text)
