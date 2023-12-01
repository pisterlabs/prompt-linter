import streamlit as st
from langchain.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain import PromptTemplate

def process_transcript(openai_api_key, youtube_url, summary_type):
    # Load Transcript
    loader = YoutubeLoader.from_youtube_url(youtube_url, language=["en", "en-US"])
    transcript = loader.load()

    # Split Transcript
    splitter = TokenTextSplitter(model_name="gpt-3.5-turbo-16k", chunk_size=10000, chunk_overlap=100)
    chunks = splitter.split_documents(transcript)

    # Set up LLM
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo-16k", temperature=0.3)

    prompt_template = f"""
    Write a {summary_type} summary of the following text.        
    Add bullet points and pragrpaphs wherever needed. 
    Add bold texts and headers wherever needed.

    TEXT: "{{text}}"

    {summary_type} SUMMARY:"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    # Summarize
    summarize_chain = load_summarize_chain(llm=llm, chain_type="refine", verbose=True, question_prompt=PROMPT)
    summary = summarize_chain.run(chunks)

    return summary


def main():
    st.title("YouTube Transcript Summarizer")
    
    # Get OpenAI API Key
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    
    # Select Summary Type
    summary_type = st.sidebar.selectbox("Summary Type", ["Concise", "Detailed"])

    # Get YouTube URL
    youtube_url = st.text_input("YouTube URL")

    # Check if API Key and URL are provided
    if not openai_api_key or not youtube_url:
        st.warning("Please enter the OpenAI API Key and YouTube URL before proceeding.")
        return
    
    # Submit Button
    if st.button("Submit"):
        summary = process_transcript(openai_api_key, youtube_url, summary_type)
        
        if summary:
            # Display summary
            st.subheader("Summary")
            st.write(summary)

        # Clear API Key
        openai_api_key = ""
    
if __name__ == "__main__":
    main()