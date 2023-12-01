import os
import streamlit as st
from dotenv import load_dotenv
from llama_index import GPTVectorStoreIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain.llms.openai import OpenAI
from biorxiv_manager import BioRxivManager 

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

st.title("Ask BioRxiv")
query = st.text_input("What would you like to ask? (source: BioRxiv files)", "")

@st.cache_data
def fetch_and_parse():
    # instantiating BioRxivManager runtime and fetch the parsed nodes
    manager = BioRxivManager()
    return manager.fetch_and_parse(interval="2023-07-01/2023-07-30")

embedded_documents = fetch_and_parse()

if st.button("Submit"):
    if not query.strip():
        st.error(f"Please provide the search query.")
    else:
        try:
            llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-4-32k", openai_api_key=openai_api_key))
            max_input_size = 32767
            num_output = 400
            chunk_overlap_ratio = 0.2  # Adjust this value according to your need.

            prompt_helper = PromptHelper(max_input_size, num_output, chunk_overlap_ratio)

            service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
            index = GPTVectorStoreIndex.from_documents(embedded_documents, service_context=service_context)
            
            response = index.query(query)
            st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
