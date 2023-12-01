import os
import tempfile
import streamlit as st
import pinecone
from langchain.llms.openai import OpenAI
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks import get_openai_callback
import yaml

st.title('GPT-4 + Document Embedding')

# Get OpenAI API key, Pinecone API key and environment, and source document input
openai_api_key = st.sidebar.text_input("OpenAI API Key")
pinecone_api_key = st.sidebar.text_input("Pinecone API Key")

st.sidebar.write("Variables")
with st.sidebar.expander("GPT-3.5-turbo"):
    st.write("Prompt Cost per Token: $0.002")
with st.sidebar.expander("GPT-4-8k"):
    st.write("Prompt Cost per Token: $0.003")
with st.sidebar.expander("GPT-4-32k"):
    st.write("Prompt Cost per Token: $0.006")
with st.sidebar.expander("WhatsApp"):
    st.write("Utility:")
    st.write("- Conversation: $0.02")
    st.write("- Message: $0.005")
    st.write("Service:")
    st.write("- Conversation: $0.0022")
    st.write("- Message: $0.005")

pinecone_env = 'eu-west1-gcp'
pinecone_index = 'newtesting'

col1, col2 = st.columns(2)

source_doc = col2.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
query = col1.text_area("What is your question?", value="Summarise this document",height=100)

# Twilio Integration
twilio_whatsapp_api_per_message = 0.005
twilio_integration = st.checkbox("Twilio WhatsApp Integration")

col1, col2 = st.columns(2)
number_of_interactions = col1.number_input("Number of Interactions", value=1, min_value=1, max_value=20)
average_time_per_interaction = col2.number_input("Average Time per Interaction (minutes)", value=1, min_value=1, max_value=3, step=1)

if 'queries_per_hour' not in st.session_state:
    st.session_state['queries_per_hour'] = 10

if st.button("Estimate Cost"):
    if not openai_api_key or not pinecone_api_key or not pinecone_env or not pinecone_index or not source_doc or not query:
        st.warning(f"Please check document or keys are provided.")
    else:
        try:
            # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(source_doc.read())
            loader = PyPDFLoader(tmp_file.name)
            pages = loader.load_and_split()
            os.remove(tmp_file.name)

            # Generate embeddings for the pages, insert into Pinecone vector database, and expose the index in a retriever interface
            pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            vectordb = Pinecone.from_documents(pages, embeddings, index_name=pinecone_index)
            retriever = vectordb.as_retriever()

            # Initialize the OpenAI module, load and run the Retrieval Q&A chain
            llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
            qa = RetrievalQA.from_chain_type(
                llm, chain_type="stuff", retriever=retriever)

            with get_openai_callback() as cb:
                response = qa.run(query)

                total_cost = cb.total_cost
                total_tokens = cb.total_tokens

                st.dataframe({
                    "Total Cost ($)": total_cost,
                    "Total Tokens": total_tokens,
                })

            pinecone_cost_per_hour = 0.111  # change this based on your config
            aws_cost_per_hour = 1.5
            total_cost_per_hour = (total_cost) + (pinecone_cost_per_hour) + aws_cost_per_hour

            total_services_costs = pinecone_cost_per_hour + aws_cost_per_hour

            if twilio_integration:
                total_cost += twilio_whatsapp_api_per_message


            st.header("Cost Breakdown")
            st.write(f"Total Cost for Document Embedding: ${total_cost:.2f}")

            # per minute
            st.write(f"Total Cost for Document Embedding with AWS + Pinecone per minute: ${(total_cost_per_hour + total_services_costs)/60:.2f}")


            # st.success(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")


