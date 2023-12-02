import streamlit as st
import openai
from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index import VectorStoreIndex
import os 

# Set API Key
openai.api_key = os.getenv("OPENAI_API_KEY")


def load_data_and_index():
    filename_fn = lambda filename: {'file_name': filename}
    documents = SimpleDirectoryReader('opiod_vb/LIB_TEXT', file_metadata=filename_fn).load_data()
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine(max_nodes=6, max_tokens=500)  # Move the query engine creation here
    return index, query_engine


def get_response(user_query, query_engine):  # Updated parameter
    response = query_engine.query(user_query)
    return response

def main():
    st.title("Opiod V_DB Search Engine")
    user_query = st.text_input("Enter your question:")

    index, query_engine = load_data_and_index()  # Unpack both the index and the query engine

    if user_query:
        response = get_response(user_query, query_engine)  # Pass in the query engine
        # Extract just the text response
        response_text = response.response
        filenames = [node.node.metadata['file_name'].split('/')[-1].split('.')[0] for node in response.source_nodes]
        
        st.write(response_text)
        st.write("Sources:", ', '.join(filenames))
    else:
        st.write("Put new Questions into the Text box and press Enter. Delete old one before entering new one.")

if __name__ == "__main__":
    main()
