import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from modify import OpenAIEmbeddings
import streamlit as st
# manage and delete directories
import shutil

# Check if the pdfs and db directories exist, if not, create them
if not os.path.exists("pdfs"):
    os.makedirs("pdfs")

if not os.path.exists("db"):
    os.makedirs("db")

def load_vector_databases(pdf_directory):
    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
    pdf_indexes = []
    embedding = OpenAIEmbeddings()

    for pdf_file in pdf_files:
        index_name = os.path.splitext(os.path.basename(pdf_file))[0]
        persist_directory = f'db/{index_name}'

        if not os.path.exists(persist_directory):
            print(f"Creating index for {pdf_file}...")
            loader = PyPDFLoader(pdf_file)
            pages = loader.load_and_split()
            vector_db = Chroma.from_documents(documents=pages, embedding=embedding, persist_directory=persist_directory)
            vector_db.persist()

        pdf_indexes.append({
            "name": index_name,
            "directory": persist_directory
        })

    return pdf_indexes

def load_indexes(indexes_to_query, embedding):
    if len(indexes_to_query) > 1:
        combined_index_name = '-'.join(sorted([index["name"] for index in indexes_to_query]))
        combined_persist_directory = f'db/{combined_index_name}'
        if not os.path.exists(combined_persist_directory):
            print(f"Creating combined index for {', '.join([index['name'] for index in indexes_to_query])}...")
            combined_docs = []
            for index_to_query in indexes_to_query:
                vector_db = Chroma(persist_directory=index_to_query["directory"], embedding_function=embedding)
                loader = PyPDFLoader(os.path.join("pdfs", f"{index_to_query['name']}.pdf"))
                docs = loader.load_and_split()
                combined_docs.extend(docs)
            combined_vector_db = Chroma.from_documents(documents=combined_docs, embedding=embedding, persist_directory=combined_persist_directory)
            combined_vector_db.persist()
        else:
            print(f"Loading combined index {combined_index_name}...")
            combined_vector_db = Chroma(persist_directory=combined_persist_directory, embedding_function=embedding)
        return combined_vector_db
    else:
        print(f"Loading index {indexes_to_query[0]['name']}...")
        vector_db = Chroma(persist_directory=indexes_to_query[0]["directory"], embedding_function=embedding)
        return vector_db


def run_query(query, vector_db, include_resources=False):
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=vector_db.as_retriever(), return_source_documents=include_resources)
    response = qa({"query": query})
    return response
    






st.set_page_config(page_title="Talk to multiple PDFs in every combination", layout="wide")
st.title("Talk to multiple PDFs in every combination :sunglasses:")

# OpenAI API Key
api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")
# api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.warning("Please enter your OpenAI API key.")
else:
    os.environ["OPENAI_API_KEY"] = api_key

    # Document uploader
    uploaded_files = st.sidebar.file_uploader("Upload PDF documents", accept_multiple_files=True, type=["pdf"])
    if uploaded_files:
        with st.spinner("Uploading documents..."):
            for uploaded_file in uploaded_files:
                with open(os.path.join("pdfs", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Uploaded {uploaded_file.name}")

    # Load indexes
    pdf_directory = "pdfs"
    pdf_indexes = load_vector_databases(pdf_directory)
    embedding = OpenAIEmbeddings()


    # Delete PDF files
    st.sidebar.header("Delete PDF Files")
    pdf_files = [f for f in os.listdir(pdf_directory) if f.lower().endswith(".pdf")]
    selected_pdf_to_delete = st.sidebar.selectbox("Choose a PDF file to delete", ["None"] + pdf_files, key="selected_pdf_to_delete")
    if selected_pdf_to_delete != "None":
        if st.sidebar.button("Delete PDF"):
            pdf_path = os.path.join(pdf_directory, selected_pdf_to_delete)
            os.remove(pdf_path)
            st.success(f"Deleted PDF {selected_pdf_to_delete}")
            # rerun the app to update the list
            st.experimental_rerun()

    # Delete Indexes
    st.sidebar.header("Delete Indexes", help="If you don't want individual PDF file index to be recreated then delete the PDF file first. refresh the page to see newly created indexes.")

    # Create a set to store index names
    index_names_set = set()

    # Add individual indexes to the set
    individual_index_names = [index["name"] for index in pdf_indexes]
    index_names_set.update(individual_index_names)

    # Add combined indexes to the set
    combined_index_directories = [os.path.join("db", d) for d in os.listdir("db") if os.path.isdir(os.path.join("db", d))]
    combined_index_names = [os.path.basename(d) for d in combined_index_directories if '-' in os.path.basename(d)]
    index_names_set.update(combined_index_names)

    # Convert the set to a list
    index_names = list(index_names_set)

    selected_index_to_delete = st.sidebar.selectbox("Choose an index to delete", ["None"] + index_names, key="selected_index_to_delete")
    if selected_index_to_delete != "None":
        if st.sidebar.button("Delete Index"):
            # Check if selected index is an individual index
            if selected_index_to_delete in individual_index_names:
                index_directory = [index["directory"] for index in pdf_indexes if index["name"] == selected_index_to_delete][0]
                shutil.rmtree(index_directory)
                st.success(f"Deleted index {selected_index_to_delete}")
                pdf_indexes = [index for index in pdf_indexes if index["name"] != selected_index_to_delete]
            # Check if selected index is a combined index
            elif selected_index_to_delete in combined_index_names:
                index_directory = os.path.join("db", selected_index_to_delete)
                shutil.rmtree(index_directory)
                st.success(f"Deleted combined index {selected_index_to_delete}")
                # rerun the app to update the list
                st.experimental_rerun()




# Document uploader
# uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type=["pdf"])
# if uploaded_files:
#     for uploaded_file in uploaded_files:
#         with open(os.path.join("pdfs", uploaded_file.name), "wb") as f:
#             f.write(uploaded_file.getbuffer())
#         st.success(f"Uploaded {uploaded_file.name}")

# Load indexes
pdf_directory = "pdfs"
pdf_indexes = load_vector_databases(pdf_directory)
embedding = OpenAIEmbeddings()

# ... (Load combined indexes)

# Select indexes to query
st.header("Select PDFs to talk to")
# Load the index list
pdf_indexes = load_vector_databases(pdf_directory)

# Create a set to store index names
index_names_set = set()

# Add individual index names to the set
individual_index_names = [index["name"] for index in pdf_indexes]
index_names_set.update(individual_index_names)

# Add combined index names to the set
combined_index_directories = [os.path.join("db", d) for d in os.listdir("db") if os.path.isdir(os.path.join("db", d))]
combined_index_names = [os.path.basename(d) for d in combined_index_directories if '-' in os.path.basename(d)]
index_names_set.update(combined_index_names)

# Convert the set to a list
all_index_names = list(index_names_set)

# Use the combined list in the multiselect component
selected_indexes = st.multiselect("Choose indexes", all_index_names, key="selected_indexes")


if not selected_indexes:
    st.warning("Please select at least one index.")
else:
    indexes_to_query = [index for index in pdf_indexes if index["name"] in selected_indexes]
    if indexes_to_query == []:
        indexes_to_query.append({
            "name": selected_indexes[0],
            "directory": os.path.join("db", selected_indexes[0])
        })
    
    with st.spinner("Loading indexes..."):
        vector_dbs = load_indexes(indexes_to_query, embedding)

    # Include resources
    include_resources = st.checkbox("Include resource documents in the response")

    # User query
    st.header("What is your question?")
    user_query = st.text_input("Query")
    if not user_query:
        st.warning("Please enter a query.")
    else:
        # Run query and display results
        with st.spinner("Searching..."):
            response = run_query(user_query, vector_dbs, include_resources)

        st.header("Search response")
        st.write(response['result'])

        if include_resources:
            st.header("Resources")
            st.write(response['source_documents'])

