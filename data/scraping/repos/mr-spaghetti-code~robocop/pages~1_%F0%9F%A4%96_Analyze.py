import os
import tempfile
import streamlit as st

from langchain.document_loaders import GitLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake

st.set_page_config(page_title="Summarize Codebase", page_icon="🤖")

st.markdown("# Summarize Codebase")

st.write(
  """First, we have to load all the code from the repo you are investigating."""
)

github_url = st.text_input(label="Enter the URL of a _public_ GitHub repo")

commit_branch = st.text_input(label="Enter the commit ID or branch (Default: main)",
  value="main")

dataset_name = st.text_input(
    label="Dataset name to save into DeepLake (eg. uniswap-v3)"
)

if "settings_override" not in st.session_state:
    st.session_state["settings_override"] = False


os.environ['OPENAI_API_KEY'] = st.session_state["openai_api_key"] if st.session_state["settings_override"] else st.secrets.openai_api_key
os.environ['ACTIVELOOP_TOKEN'] = st.session_state["activeloop_api_key"] if st.session_state["settings_override"] else st.secrets.activeloop_api_key

with st.expander("Advanced settings"):
    filter_extension = st.text_input(
        label="Filter files based on their extension (Default: .sol)",
        value=".sol"
    )
    chunk_size = st.text_input(
        label="(advanced) Chunk size - in tokens - for embedding computation (default: 1000)",
        value="1000"
    )



def load_text(clone_url):
  loader = GitLoader(
      clone_url=clone_url,
      repo_path=tmpdirname,
      branch=commit_branch,
      file_filter = lambda file_path: file_path.endswith(filter_extension)
  )
  data = loader.load()
  print(data[0])
  text_splitter = CharacterTextSplitter(chunk_size=int(chunk_size), chunk_overlap=20)
  texts = text_splitter.split_documents(data)
  return texts


def compute_embeddings(texts):
  dataset_path = f'hub://mrspaghetticode/{dataset_name}'
  embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')
  DeepLake.from_documents(texts, embeddings, dataset_path=dataset_path)
  return dataset_name


if st.button("Analyze"):
  status = st.info(f'Pulling from {github_url}', icon="ℹ️")
  if not github_url or not commit_branch or not dataset_name:
    status.warning("Make sure you fill in all the fields above.")
  else:
    with st.spinner('Processing...'):
      with tempfile.TemporaryDirectory() as tmpdirname:
        print('created temporary directory', tmpdirname)

        status.info("Loading data")

        texts = load_text(clone_url=github_url)

        status.info("Generating embeddings")

        dataset_path = compute_embeddings(texts)

        status.success(f"Done! Sent the data to {dataset_path}")
        st.balloons()
