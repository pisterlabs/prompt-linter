import os
import openai
import pypdf
import streamlit as st
from streamlit_chat import message
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ChatVectorDBChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

#  QA template and general prompt 
system_template="""Use the following pieces of context to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
----------------
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

@st.cache_data
def split_pdf(fpath,chunk_chars=4000,overlap=50):
    """
    Pre-process PDF into chunks
    Some code from: https://github.com/whitead/paper-qa/blob/main/paperqa/readers.py
    """
    st.info("`Reading and splitting doc ...`")
    pdfReader = pypdf.PdfReader(fpath)
    splits = []
    split = ""
    pages = []
    for i, page in enumerate(pdfReader.pages):
        pages.append(str(i + 1))
        split += page.extract_text()
        if len(split) > chunk_chars:
            splits.append(split[:chunk_chars])
            split = split[chunk_chars - overlap:]
    return splits

@st.cache_resource
def create_ix(splits,_general_prompt):
    """ 
    Create vector DB index of PDF w/ new qa chain and chat history
    """
    st.info("`Building index ...`")
    embeddings = OpenAIEmbeddings()
    ix = FAISS.from_texts(splits,embeddings)
    # Use ChatGPT with index QA chain
    llm = ChatOpenAI(temperature=0)
    qa = ChatVectorDBChain.from_llm(llm,ix,qa_prompt=_general_prompt)
    chat_history = []
    return qa, chat_history

# Auth
st.sidebar.image("Img/reading.jpg")
api_key = st.sidebar.text_input("`OpenAI API Key:`", type="password")
st.sidebar.write("`By:` [@RLanceMartin](https://twitter.com/RLanceMartin)")
os.environ["OPENAI_API_KEY"] = api_key
chunk_chars = st.sidebar.radio("`Choose chunk size for splitting`", (2000, 3000, 4000), index=1)
st.sidebar.info("`Larger chunk size can produce better answers, but may high ChatGPT context limit (4096 tokens)`")

# App 
st.header("`doc-gpt-chatbot`")
st.info("`Hello! I am a ChatGPT connected to whatever document you upload.`")
uploaded_file_pdf = st.file_uploader("`Upload PDF File:` ", type = ['pdf'] , accept_multiple_files=False)

# Re-set history w/ new doc
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

if uploaded_file_pdf and api_key:

    # Split text and create index
    d=split_pdf(uploaded_file_pdf,chunk_chars)
    qa,chat_history=create_ix(d,prompt)
    
    # Query
    query = st.text_input("`Please ask a question:` ","What is this document about?")

    # Run
    try:

        # Get response
        result = qa({"question": query, "chat_history": chat_history})
        output = result["answer"]

        # Update history
        chat_history.append((query, output))
        st.session_state.past.append(query)
        st.session_state.generated.append(output)

        # Visualize chat history 
        if st.session_state['generated']:
            for i in range(len(st.session_state['generated'])-1, -1, -1):
                message(st.session_state["generated"][i], key=str(i),avatar_style="bottts",seed=130)  
                message(st.session_state['past'][i], is_user=True, key=str(i) + '_user',avatar_style="pixel-art",seed=124) 

    except openai.error.InvalidRequestError:
        # 4096 token ChatGPT context length https://github.com/acheong08/ChatGPT/discussions/649
        st.warning('Error with model request, often due to context length. Try reducing chunk size.', icon="⚠️")

else:
    st.info("`Please enter OpenAI Key and upload pdf file`")