# Import os to set API key
import os
# Import OpenAI as main LLM service
from langchain.llms import OpenAI
# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders...there's other ones as well!
from langchain.document_loaders import PyPDFLoader
# Import chroma as the vector store
from langchain.vectorstores import Chroma


# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

# Set APIkey for OpenAI Service
# Can sub this out for other LLM providers
os.environ['OPENAI_API_KEY'] = ''

# Create instance of OpenAI LLM
llm = OpenAI(temperature=0.1, verbose=True)

def process_pdf(book_filename):
    # Create and load PDF Loader
    loader = PyPDFLoader(book_filename)
    # Split pages from pdf
    pages = loader.load_and_split()
    # Load documents into vector database aka ChromaDB
    store = Chroma.from_documents(pages, collection_name='user_book')

    # Create vectorstore info object - metadata repo?
    vectorstore_info = VectorStoreInfo(
        name="user_book",
        description="a user's uploaded book as a pdf",
        vectorstore=store
    )
    # Convert the document store into a langchain toolkit
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

    # Add the toolkit to an end-to-end LC
    agent_executor = create_vectorstore_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )

    return agent_executor, store

st.title('🦜🔗 GPT Book')
st.subheader('Upload seu livro em PDF')

uploaded_file = st.file_uploader("escolha o arquivo", type=['pdf'])

if uploaded_file is not None:
    with st.spinner('Processando...'):
        book_filename = f'user_uploaded_book.pdf'
        with open(book_filename, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        agent_executor, store = process_pdf(book_filename)
        st.success('Livro processado com sucesso!')

    # Create a text input box for the user
    prompt = st.text_input('Escreva seu prompt aqui')

    # If the user hits enter
    if prompt:
        # Then pass the prompt to the LLM
        response = agent_executor.run(prompt)
        # ...and write it out to the screen
        st.write(response)

        # With a streamlit expander
        with st.expander('Pesquisa de Similaridade de Documentos'):
            # Find the relevant pages
            search = store.similarity_search_with_score(prompt)
            # Write out the first
            st.write(search[0][0].page_content)
else:
    st.warning('Por favor upload um documento em formato PDF.')
