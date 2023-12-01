import os
from langchain.llms import OpenAI
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

os.environ['OPENAI_API_KEY'] = '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'

# Instancie o serviço LLM da OpenAI com a temperatura de amostragem e ativando o modo verbose para debug
llm = OpenAI(temperature=0.1, verbose=True)

def process_pdf(book_filename):
    # Crie um objeto loader para PDF do livro enviado
    loader = PyPDFLoader(book_filename)
    # Divida o arquivo em páginas
    pages = loader.load_and_split()
    # Carregue cada página do livro em uma base de dados vetorial chamada ChromaDB
    store = Chroma.from_documents(pages, collection_name='user_book')

    # Crie um objeto VectorStoreInfo que contém informações sobre a coleção de vetores
    vectorstore_info = VectorStoreInfo(
        name="user_book",
        description="a user's uploaded book as a pdf",
        vectorstore=store
    )
    # Converta a document store (ChromaDB) em um toolkit de LangChain
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

    # Adicione o toolkit de LangChain a um agente de ponta a ponta (end-to-end)
    agent_executor = create_vectorstore_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )

    return agent_executor, store

# Inicialize a aplicação Streamlit
st.title('🔗 Rapha IA GPT Book')
st.subheader('Upload seu livro em PDF')

uploaded_file = st.file_uploader("escolha o arquivo", type=['pdf'])

if uploaded_file is not None:
    with st.spinner('Processando...'):
        # Salve o arquivo temporariamente
        book_filename = f'user_uploaded_book.pdf'
        with open(book_filename, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        # Carregue o arquivo de leitura e extraia suas informações para um ChromaDB e um agente de ponta a ponta para uso posterior
        agent_executor, store = process_pdf(book_filename)
        st.success('Livro processado com sucesso!')

    # Crie uma caixa de entrada de texto para que o usuário possa fazer uma consulta
    prompt = st.text_input('Pesquise aqui:')

    # Se o usuário pressionar Enter
    if prompt:
        # Passe a consulta para o agente de LangChain
        response = agent_executor.run(prompt)
        # Escreva a resposta na tela
        st.write(response)

        # Dentro de um expander, mostre mais detalhes sobre a resposta
        with st.expander('Resposta detalhada'):
            # Encontre as páginas relevantes no ChromaDB
            search = store.similarity_search_with_score(prompt)
            # Escreva o conteúdo da primeira página nos resultados
            st.write(search[0][0].page_content)
else:
    st.warning('Por favor upload um documento em formato PDF.')