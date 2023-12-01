import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings   
from langchain.vectorstores import Chroma
import pinecone
import os

##from dotenv import load_dotenv, find_dotenv
##oad_dotenv(find_dotenv(),override=True)

def  load_document(file):
    import os
    name , extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'loading file {file}')
        loader= PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'loading file {file}')
        loader= Docx2txtLoader(file) 
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        print(f'loading file {file}')
        loader= TextLoader(file)
    else:
        print('file extension not supported')
        return None
    data =loader.load()
    return data


def chunk_data(data,chunk_size=256,chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks



### EMBEDDING and UPLOADING TO PINECONE
def insert_or_fetch_embeddings(chunks,index_name):
    import pinecone
    from langchain.vectorstores import Pinecone
    from langchain.embeddings import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()
    pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"),environment=os.environ.get("PINECONE_ENV"))
    pinecone.create_index(name=index_name, dimension=1536,metric='cosine', shards=1)
    vector_store = Pinecone.from_documents(chunks,embeddings, index_name=index_name)
    return vector_store

##index_name = 'askhdocument'
##vector_store = insert_or_fetch_embeddings(index_name)    

def ask_and_get_answer(vector_store,query,k=5):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(model='gpt-3.5-turbo',temperature=1,max_tokens=1024)
    retriever = vector_store.as_retriever(search_type='similarity' , search_kwargs={'k':k})
    chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever)

    answer = chain.run(query)
    return answer

def print_embedding_cost(texts):
    import  tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum(len(enc.encode(page.page_content)) for page in texts)
    return total_tokens * 0.0000001

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(),override=True)

   ### st.image('Emtel_FF_Stacked_Red_Black.png',width=200)
    st.subheader('EMTEL FAQ')

    with st.sidebar:

        uploaded_file = st.file_uploader("Choose a file", type=['pdf','docx','txt'])    
        chunk_size = st.number_input('Chunk size',min_value=100,max_value=2048,value=512)
        k=st.number_input('k',min_value=1,max_value=10,value=5)
        add_data=st.button('Add data')

        if uploaded_file and add_data:
            with st.spinner("Reading , chunking and embedding file"):
                bytes_data = uploaded_file.read()
                file_name=os.path.join('./',uploaded_file.name)
                with open(file_name,'wb') as f:
                    f.write(bytes_data)
                data=load_document(file_name)
                chunks=chunk_data(data,chunk_size=chunk_size)
                st.write(f'Number of chunks {len(chunks)}')

                vector_store = insert_or_fetch_embeddings(chunks,'testingindex')
                st.session_state.vs =vector_store
                st.success('Done')


    q = st.text_input('Ask a question')
    if q:
        if 'vs' in st.session_state:
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer= ask_and_get_answer(vector_store,q,k)
            st.text_area('Answer',value=answer)
