# pip install langchain=0.0.142
# pip install openai=0.27.4
# pip install tiktoken=0.3.3
# pip install pinecone-client
# https://blog.futuresmart.ai/building-a-document-based-question-answering-system-with-langchain-pinecone-and-llms-like-gpt-4-and-chatgpt#heading-7-embedding-documents-with-openai
import streamlit as st
import os
import pinecone
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Pinecone
import tiktoken
import re
from components import convert
import pandas as pd
import numpy as np
import time
import requests
# import pprint
import json


title = 'LLM Learning'
st.set_page_config(page_title=title, page_icon="🐍", layout="wide")
st.title(title)

if convert.check_password() == False:
    st.stop()

os.environ['OPENAI_API_KEY'] = st.secrets["api_key"]

# tab1, tab2, tab3, tab4 , tab5 = st.tabs(
#     ["학습(txt)_Pinecone", 
#     "전체목록_Pinecone",
#     "🗃 학습(txt)_Chroma",
#     "전체목록_Chroma",
#     "학습(csv)_FAISS"]
#     )

tab3, tab4, tab1, tab2, tab5 = st.tabs(
    [
        "🗃 학습(txt)_Chroma",
        "전체목록_Chroma",
        "학습(txt)_Pinecone", 
        "전체목록_Pinecone",
        "학습(csv)_FAISS"
        ]
    )

# llm_learn_type = st.radio(label = 'select llm Learning', options = ['Pinecone', 'Chroma', 'FAISS'], index=0)
# st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


#########################################################################
### 학습(txt)_Chroma #####################################################
#########################################################################
with tab3:
    # print(f'start tab3 {tab3}')
    st.write(""" ### 🧐 전체 목록 보기 """)
    # persist_directory="db"
    # embedding = OpenAIEmbeddings()
    
    readAll_button = st.button("Read All Data From Chroma Local DB", key="readAllChroma", type='secondary')
    if readAll_button:
        persist_directory="db"
        embedding = OpenAIEmbeddings()
        chromadb = Chroma(
        embedding_function=embedding,
        persist_directory=persist_directory)
        ids_df = pd.DataFrame(chromadb.get())
        st.dataframe(
            data=ids_df,
            # height=1000,
            width=1200,
            hide_index=False,
        )
        chromadb.persist()
        chromadb = None

    st.write("""  """)
    st.write(""" ### 🎃 txt 파일 학습하기 """)
    # file_glob = 'DTSM-IR-203_008.txt'
    file_glob = 'DTSM-PU*'

    # txt_chroma_button = st.button(f"Read {file_glob}", key="txtChroma", type='secondary')
    # if txt_chroma_button:
    loader = DirectoryLoader('./sources', glob=file_glob, loader_cls=TextLoader)
    documents_chroma = loader.load()

    def num_tokens_from_string(string: str, encoding_name: str) -> int:  
        """Returns the number of tokens in a text string."""  
        encoding = tiktoken.get_encoding(encoding_name)  
        num_tokens = len(encoding.encode(string))  
        return num_tokens
    # st.write(f'{num_tokens_from_string(documents[0].page_content, encoding_name="cl100k_base")} 토근이 예상됩니다')

    # 텍스트를 청크(chunk) 단위로 분할하기
    chunk_size = 1000
    text_chroma_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=20,
        add_start_index = True,
        )
    texts_chroma = text_chroma_splitter.split_documents(documents_chroma)
    st.info(f""" 
            {len(documents_chroma)}개의 문서에 포함된 {len(documents_chroma[0].page_content)}개의 단어를 {chunk_size} 청크 단위로 {len(texts_chroma)}개의 문서로 분할 하였습니다.

            
            """)
    # {num_tokens_from_string(documents[0].page_content, encoding_name="cl100k_base")} 토근이 예상됩니다.

    # 업로드 텍스트 Dataframe 형식으로 확인하기
    texts_chroma_df = pd.DataFrame()
    for text_chroma in texts_chroma :
        text_chroma_df = pd.DataFrame({'page_content': [text_chroma.page_content], 'metadata': [text_chroma.metadata]})
        texts_chroma_df = pd.concat([texts_chroma_df, text_chroma_df])
    texts_chroma_df.reset_index(drop=True, inplace=True)
    st.dataframe(
        data=texts_chroma_df,
        # height=1000,
        width=1200,
        hide_index=False,
    )

    upsert_button = st.button("Upsert to Chroma Local DB", key="txtUpsertChroma", type='primary')
    if upsert_button :
        # persist_directory="/content/drive/My Drive/Colab Notebooks/chroma/romeo"
        persist_directory="db"
        # os.environ['OPENAI_API_KEY'] = st.secrets["api_key"]
        embedding = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(
            documents=texts_chroma,
            embedding=embedding, 
            persist_directory=persist_directory)  
        vectordb.persist()
        vectordb = None
        st.info(f""" 
        ### 업로드를 완료하였습니다.
        #### Chroma
        [https://docs.trychroma.com/getting-started/](https://docs.trychroma.com/getting-started/)
        """)

#########################################################################
### 학습(txt)_Pinecone ###################################################
#########################################################################
with tab1:
    # print(f'start tab1 {tab1}')
    st.header("학습(txt)_Pinecone")
    pinecone.init(api_key=f"{st.secrets['api_pine']}", environment='gcp-starter')
    index_name = 'dwlangchain'
    # st.write('pinecone.list_indexes()')
    # st.write(pinecone.list_indexes())
    # loader = DirectoryLoader('./sources', glob='*.txt', loader_cls=TextLoader)

    st.write("""  """)
    st.write(""" ### 🎃 txt 파일 학습하기 """)
    # file_glob = 'DTSM-IR-203_009.txt'
    # txt_pinecone_button = st.button(f"Read {file_glob}", key="txtPinecone", type='secondary')
    # if txt_pinecone_button:
    loader_pinecone = DirectoryLoader('./sources', glob='DTSM-IR-203_009.txt', loader_cls=TextLoader)
    documents_pinecone = loader_pinecone.load()
    # text 정제
    output = []
    # https://study-easy-coding.tistory.com/67
    for page_pinecone in documents_pinecone:
        text = page_pinecone.page_content
        text = re.sub(r'(\w+)-\n((\w+))', r'\1\2', text) # 안녕-\n하세요 -> 안녕하세요
        text = re.sub(r'(?<!\n\s)\n(?!\s\n)', ' ' , text.strip()) # "인\n공\n\n지능팩토리 -> 인공지능팩토리
        text = re.sub(r'\n\s*\n', '\n\n' , text) # "\n버\n\n거\n\n킹\n => 버\n거\n킹
        delete_word = re.sub(r'[\(]+[\w\s]+[\)]+','',text) #스트링 병기문자를 삭제하고 그 값을 replace_word에 스트링으로 담김
        output.append(text)
    # st.write(documents)
    # st.write(output)

    # 텍스트를 청크(chunk) 단위로 분할하기
    chunk_size = 1000
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
    texts = text_splitter.split_documents(documents_pinecone)

    st.info(f""" 
            {len(documents_pinecone)}개의 문서에 포함된 {len(documents_pinecone[0].page_content)}개의 단어를 {chunk_size} 청크 단위로 {len(texts)}개의 문서로 분할 하였습니다.

            
            """)
    texts_df = pd.DataFrame()
    for text in texts :
        text_df = pd.DataFrame({'page_content': [text.page_content], 'metadata': [text.metadata]})
        texts_df = pd.concat([texts_df, text_df])
    texts_df.reset_index(drop=True, inplace=True)
    st.dataframe(
        data=texts_df,
        # height=1000,
        width=1200,
        hide_index=False,
    )

    upsert_button = st.button("Upsert to Pinecone DB", key="upsertPinecone", type='primary')
    if upsert_button:
        # index = pinecone.Index("dwlangchain")
        index_name = 'dwlangchain'
        embedding = OpenAIEmbeddings()
        index = Pinecone.from_documents(texts, embedding, index_name=index_name)
        st.info(f""" 
        ### 업로드를 완료하였습니다.
        #### Pinecone
        [https://app.pinecone.io/](https://app.pinecone.io/).
        """)

#########################################################################
### 전체목록_Pinecone #####################################################
#########################################################################
with tab2:
    # print(f'start tab2 {tab2}')
    pinecone.init(api_key=f"{st.secrets['api_pine']}", environment='gcp-starter')
    index_name = 'dwlangchain'

    st.info(f""" 
        ###### Pinecone
        [https://app.pinecone.io/](https://app.pinecone.io/)
        """)
    
    def all_ids_data():
        indexquery = pinecone.Index(index_name)
        namespace = ''
        if len(indexquery.describe_index_stats()["namespaces"]) == 0:
            st.write('데이터가 존재하지 않습니다.')
            st.stop()
        num_vectors = indexquery.describe_index_stats()["namespaces"][namespace]['vector_count']
        st.write(num_vectors)

        num_dimensions = 1536
        all_ids_df = pd.DataFrame()
        while len(all_ids_df) < num_vectors:
            input_vector = np.random.rand(num_dimensions).tolist()
            results = indexquery.query(
                vector=input_vector, 
                top_k=10000,
                include_values=False,
                include_metadata=True
                )
            for result in results['matches']:
                ids_df = pd.DataFrame([[result['id'], result['metadata']['text'], result['metadata']['source']]])
                all_ids_df = pd.concat([all_ids_df, ids_df])
        all_ids_df.reset_index(drop=True, inplace=True)
        all_ids_df.columns = ['id', 'text', 'source']
        return all_ids_df

    with st.spinner('Wait for it...'):
        all_ids_df = all_ids_data().sort_values(by=['text'], ascending=True)
        st.write(all_ids_df)

        # 벡터 중복값 제거
        delete_dup_ids_button = st.button("중복값 제거", key="deleteadup", type='primary')
        if delete_dup_ids_button:
            all_ids_df = all_ids_data().sort_values(by=['text'], ascending=True)
            dup = all_ids_df.duplicated(['text'], keep='first')
            all_dup_df = pd.concat([all_ids_df, dup], axis=1)
            all_dup_df.rename(columns = {0 : 'dup'}, inplace = True)
            st.write(all_dup_df)
            remain_first_df = all_dup_df[all_dup_df['dup'] == True]['id'].values.tolist()
            st.write(remain_first_df)
            indexquery = pinecone.Index(index_name)
            indexquery.delete(ids=remain_first_df, namespace='')
            st.info(f""" 중복값을 제거하였습니다. """)  
            time.sleep(2)
        
        # 벡터 전체값 제거
        delete_all_ids_button = st.button("전체값 제거", key="deleteall", type='primary')
        if delete_all_ids_button:
            all_ids_df = all_ids_data()
            all_ids = all_ids_df['id'].values.tolist()
            indexquery = pinecone.Index(index_name)
            indexquery.delete(ids=all_ids, namespace='')
            st.info(f""" 전체 데이터를 제거하였습니다. """)
            time.sleep(2)

#########################################################################
### 전체목록_Chroma #######################################################
#########################################################################
with tab4:
    # cli = chromadb.Client()
    # client = chromadb.PersistentClient(path='http://scal.daewooenc.com:8501/flydw/chroma/')
    # client = chromadb.HttpClient(host="http://scal.daewooenc.com:8501/chroma/", port=8501)
    # client = chromadb.HttpClient(host="http:/wooenc.com:8501/chroma/", port=8501)
    # st.write(client)
    # collection2 = client.list_collections()
    # st.write(collection2)

    # 불러오기
    persist_directory="db"
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(
        embedding_function=embedding, 
        persist_directory=persist_directory)  
    
    ids_df = pd.DataFrame(vectordb.get())
    st.dataframe(
        data=ids_df,
        height=2000,
        width=2000,
        hide_index=False,
        column_config={
            # "documents": st.column_config.LinkColumn(
            #     # "Trending apps",
            #     # help="The top trending Streamlit apps",
            #     # validate="^https://[a-z]+\.streamlit\.app$",
            #     max_chars=150,
            # ),
            "documents": st.column_config.TextColumn(
                # "본문 내용", 이름 변경
                width=900,
                # help="Streamlit **widget** commands 🎈",
                # default="st.",
                # max_chars=500,
                # validate="^st\.[a-z_]+$",
            ),
            "widgets": st.column_config.Column(
                width='large'
            )
        }

    )

    # Using embedded DuckDB with persistence: data will be stored in: ./chroma_db
    # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])
    # 7580

    # retriever = vectordb.as_retriever()
    # retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    # docs = retriever.get_relevant_documents("퇴사자 수 알려줘")

    # for doc in docs:
    #     st.write(doc.metadata["source"])

# st.write(pinecone.list_indexes())
# st.write(index)

# Upsert sample data (5 8-dimensional vectors)
# index.upsert([
#     ("A", [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
#     ("E", [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
# ])

# index_name = pinecone.Index("dwlangchain")


# import openai

# # get api key from platform.openai.com
# openai.api_key = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'

# embed_model = "text-embedding-ada-002"
# query = (
#     "Which training method should I use for sentence transformers when " +
#     "I only have pairs of related sentences?"
# )
# res = openai.Embedding.create(
#     input=[query],
#     engine=embed_model
# )

# # retrieve from Pinecone
# xq = res['data'][0]['embedding']
# # get relevant contexts (including the questions)
# res = index.query(xq, top_k=2, include_metadata=True)

#########################################################################
### 학습(csv)_FAISS ######################################################
#########################################################################
with tab5:
    st.stop()
    from langchain.document_loaders.csv_loader import CSVLoader
    from langchain.vectorstores import FAISS
    from langchain.prompts import PromptTemplate
    from langchain.chat_models import ChatOpenAI
    from langchain.chains import LLMChain
    # loader = DirectoryLoader('./sources', glob='DTSM-PU-310_002.txt', loader_cls=TextLoader)
    loader = CSVLoader(file_path="./sources/sales_response.csv")
    documents = loader.load()

    st.write(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(documents, embeddings)


    def retrieve_info(query):
        similar_response = db.similarity_search(query, k=3)
        page_contents_array = [doc.page_content for doc in similar_response]
        # print(page_contents_array)
        return page_contents_array

    # 3. Setup LLMChain & prompts
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

    template = """
    You are a world class business development representative. 
    I will share a prospect's message with you and you will give me the best answer that 
    I should send to this prospect based on past best practies, 
    and you will follow ALL of the rules below:

    1/ Response should be very similar or even identical to the past best practies, 
    in terms of length, ton of voice, logical arguments and other details

    2/ If the best practice are irrelevant, then try to mimic the style of the best practice to prospect's message

    Below is a message I received from the prospect:
    {message}

    Here is a list of best practies of how we normally respond to prospect in similar scenarios:
    {best_practice}

    Please write the best response that I should send to this prospect:
    """

    prompt = PromptTemplate(
        input_variables=["message", "best_practice"],
        template=template
    )
    st.write(prompt)
    chain = LLMChain(llm=llm, prompt=prompt)


    # 4. Retrieval augmented generation
    def generate_response(message):
        best_practice = retrieve_info(message)
        response = chain.run(message=message, best_practice=best_practice)
        return response
    st.write('5712')
    message = st.text_area("customer message")

    if message:
        st.write("Generating best practice message...")

        result = generate_response(message)

        st.info(result)