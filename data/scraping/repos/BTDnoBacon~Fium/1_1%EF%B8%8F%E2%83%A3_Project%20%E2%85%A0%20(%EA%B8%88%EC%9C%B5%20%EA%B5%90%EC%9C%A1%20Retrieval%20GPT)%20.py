import re
import os
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from pytube import Playlist
from youtube_transcript_api import YouTubeTranscriptApi
import streamlit as st
from st_chat_message import message
#from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)




# //데이터 추출============================================
def extract_title_and_content(url):
    # Sending a GET request to the URL
    response = requests.get(url)

    # Parsing the HTML content of the page with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Using CSS selectors to extract the title and content
    title_element = soup.find("dd", class_="fc1").find("strong")
    content_element = soup.find("div", class_="scrollY", attrs={"tabindex": "0"})

    # If title_element exists, get the text
    if title_element:
        title = title_element.get_text(strip=True)
        forbidden_chars = r'\\/:*?"<>|'

        title = ''.join(char for char in title if char not in forbidden_chars)
    else:
        title = ""

    # If content_element exists, get the text
    if content_element:
        content = content_element.get_text(strip=True)
    else:
        content = "Content not found"

    return title, content

def load_single_document(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()[0]

def load_documents(source_dir):
    all_files = os.listdir(source_dir)
    return [load_single_document(f"{source_dir}/{file_name}") for file_name in all_files]

def get_response_from_query(vector_db, query, target, k):
    docs = vector_db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=1)

    # Template to use for the system message prompt
    template = """
        You are a helpful assistant that that can ALL answer or explain  to {target}.
        Document retrieved from your DB : {docs}

        Answer the questions referring to the documents which you Retrieved from DB as much as possible.
        If you feel like you don't have enough information to answer the question, say "I don't know".

        Since your answer targets {target}, you should return an answer that is optimized for understanding by {target}.
        """

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    # Human question prompt
    human_template = "Answer the following question IN KOREAN: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=query, docs=docs_page_content, target=target)
    response = response.replace("\n", "")
    return response, docs
#=========================================================



# //맞춤형 DB 생성============================================
def Crawling_DB_Child():
  # Start and end ids of '21일간의 경제여행(21편)'
  child_1_start_id = 165557
  child_1_end_id = 165577

  # Start and end ids of '마법소년 재민이의 경제세상 적응기(11편)'
  child_2_start_id = 10000017
  child_2_end_id = 10000027

  # Start and end ids of '초등학생을 위한 알기 쉬운 경제이야기'
  child_3_id = 236250

  # Base URL
  child_base_url = 'https://www.bok.or.kr/portal/bbs/B0000216/view.do?nttId={id}&type=CHILD&searchOptn8=01&menuNo=200646&listType=G&searchOptn4=CHILD&pageIndex=1'

  # List to store all URLs
  child_vod_url_list = []
  child_pdf_url_list = []

  # [어린이] - '21일간의 경제여행(21편)'
  for id in range(child_1_start_id, child_1_end_id + 1):
      # Inserting id into the base url
      url = child_base_url.format(id=id)
      # Adding url to the list
      child_vod_url_list.append(url)

  # [어린이] - '마법소년 재민이의 경제세상 적응기(11편)'
  for id in range(child_2_start_id, child_2_end_id + 1):
      # Inserting id into the base url
      url = child_base_url.format(id=id)
      # Adding url to the list
      child_vod_url_list.append(url)

  # [어린이] - '초등학생을 위한 알기 쉬운 경제이야기'
  child_3_url = child_base_url.format(id=child_3_id)
  child_pdf_url_list.append(child_3_url)

  # Ensure "child" directory exists
  if not os.path.exists("DB/text/Child"):
      os.makedirs("DB/text/Child")


  # [어린이]DB : VOD -> txt 파일
  for url in child_vod_url_list:
      # Extract title and content
      title, content = extract_title_and_content(url)

      # Save the content in a text file with the title as the name
      with open(f"DB/text/Child/{title}.txt", 'w', encoding='utf-8') as f:
          f.write(content)

  # [어린이]DB : PDF -> txt 파일
  for url in child_pdf_url_list:
      # URL of the webpage
      webpage_url = 'https://www.bok.or.kr/portal/bbs/B0000216/view.do?nttId=236250&type=CHILD&searchOptn8=22&menuNo=200646&listType=G&searchOptn4=CHILD&pageIndex=1'

      # Base URL
      base_url = 'https://www.bok.or.kr'

      # Send a GET request to the webpage URL
      response = requests.get(webpage_url)
      soup = BeautifulSoup(response.content, 'html.parser')

      # Find the URL of the PDF file
      pdf_url_suffix = soup.find('div', {'class': 'addfile'})
      pdf_url_suffix = pdf_url_suffix.find('a')['href']
      pdf_url = base_url + pdf_url_suffix
      
      # Send a GET request to the PDF URL
      response = requests.get(pdf_url)

      # Save the PDF in a file
      with open("DB/text/Child/Economic_Story.pdf", 'wb') as f:
          f.write(response.content)

      reader = PdfReader("DB/text/Child/Economic_Story.pdf")
      number_of_pages = len(reader.pages)

      pdf_to_text=''
      for i in range(15,number_of_pages):
        page = reader.pages[i]
        extract_text = page.extract_text()
        pdf_to_text += extract_text.strip()
      pdf_to_text = pdf_to_text.replace('\n', ' ')

      with open('DB/text/Child/초등학생을 위한 알기 쉬운 경제이야기.txt', 'w', encoding='utf-8') as f:
          f.write(pdf_to_text)

      # 원본 pdf 삭제
      if os.path.exists("DB/text/Child/Economic_Story.pdf"):
        os.remove('DB/text/Child/Economic_Story.pdf')

def Crawling_DB_Student():
  # this fixes the empty playlist.videos list
  playlist = Playlist('https://www.youtube.com/playlist?list=PL80z1RKB1KmwsvqBiDLspG9YqXu3Xz_ON')
  playlist._video_regex = re.compile(r"\"url\":\"(/watch\?v=[\w-]*)")

  # Ensure "student" directory exists
  if not os.path.exists("DB/text/Student"):
      os.makedirs("DB/text/Student")

  for url in playlist.video_urls:
      yt = requests.get(url)
      yt_text = BeautifulSoup(yt.text, 'lxml')
      title = yt_text.select_one('meta[itemprop="name"][content]')['content']
      title = title.replace(":", "_")
      
      url_id = url.split('v=')[-1]
      try:
        transcript_list = YouTubeTranscriptApi.get_transcript(url_id,languages=['ko'])
        srt =[transcript['text'] for transcript in transcript_list]
        transcript = ''.join(srt)

        # Save the content in a text file with the title as the name
        with open(f"DB/text/Student/{title}.txt", 'w', encoding='utf-8') as f:
            f.write(transcript)

      except:
        continue


def Crawling_DB_Adult():

  adult_list=['https://www.youtube.com/playlist?list=PL80z1RKB1Kmy-LMsm1MRR4NKiPeI6R14R','https://www.youtube.com/playlist?list=PL80z1RKB1KmymSwpImyjMR4fsUKP1Z9WH']

  for adult_url in adult_list:
    # this fixes the empty playlist.videos list
    playlist = Playlist(adult_url)
    playlist._video_regex = re.compile(r"\"url\":\"(/watch\?v=[\w-]*)")

    # Ensure "student" directory exists
    if not os.path.exists("DB/text/Adult"):
        os.makedirs("DB/text/Adult")

    for url in playlist.video_urls:
        yt = requests.get(url)
        yt_text = BeautifulSoup(yt.text, 'lxml')
        title = yt_text.select_one('meta[itemprop="name"][content]')['content']

        url_id = url.split('v=')[-1]
        try:
          transcript_list = YouTubeTranscriptApi.get_transcript(url_id,languages=['ko'])
          srt =[transcript['text'] for transcript in transcript_list]
          transcript = ''.join(srt)

          # Save the content in a text file with the title as the name
          with open(f"DB/text/Adult/{title}.txt", 'w', encoding='utf-8') as f:
              f.write(transcript)

        except:
          continue
#=========================================================



# //설정창=============================================
def init(): # Web App 설정

    st.set_page_config(
        page_title="SAFFY 금융/경제 지식교육 GPT"
    )


def init_db(): # [어린이/청소년/성인] 맞춤형 VectorDB 구축

    # //Text DB 구축====================================
    Crawling_DB_Child()
    print("어린이용 금융/경제 DB 구축 완료")

    Crawling_DB_Student()
    print("청소년용 금융/경제 DB 구축 완료")

    Crawling_DB_Adult()
    print("성인용 금융/경제 DB 구축 완료")
    # #===================================================


    # # //Vector DB 구축==================================
    if not os.path.exists(f"DB/vector"):
        os.makedirs(f"DB/vector")

    # # //init_db()함수만을 호출 할때 함수 내에서 openai_api_key지정===========
    embedding = OpenAIEmbeddings(openai_api_key='발급받은 OPENAI API Key를 입력해주세요')
    
    list_en=['Child','Student','Adult']
    list_kr=['어린이','학생','성인']

    for level_en,level_kr in zip(list_en,list_kr):   
        file_path = 'DB/text/' + level_en
        transcript = load_documents(file_path)
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(transcript)

        db = FAISS.from_documents(docs, embedding)
        db.save_local(f"DB/vector/{level_en}")
        print(f"{level_kr} VectorDB 구축 완료")
    #===================================================
#=========================================================




def finance_gpt(user_name,user_input,refer_db):
    st.header('')
    k = int(refer_db[0])
    
    # 질문 입력시,
    with st.container():
        # 사용자 질문 GUI 표시
        message(user_input,is_user=True)
        st.subheader('')

        # LLM , Embedding 세팅
        embedding = OpenAIEmbeddings()
        level_dict = {'어린이':'Child','청소년':'Student','성인':'Adult'}

        level_kr = user_name
        level_en = level_dict[level_kr]

        st.subheader(" ")
        st.subheader(f'{level_kr} 맞춤 답변')
        with st.spinner(f"{level_kr} 맞춤형 답변 생성중..."):
            vector_db = FAISS.load_local(f"DB/vector/{level_en}",embedding)
            response, docs = get_response_from_query(vector_db, user_input, level_en, k)

        # GPT 답변
        message(response, is_user=False)
        st.title('')
        st.title('')


        st.subheader(f'{level_kr} 맞춤 답변 참고 문헌 ({refer_db})')
        doc_name_list = [d.metadata['source'].split("/")[-1] for d in docs]
        doc_content_list = [d.page_content for d in docs]

        for i in range(k):
            idx = i+1
            st.text('')
            st.text(f'참고 문헌 {idx}.')
            with st.expander(f'{doc_name_list[i]}'):
                st.info(doc_content_list[i])
            st.header('')

        


# Web App 실행 함수
def PJT1():
    init()

    # 메인 화면 GUI
    st.title("SSAFY PJT I")
    st.subheader(" : 금융/경제 지식교육 RetrievalGPT")
    
    with st.sidebar:
        st.header('사용자 정보 입력')
        st.text('')

        user_name = st.selectbox("🎯 교육 대상", ('','어린이','청소년','성인',))
        st.caption('')

        finance_db = st.selectbox("💰 금융/경제 지식 DB", ('','한국은행'))
        st.caption('')

        refer_db = st.selectbox("📚 참고문헌 건수", ('','3건','4건','5건',))
        st.caption('')

        st.header('챗봇모델 정보 입력')
        st.text('')

        
        chatgpt_api = st.text_input('ChatGPT API Key:', type='password')
        if chatgpt_api:
            st.success('API Key 확인 완료!', icon='✅')
            os.environ["OPENAI_API_KEY"] = chatgpt_api
        else:
            st.warning('API key를 입력하세요.', icon='⚠️')

        st.text('')
        st.text('')
        st.subheader('📋 옵션')
        gpt_visualize = st.checkbox('🤖 챗봇 시작하기')


    st.divider()
    st.title(f"🎯 {user_name} 맞춤 금융/경제 교육")
    st.caption('')
    st.title(" ")
    st.title(" ")
    

    if gpt_visualize:
        with st.form("my_form"):
            user_input = st.text_input('금융/경제 관련 질문', '예시) 금융공부를 해야하는 이유를 알려줘')
            submitted = st.form_submit_button("질문 입력")

        st.divider()
        st.title(" ")
        finance_gpt(user_name,user_input,refer_db)




# # // VectorDB 구축==============================================================
# init_db()   # Text -> VectorDB 구축을 위해 최초 실행 (첨부한 DB.Zip파일로 대체 가능)
# #===============================================================================



# .venv\Scripts\activate.bat
PJT1()