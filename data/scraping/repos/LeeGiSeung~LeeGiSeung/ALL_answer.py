import http.server
import socketserver
import json
from pymongo import MongoClient
import requests
import threading
import gensim
import numpy as np
import os
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import preprocess_string
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from konlpy.tag import Komoran
from gensim.models import CoherenceModel
from sklearn.metrics.pairwise import cosine_similarity
import pymongo
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from pymongo.cursor import CursorType
from selenium import webdriver    # 웹 브라우저 자동화
import time                       # 시간 지연
from tqdm import tqdm_notebook    # 진행상황 표시
from selenium.webdriver.common.by import By
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from konlpy.tag import Kkma
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from urllib.parse import quote_plus
from bs4 import BeautifulSoup
from selenium import webdriver
from googleapiclient.discovery import build



chrome_options = Options()
chrome_options.add_argument("--headless")  # Headless 모드 활성화
chrome_options.add_argument("--disable-gpu")  # GPU 가속 비활성화 (필요한 경우)
komoran = Komoran()
kkma = Kkma()
recommend = " 추천"
# MongoDB 연결
client = MongoClient('localhost', 27017)
db = client['VScodeDB']
collection = db['DemoDatabase']

class MyRequestHandler(http.server.BaseHTTPRequestHandler):
    def _set_response(self, status_code=200):
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_GET(self):
        self._set_response()
        name = self.path.split('/')[-1]
        result = collection.find_one({'name': name})
        if result:
            self.wfile.write(json.dumps(result).encode('utf-8'))
        else:
            self.wfile.write(json.dumps({'message': 'Data not found'}).encode('utf-8'))

    def do_POST(self):
        self._set_response()
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))
        collection.insert_one(data)
        self.wfile.write(json.dumps({'message': 'Data inserted successfully'}).encode('utf-8'))

def run(server_class=socketserver.ThreadingTCPServer, handler_class=MyRequestHandler, port=8080):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on port {port}\n")

    # 서버를 별도의 스레드에서 실행
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.start()

# 불용어 목록을 읽어오는 함수
def load_stopwords(file_path):
    stopwords = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            word = line.strip()  # 개행 문자 제거
            stopwords.append(word)
    return stopwords

# 입력 받은 문장 토픽 구하기
def extract_nouns(text):
    nouns = kkma.nouns(text)
    return nouns

def process_and_store_data(sentence_input, user_input):
    # 형태소 분석 및 품사 태깅
    pos_tags = kkma.pos(sentence_input)

    # 명사(Noun)와 형용사(Adjective)만 추출
    select_token = [word for word, pos in pos_tags if pos in ['NNG', 'NNP', 'VA', 'VV']]
    filtered_words = [word for word in select_token if not (word.isdigit() or len(word) == 1)]
    print("스탑워드전",filtered_words)
    filtered_words = [word for word in filtered_words if word not in stopwords]
    print("스탑워드후",filtered_words)

    dictionary = corpora.Dictionary([filtered_words])

    # TF-IDF 행렬 생성
    tfidf_vectorizer = TfidfVectorizer(vocabulary=dictionary.token2id)
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_words)

    # CSR 포맷으로 변환
    tfidf_matrix_csr = csr_matrix(tfidf_matrix.transpose())

    # CSR 포맷의 행렬을 Gensim Corpus 형식으로 변환하여 MmCorpus 생성
    tfidf_corpus = gensim.matutils.Sparse2Corpus(tfidf_matrix_csr)

    # 토픽 모델링 실행
    num_topics = 1
    lda_model = LdaModel(corpus=tfidf_corpus, num_topics=num_topics, id2word=dictionary, passes=1)
    # 
    # 토픽과 각 토픽의 단어 및 확률 출력
    topics = lda_model.print_topics(num_topics=num_topics, num_words=2)
    for topic in topics:
        print(topic)

    # id : 데이터 넣을때 유저 id값 (유니크 키)
    # original_data : 원본 입력 문장
    # LAD_data : 분석한 LDA값 LDA_data가 맞는데 오타로 LAD_data로 넣음
    # satisfaction : 만족스러운지 불만족 스러운지 키는 추가했는데 만족/불만족 여부 수정하는 기능은 추가 안함
    # emotion : 윤진이 감성 분석 결과 저장

    topic_dic = {
        'id': user_input,
        'original_data': sentence_input,
        'LAD_data': topics,
        'satisfaction': 0,
        'emotion' : 0
    }

    tokenized_documents = [doc.split() for doc in filtered_words]

    # LDA 모델을 사용하여 일관성 점수 계산
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_documents, dictionary=dictionary, coherence='c_v', processes=1)

    # 일관성 점수 계산
    coherence_score = coherence_model_lda.get_coherence()
    print("Coherence Score:", coherence_score)

    return topic_dic

# 지식인 답변 검색 토픽 구하기
def process_intellectual(setntence_input):
    pos_tags = kkma.pos(setntence_input)

    # 명사(Noun)와 형용사(Adjective)만 추출
    select_token = [word for word, pos in pos_tags if pos in ['NNG', 'NNP', 'VA', 'VV']]
    filtered_words = [word for word in select_token if not (word.isdigit() or len(word) == 1)]
    # print("스탑워드전",filtered_words)
    filtered_words = [word for word in filtered_words if word not in stopwords]
    print("스탑워드후",filtered_words)

    dictionary = corpora.Dictionary([filtered_words])

    # TF-IDF 행렬 생성
    tfidf_vectorizer = TfidfVectorizer(vocabulary=dictionary.token2id)
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_words)

    # CSR 포맷으로 변환
    tfidf_matrix_csr = csr_matrix(tfidf_matrix.transpose())

    # CSR 포맷의 행렬을 Gensim Corpus 형식으로 변환하여 MmCorpus 생성
    tfidf_corpus = gensim.matutils.Sparse2Corpus(tfidf_matrix_csr)

    # 토픽 모델링 실행
    num_topics = 1
    lda_model = LdaModel(corpus=tfidf_corpus, num_topics=num_topics, id2word=dictionary, passes=1)

    # 토픽과 각 토픽의 단어 및 확률 출력
    topics = lda_model.print_topics(num_topics=num_topics, num_words=2)
    for topic in topics:
        print(topic)

    tokenized_documents = [doc.split() for doc in filtered_words]

    # LDA 모델을 사용하여 일관성 점수 계산
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_documents, dictionary=dictionary, coherence='c_v', processes=1)

    # 일관성 점수 계산
    coherence_score = coherence_model_lda.get_coherence()
    print("Coherence Score:", coherence_score)

    return topics

# 지식인(text), 구글 답변 검색
def Find_Answer(insert_keyword):
    import re

    ### Step 1. 크롤링할 블로그 url 수집
    # 검색어
    # 만약 검색했는데 검색된 글이 아무것도 없다면 오류가 나옴
    print("########################")
    print("입력 문장 : ",insert_keyword)
    print("########################")
    keyword1 = process_intellectual(insert_keyword)
    # 크롬 웹브라우저 실행
    ch_driver = r"C:\Users\82103\Desktop\LSA\chromedriver-win64\chromedriver-win64\chromedriver"

    # 검색창에 '검색어' 검색


    filtered_keyword = re.sub(r'[^a-zA-Z가-힣\s]', '', str(keyword1))
    print("#########################")
    print("키워드 검색",filtered_keyword)
    print("#########################")

        # YouTube API 클라이언트 생성
    api_key = "AIzaSyC5jGsgnjqntDIiOFn-QjKwnIrvymlLuz8"
    api_service_name = "youtube"
    api_version = "v3"

    youtube = build(api_service_name, api_version, developerKey=api_key)

    # 검색어 설정
    search_query = filtered_keyword + recommend  # 원하는 검색어로 교체하세요

    # 검색 결과 가져오기
    search_response = youtube.search().list(
        q=search_query,
        type="video",
        part="id",
        maxResults=1  # 맨 위의 동영상 하나만 가져올 경우
    ).execute()

    # 첫 번째 동영상 정보 가져오기
    search_results = search_response.get("items", [])
    if len(search_results) > 0:
        video = search_results[0]
        video_id = video["id"]["videoId"]
        
        # 동영상 정보 가져오기
        video_response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()
        
        # 썸네일 이미지, 제목, URL 추출
        video_snippet = video_response["items"][0]["snippet"]
        video_thumbnail = video_snippet["thumbnails"]["high"]["url"]
        video_title = video_snippet["title"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"

    blog_driver = webdriver.Chrome(ch_driver, options=chrome_options)

    # 사이트 주소
    blog_driver.get("http://www.naver.com")
    time.sleep(2)

    # 검색창에 '검색어' 검색

    element = blog_driver.find_element(By.ID, "query")
    element.send_keys(filtered_keyword+recommend)
    element.submit()
    time.sleep(1)

    blog_driver.find_element(By.LINK_TEXT, "VIEW").click()

    # '블로그' 클릭

    blog_driver.find_element(By.LINK_TEXT, "블로그").click()
    time.sleep(1)

    # 스크롤 다운
    def scroll_down(driver):
        driver.execute_script("window.scrollTo(0, 99999999)")
        time.sleep(1)

    # n: 스크롤할 횟수 설정
    n = 0
    i = 0
    while i < n:
        scroll_down(blog_driver)
        i = i+1

    # 블로그 글 url들 수집
    url_list = []
    title_list = []

    article_raw = blog_driver.find_elements(By.CLASS_NAME, "api_txt_lines.total_tit")

    # 크롤링한 url 정제 시작
    for article in article_raw:
        url = article.get_attribute('href')   
        url_list.append(url)

    time.sleep(1)
        
    # 제목 크롤링 시작    
    for article in article_raw:
        title = article.text
        title_list.append(title)

    google_url = 'https://www.google.com/search?q=' + quote_plus(filtered_keyword+recommend)

    # chromedriver path input
    google_driver = r"C:\Users\82103\Desktop\LSA\chromedriver-win64\chromedriver-win64\chromedriver"
        # option
    google_driver = webdriver.Chrome(google_driver,options=chrome_options)

    google_driver.get(google_url)
    google_driver.implicitly_wait(10)

    html = google_driver.page_source
    soup = BeautifulSoup(html)

    v = soup.select('.yuRUbf')
    google_driver.close()

    # option
    driver = webdriver.Chrome(ch_driver,options=chrome_options)

    # 사이트 주소
    driver.get("http://www.naver.com")
    time.sleep(2)
    element = driver.find_element(By.ID, "query")
    if isinstance(filtered_keyword, str):
        element.send_keys(filtered_keyword+recommend)
    
    element.submit()
    time.sleep(1)

    # 'VIEW' 클릭
    driver.find_element(By.LINK_TEXT, "지식iN").click()
    
    try:
        # '옵션' 클릭
        driver.find_element(By.LINK_TEXT, "옵션").click()
        time.sleep(1)

        # # '전체' 클릭
        # driver.find_element(By.LINK_TEXT, "1개월").click()
        # time.sleep(0.5)
                
        # '전체' 클릭
        driver.find_element(By.LINK_TEXT, "지존이상").click()
        time.sleep(0.5)

        # 스크롤 다운
        def scroll_down(driver):
            driver.execute_script("window.scrollTo(0, 99999999)")
            time.sleep(1)
    
        # n: 스크롤할 횟수 설정
        n = 0
        i = 0
        while i < n:
            scroll_down(driver)
            i = i+1

        # 블로그 글 url들 수집
        url_list = []
        title_list = []
        article_raw = driver.find_elements(By.CLASS_NAME, "question_text")

        # 크롤링한 url 정제 시작
        for article in article_raw:
            url = article.get_attribute('href')   
            url_list.append(url)

        time.sleep(1)
            
        # 제목 크롤링 시작    
        for article in article_raw:
            title = article.text
            title_list.append(title)

        ### Step 2. 블로그 내용 크롤링
        import sys
        import os
        import pandas as pd
        import numpy as np

        num_list = len(url_list)

        # print("url리스트 갯수 : ",num_list)

        data_dict = {}    # 전체 크롤링 데이터를 담을 그릇

        number = num_list    # 수집할 글 갯수
        answer_result = []
        # 수집한 url 돌면서 데이터 수집
        for i in range(min(3, len(url_list))):
            # 글 띄우기
            # print("순서",i)
            url = url_list[i]
            # options=chrome_options
            driver = webdriver.Chrome(r"C:\Users\82103\Desktop\LSA\chromedriver-win64\chromedriver-win64\chromedriver",options=chrome_options)
            driver.get(url)   # 글 띄우기
            
        # 크롤링

                # 질문 제목 크롤링
            try:
                overlays = "div.c-heading__title .title"                        
                tit = driver.find_element(By.CSS_SELECTOR, overlays)
                title = tit.text
                # print("질문 제목 크롤링",title)
                # print()
            except:
                print("양식에 맞지않아 통과합니다.")

                # 질문 내용 크롤링
            try:
                overlays = ".c-heading__content"                                 
                cont = driver.find_element(By.CSS_SELECTOR, overlays)
                content = cont.text
                # print("질문 내용 : ",content)
                # print()
            except:
                print("질문 내용은 없습니다.")

                # 답변 크롤링

            try:    
                overlays = ".c-heading-answer__content"    
                ans = driver.find_element(By.CSS_SELECTOR, overlays)
                answer = ans.text
                # print("답변",answer)
                answer_result.append(answer)

            except:
                print("답변 내용은 없습니다.")
            try:
                overlays = "total_thumb"
            #     # CSS 선택자를 사용하여 이미지 요소를 찾습니다.
                image_element = driver.find_element(By.CLASS_NAME, overlays)
                image_url = image_element.get_attribute("href")
            except:
                print("이미지가 없습니다.")

        try:
            parser = PlaintextParser.from_string(answer_result[0], Tokenizer("korean"))
            summarizer = LsaSummarizer()
            if(len(answer_result[0])<200):
                summary = summarizer(parser.document, sentences_count=3)  # 요약문에서 추출할 문장 수를 지정할 수 있습니다.
            else:
                summary = summarizer(parser.document, sentences_count=5)
            print("[추천]")
            print("####################")
            for sentence in summary:   
                print(sentence)
        except:
            print("지식인 추천이 없습니다.")
    except:
        print("글이 없습니다.")
    print()
    print("[블로그 추천]")
    print(title_list[0])
    print(url_list[0])
    print()
    print("[유튜브 추천]")
    print("썸네일 이미지 URL:", video_thumbnail)
    print("제목:", video_title)
    print("URL:", video_url)
    print()
    print("[구글 추천]")
    print(v[0].select_one('.LC20lb.DKV0Md').text)
    print(v[0].a.attrs['href'])
    print("####################")
    print()
    driver.close() 

# 유사도 계산
def similarity(setntence_input):
    ori = []

    result = collection.find({}, {"original_data": 1, "_id": 0}).sort([("$natural", pymongo.DESCENDING)]).skip(1).limit(14) 
    ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
    new_setntence = setntence_input
    print("유사도 분석 문장",new_setntence)
    for document in result:
        original_data = document.get("original_data")
        ori.append(original_data)

    sim_result = []

    for i in range(min(14, len(ori))):
        # 이 부분으로 기존 문장들을 벡터화 시킴
        # 코사인 유사도 계산할때는 무조건 벡터화 해야함

        X = ngram_vectorizer.fit_transform([new_setntence, ori[i]])
        similarity = cosine_similarity(X[0], X[1])
        sim_result.append({"new" : new_setntence,"ori" : ori[i],"sim" : similarity[0][0]})

        # print()
        # print("신규 문장 : ", new_setntence)
        # print("기존 문장 : ", ori[i])
        # print("두 문장의 유사도 : ", similarity[0][0])
        # print() 
        # key = lambda x: x['sim'] sim이라는 키에 키 선택 우선순위가 'sim'이 됨
        # reverse 뒤집다. 원래는 s::-1 이라는 식으로 뒤집에서 정렬하는데 그냥 reverse=True을 사용해서 뒤집은 다음에 정렬함 이게 더 좋은듯?
        sim_result_sorted = sorted(sim_result, key=lambda x: x['sim'], reverse=True)

    if sim_result_sorted:
        most_similar_pair = sim_result_sorted[0]
        if(most_similar_pair["sim"]>0.7):
            print("###################")
            print("가장 유사한 문장")
            print()
            print("신규 문장:", most_similar_pair["new"])
            print()
            print("기존 문장:", most_similar_pair["ori"])
            print()
            print("코사인 유사도:", most_similar_pair["sim"])
            print("###################")
            print()
            print("기존 글과 비교해 가장 유사한 일기의 추천을 재추천 드리겠습니다.")
            Find_Answer(most_similar_pair["ori"])
            print()
        else:
            print("유사한 문장이 없습니다.")

def satisfaction(setntence_input):
    # 형태소 분석 및 품사 태깅
    pos_tags = kkma.pos(setntence_input)

    # 명사(Noun)와 형용사(Adjective)만 추출
    select_token = [word for word, pos in pos_tags if pos in ['NNG', 'NNP', 'VA', 'VV']]
    filtered_words = [word for word in select_token if not (word.isdigit() or len(word) == 1)]
    # print("스탑워드전",filtered_words)
    filtered_words = [word for word in filtered_words if word not in stopwords]
    print("스탑워드후",filtered_words)

    dictionary = corpora.Dictionary([filtered_words])

    # TF-IDF 행렬 생성
    tfidf_vectorizer = TfidfVectorizer(vocabulary=dictionary.token2id)
    tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_words)

    # CSR 포맷으로 변환
    tfidf_matrix_csr = csr_matrix(tfidf_matrix.transpose())

    # CSR 포맷의 행렬을 Gensim Corpus 형식으로 변환하여 MmCorpus 생성
    tfidf_corpus = gensim.matutils.Sparse2Corpus(tfidf_matrix_csr)

    # 토픽 모델링 실행
    num_topics = 2
    lda_model = LdaModel(corpus=tfidf_corpus, num_topics=num_topics, id2word=dictionary, passes=1)

    # 토픽과 각 토픽의 단어 및 확률 출력
    topics = lda_model.print_topics(num_topics=num_topics, num_words=2)
    for topic in topics:
        print(topic)

    tokenized_documents = [doc.split() for doc in filtered_words]

    # LDA 모델을 사용하여 일관성 점수 계산
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_documents, dictionary=dictionary, coherence='c_v', processes=1)

    # 일관성 점수 계산
    coherence_score = coherence_model_lda.get_coherence()
    print("Coherence Score:", coherence_score)

    for selected_topic in topics:
        Find_satisfatction_Answer(selected_topic)

def Find_satisfatction_Answer(selected_topic):
    import re

    ### Step 1. 크롤링할 블로그 url 수집
    # 검색어
    # 만약 검색했는데 검색된 글이 아무것도 없다면 오류가 나옴
    print("########################")
    print("지식인 검색 : ",selected_topic)
    print("########################")
    # 크롬 웹브라우저 실행
    ch_driver = r"C:\Users\82103\Desktop\LSA\chromedriver-win64\chromedriver-win64\chromedriver"

    driver = webdriver.Chrome(ch_driver,options=chrome_options)

    # 사이트 주소
    driver.get("http://www.naver.com")
    time.sleep(2)

    # 검색창에 '검색어' 검색
    # element = driver.find_element(By.ID, "query")1
    element = driver.find_element(By.ID, "query")

    filtered_keyword = re.sub(r'[^a-zA-Z가-힣\s]', '', str(selected_topic))

    api_key = "AIzaSyC5jGsgnjqntDIiOFn-QjKwnIrvymlLuz8"
    api_service_name = "youtube"
    api_version = "v3"

    youtube = build(api_service_name, api_version, developerKey=api_key)

    # 검색어 설정
    search_query = filtered_keyword + recommend  # 원하는 검색어로 교체하세요

    # 검색 결과 가져오기
    search_response = youtube.search().list(
        q=search_query,
        type="video",
        part="id",
        maxResults=1  # 맨 위의 동영상 하나만 가져올 경우
    ).execute()

    # 첫 번째 동영상 정보 가져오기
    search_results = search_response.get("items", [])
    if len(search_results) > 0:
        video = search_results[0]
        video_id = video["id"]["videoId"]
        
        # 동영상 정보 가져오기
        video_response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()
        
        # 썸네일 이미지, 제목, URL 추출
        video_snippet = video_response["items"][0]["snippet"]
        video_thumbnail = video_snippet["thumbnails"]["high"]["url"]
        video_title = video_snippet["title"]
        video_url = f"https://www.youtube.com/watch?v={video_id}"
    # print("필터링 지식인 검색 키워드",filtered_keyword+" 추천")
    if isinstance(filtered_keyword, str):
        element.send_keys(filtered_keyword+recommend)

    element.submit()
    blog_driver = webdriver.Chrome(ch_driver, options=chrome_options)

    # 사이트 주소
    blog_driver.get("http://www.naver.com")
    time.sleep(2)

    # 검색창에 '검색어' 검색

    element = blog_driver.find_element(By.ID, "query")
    element.send_keys(filtered_keyword+recommend)
    element.submit()
    time.sleep(1)

    blog_driver.find_element(By.LINK_TEXT, "VIEW").click()

    # '블로그' 클릭

    blog_driver.find_element(By.LINK_TEXT, "블로그").click()
    time.sleep(1)

    # 스크롤 다운
    def scroll_down(driver):
        driver.execute_script("window.scrollTo(0, 99999999)")
        time.sleep(1)

    # n: 스크롤할 횟수 설정
    n = 0
    i = 0
    while i < n:
        scroll_down(blog_driver)
        i = i+1

    # 블로그 글 url들 수집
    url_list = []
    title_list = []

    article_raw = blog_driver.find_elements(By.CLASS_NAME, "api_txt_lines.total_tit")
    # article_raw[0]

    # 크롤링한 url 정제 시작
    for article in article_raw:
        url = article.get_attribute('href')   
        url_list.append(url)

    time.sleep(1)
        
    # 제목 크롤링 시작    
    for article in article_raw:
        title = article.text
        title_list.append(title)

    google_url = 'https://www.google.com/search?q=' + quote_plus(filtered_keyword+recommend)

    # chromedriver path input
    google_driver = r"C:\Users\82103\Desktop\LSA\chromedriver-win64\chromedriver-win64\chromedriver"
        # option
    google_driver = webdriver.Chrome(google_driver,options=chrome_options)

    google_driver.get(google_url)
    google_driver.implicitly_wait(10)

    html = google_driver.page_source
    soup = BeautifulSoup(html)

    v = soup.select('.yuRUbf')

    google_url = 'https://www.google.com/search?q=' + quote_plus(filtered_keyword+recommend)

    # chromedriver path input
    google_driver = r"C:\Users\82103\Desktop\LSA\chromedriver-win64\chromedriver-win64\chromedriver"
        # option
    google_driver = webdriver.Chrome(google_driver,options=chrome_options)

    google_driver.get(google_url)
    google_driver.implicitly_wait(10)

    html = google_driver.page_source
    soup = BeautifulSoup(html)

    v = soup.select('.yuRUbf')

    time.sleep(1)

    # 'VIEW' 클릭
    driver.find_element(By.LINK_TEXT, "지식iN").click()

    # '옵션' 클릭
    driver.find_element(By.LINK_TEXT, "옵션").click()
    time.sleep(1)

    driver.find_element(By.LINK_TEXT, "지존이상").click()
    time.sleep(0.5)


    # 스크롤 다운
    def scroll_down(driver):
        driver.execute_script("window.scrollTo(0, 99999999)")
        time.sleep(1)

    # n: 스크롤할 횟수 설정
    n = 0
    i = 0
    while i < n:
        scroll_down(driver)
        i = i+1

    # 블로그 글 url들 수집
    url_list = []
    title_list = []
    article_raw = driver.find_elements(By.CLASS_NAME, "question_text")

    # 크롤링한 url 정제 시작
    for article in article_raw:
        url = article.get_attribute('href')   
        url_list.append(url)

    time.sleep(1)
        
    # 제목 크롤링 시작    
    for article in article_raw:
        title = article.text
        title_list.append(title)

    ### Step 2. 블로그 내용 크롤링
    import sys
    import os
    import pandas as pd
    import numpy as np

    num_list = len(url_list)
    answer_result = []
    # 수집한 url 돌면서 데이터 수집
    for i in range(min(3, len(url_list))):
        # 글 띄우기
        # print("순서",i)
        url = url_list[i]
        # option
        driver = webdriver.Chrome(r"C:\Users\82103\Desktop\LSA\chromedriver-win64\chromedriver-win64\chromedriver",options=chrome_options)
        driver.get(url)   # 글 띄우기
        
    # 크롤링

            # 질문 제목 크롤링
        try:
            overlays = "div.c-heading__title .title"                        
            tit = driver.find_element(By.CSS_SELECTOR, overlays)
            title = tit.text
            # print("질문 제목 크롤링",title)
            print()
        except:
            print("양식에 맞지않아 통과합니다.")

            # 질문 내용 크롤링
        try:
            overlays = ".c-heading__content"                                 
            cont = driver.find_element(By.CSS_SELECTOR, overlays)
            content = cont.text
            # print("질문 내용 : ",content)
            print()
        except:
            print("질문 내용은 없습니다.")

            # 답변 크롤링

        try:    
            overlays = ".c-heading-answer__content"    
            ans = driver.find_element(By.CSS_SELECTOR, overlays)
            answer = ans.text
            # print("추가 답변",answer)
            answer_result.append(answer)

        except:
            print("답변 내용은 없습니다.")
    try:
        parser = PlaintextParser.from_string(answer_result[0], Tokenizer("korean"))
        summarizer = LsaSummarizer()
        if(len(answer_result[0])<200):
            summary = summarizer(parser.document, sentences_count=3)  # 요약문에서 추출할 문장 수를 지정할 수 있습니다.
        else:
            summary = summarizer(parser.document, sentences_count=5)  # 요약문에서 추출할 문장 수를 지정할 수 있습니다.
        print("[재추천]")
        print("####################")
        for sentence in summary:   
            print(sentence)
        print()
    except:
        print("네이버 추천을 찾지 못했습니다.")
    
    print("[블로그 추천]")
    print(title_list[0])
    print(url_list[0])
    print()
    print("[유튜브 추천]")
    print("썸네일 이미지 URL:", video_thumbnail)
    print("제목:", video_title)
    print("URL:", video_url)
    print()
    print("[구글추천]")
    print(v[0].select_one('.LC20lb.DKV0Md').text)
    print(v[0].a.attrs['href'])
    print("####################") 
    print()
    google_driver.close()
    driver.close() 

def Insert_data():
    user_input = input("유저 번호를 입력하세요:")
    setntence_input = input("문장을 입력하세요: ")
    topic_dic = process_and_store_data(setntence_input, user_input)

    collection.insert_one(topic_dic)
    print("데이터 입력이 완료되었습니다.")
    print()
    similarity(setntence_input)
    Find_Answer(setntence_input)
    satisfaction_input = input("답변이 마음에 드시나요? (예/아니요)")
    if(satisfaction_input == "예"):
        print("만족 하셨다니 다행입니다.")
    else:
        print("다른 추천을 찾아보겠습니다.")
        satisfaction(setntence_input)

def Delete_data():
    # 데이터 삭제
    Delete_input = input("찾고싶은 유저 번호를 입력하세요:")

    result = collection.delete_one({'_id': Delete_input})  #delete_one pymongo 명령어 콜렉션 내 데이터 삭제임
    if result.deleted_count > 0:
        print("데이터를 성공적으로 삭제했습니다.\n")
    else:
        print("데이터를 삭제하는데 실패했습니다.\n")

def Find_data():
    # 데이터 찾기
    find_input = input("찾고싶은 유저 번호를 입력하세요:")
    query = {'id': find_input}
    result = collection.find(query)

    for document in result:
        print("\n######################")
        print("유저 아이디 : "+document['id'])
        print("입력 문장 : "+document['original_data'])
        print("키워드 추출 : "+str(document['LAD_data']))
        print("######################\n")   

if __name__ == '__main__':
    run()
    # 불용어 파일의 경로
    stopwords_file = 'stopwords.txt'

    # 불용어 목록을 읽어옴
    stopwords = load_stopwords(stopwords_file)
    while True:
        print("1 : 데이터 입력")
        print("2 : 데이터 찾기")
        print("3 : 데이터 삭제")
        print("4 : 종료\n")
        input_command = input("원하는 기능을 선택하세요: ")

        if input_command == "1":
            Insert_data()
        elif input_command == "2":
            Find_data()
        elif input_command == "3":
            Delete_data()
        elif input_command == "4":
            print("프로그램을 종료합니다.")
            break
        else:
            print("올바른 기능을 선택하세요.")