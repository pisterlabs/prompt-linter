import streamlit as st
import pandas as pd
import googletrans
import openai
import time
import requests
import urllib.request
import json
import urllib
import ssl
import re
from newspaper import Config
from newspaper import Article
from langdetect import detect
from youtube_transcript_api import YouTubeTranscriptApi
from konlpy.tag import Kkma
from pykospacing import Spacing
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup


#### 코드에서 주의 할 부분
# 로컬변수, 전역변수 이름 동일한거 수정 -> 로컬변수 앞에 _ 언더바 하나나 두개
# 변수가 하는 일에 대한 설명이 되도록 이름 설정 필요


# 프로그램 실행 순서
# 1. 크롤링하고자 하는 URL 입력
# 2. 크롤링 진행
# 3. 크롤링 된 데이터가 영어가 아닌 경우 googletrans로 번역
# 4. 번역된 결과를 chatGPT prompt에 담아서 전송
# (전송형태 : [사용자가 입력한 prompt] + 크롤링 된 en 데이터 )
# 5. 요청 결과 출력 (kr)

ssl._create_default_https_context = ssl._create_unverified_context
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
config = Config()
translator = googletrans.Translator()
final_result = []

# openAPI 개인용 시크릿키
YOUR_API_KEY = '보안상의 이유로 깃허브에는 올리지 않습니다'

#  chatGPT에 명령어를 날리기 위한 함수
def chatGPT(prompt, API_KEY=YOUR_API_KEY):

    # set api key
    openai.api_key = API_KEY

    # Call the chat GPT API
    completion = openai.Completion.create(
        # 'text-curie-001'  # 'text-babbage-001' #'text-ada-001'
        engine='text-davinci-003', prompt=prompt, temperature=0.5, max_tokens=2048, top_p=1, frequency_penalty=0, presence_penalty=0)

    return completion['choices'][0]['text']


# 유튜브 스크립트 크롤링용 함수
def youtube_script_crawling(id):
    st.write("유튜브 스크립트 크롤링 실행중...")

    # 1. 영상 제목 값 가져오기
    # url과 url에 포함된 video id를 통해 제목을 크롤링 해올 동영상 주소에 접근
    params = {"format": "json", "url": "https://www.youtube.com/watch?v=%s" % id}
    url = "https://www.youtube.com/oembed"
    query_string = urllib.parse.urlencode(params)
    url = url + "?" + query_string

    with urllib.request.urlopen(url) as response:
        response_text = response.read()
        data = json.loads(response_text.decode())
        video_title = data['title']
        # 응답받은 값에서 제목 데이터를 가져옴
        st.write("영상 제목 : ", video_title)

    # 2.영상의 자동생성된 자막 가져오기 - 한국어 스크립트만 가능
    # 자동자막을 지원하지 않는 영상인 경우 에러남
    youtube_script = YouTubeTranscriptApi.get_transcript(id, languages=['ko']) 

    text = ''

    for i in range(len(youtube_script)):
        text += youtube_script[i]['text'] + ' '  # text 부분만 가져옴

    # 공백 제거
    text_ = text.replace(' ', '')

    kkma = Kkma()
    text_sentences = kkma.sentences(text_)

    # 문장이 끝나는 어절을 임의로 지정해서 분리할 용도로 사용
    lst = ['죠', '다', '요', '시오', '습니까', '십니까', '됩니까', '옵니까', '뭡니까',]

    # 제외하고자 하는 단어 파일을 읽어들여서 for문에 사용 - 추후 단어데이터 set 추가 필요
    df = pd.read_csv('not2.csv', encoding='utf-8')
    not_verb = df.loc[:, 'col1'].to_list()

    text_all = ' '.join(text_sentences).split(' ')

    for n in range(len(text_all)):
        i = text_all[n]
        if len(i) == 1:  # 한글자일 경우 추가로 작업하지 않음
            continue

        else:
            for j in lst:  # 종결 단어
                # 질문형
                if j in lst[4:]:
                    i += '?'

                # 명령형
                elif j == '시오':
                    i += '!'

                # 마침표
                else:
                    if i in not_verb:  # 특정 단어 제외
                        continue

                    else:
                        if j == i[len(i)-1]:  # 종결
                            text_all[n] += '.'
                            # print("\n", text_all[n], end='/ ')

    # 나뉜 문장을 다시 조인
    spacing = Spacing()
    text_all_in_one = ' '.join(text_all)
    result = spacing(text_all_in_one.replace(' ', ''))
    st.write(result) # 원본 크롤링 데이터 출력

    return result

# 브런치 링크 크롤러
# 구글 검색결과를 크롤링해오는 과정에서 선언한 browser 그대로 사용
# <br> 태그를 기준으로 문단 분리
def brunch_crawler(url):
    st.write("브런치 링크입니다.")

    # 1. url 접속을 위한 playwirght 시작
    playwright = sync_playwright().start() # 크롤링용 플레이라이트 시작
    browser = playwright.chromium.launch(headless=True) 
    page = browser.new_page() # 웹브라우저 열기
    page.goto(url) # 파라미터로 받은 url 주소로 이동
    ctx = page.locator('xpath=/html/body') # 오픈한 페이지에서 가져올 html code path 지정(body로 지정해서 전체 가져옴)
    ctx_inner_html = ctx.inner_html() # 조회한 페이지에서 html 코드 가져오기

    # 브런치 게시글의 제목을 가져오기 위함
    # 제목 태그명을 선택한 뒤 class 이름 붙여서 접근
    title_tag = page.locator('h1.cover_title')
    title_text = title_tag.inner_html().replace("  ","")
    st.write("게시글 제목 : ", title_text)

    # 앞을 잘라내기 위해 브런치 start 태그 지정
    # 해당 태그 다음부터 본문의 내용이 시작된다(제목, 본문)
    under_start_tag_location = ctx_inner_html.find("<div class=\"wrap_body text_align_left finish_txt")

    # 뒷부분을 잘라내기 위해 브런치 end 태그 지정
    # 해당 태그가 위치한 곳부터 내용을 잘라낸다(본문이 끝난 이후 바로 다음에 등장하는 태그로 지정)
    over_end_tag_location = ctx_inner_html.find("<div class=\"wrap_body_info") 

    # 조회한 태그의 위치만큼 html 코드를 잘라내기
    tag_data = ctx_inner_html[under_start_tag_location:over_end_tag_location]

    # re (정규표현식??) 사용해서 원하는 문자열이 전체에서 어느 위치에 있는지 확인
    get_br_lotation_list = [m.start() for m in re.finditer('<br>', str(tag_data))]
    print("<br>태그 위치 확인:",get_br_lotation_list)

    st.write("브런치 게시글의 문단을 분리합니다.")

    # 반복문으로 위의 list를 돌면서, <br> 태그의 위치를 기준으로 텍스트를 잘라냄
    # 자른 데이터를 순수 텍스트로 만들어서 결과 list에 따로 저장
    brunch_iteration_count = 0
    sentence_separation_result_list = []
    for i in range(len(get_br_lotation_list)):
        
        # 맨 처음은 0으로 고정
        print("이전 :",get_br_lotation_list[i-1])
        print("현재 :",get_br_lotation_list[i])

        # 첫 번째 반복인 경우에만
        if brunch_iteration_count == 0:

            present_paragraph_location = get_br_lotation_list[i]

            paragraph_text_data = tag_data[0:present_paragraph_location]
            print("paragraph_text_data:",paragraph_text_data)

            # 잘라낸 html 코드에서 tag 데이터를 지우고 순수 텍스트만 가져오기
            cleantext = BeautifulSoup(paragraph_text_data, "lxml").text
            print(cleantext.replace("\n\n", "")) # 불필요한 공백(엔터) 제거 후 출력확인

            brunch_iteration_count += 1

        # 두번째 반복부터 모두 해당
        else:
            present_paragraph_location = get_br_lotation_list[i]
            previous_paragraph_location = get_br_lotation_list[i-1]

            paragraph_text_data = tag_data[previous_paragraph_location:present_paragraph_location]
            print("paragraph_text_data:",paragraph_text_data)

            # 잘라낸 html 코드에서 tag 데이터를 지우고 순수 텍스트만 가져오기
            cleantext = BeautifulSoup(paragraph_text_data, "lxml").text
            # cleantext = cleantext.replace("\n\n", "") # 불필요한 공백(엔터) 제거
            st.write("문장:",cleantext.replace("\n\n", "")) # 불필요한 공백(엔터) 제거 후 출력확인

            brunch_iteration_count += 1
            
        sentence_separation_result_list.append(str(cleantext))
        sentence_separation_result_list = list(filter(None, sentence_separation_result_list)) # 빈요소가 있으면 제거
        sentence_separation_result_list = [word.strip('\xa0') for word in sentence_separation_result_list ] # 특정 문자열 '\xa0' 제거
        sentence_separation_result_list = [word.strip('\n') for word in sentence_separation_result_list ] # 특정 문자열 '\n' 제거

    # 잘라낸 html 코드에서 tag 데이터를 지우고 순수 텍스트만 가져오기
    # cleantext = BeautifulSoup(tag_data, "lxml").text
    # print(cleantext.replace("\n\n", "")) # 불필요한 공백(엔터) 제거 후 출력확인
    st.write("문단 분리 결과:",sentence_separation_result_list)
    st.write("문단 분리 결과 리스트의 길이:",len(sentence_separation_result_list))

    if len(sentence_separation_result_list) == 0: # 제대로 크롤링되지 않은 경우 라이브러리 실행으로 대체
        st.write("라이브러리로 실행됩니다...")
        sentence_separation_result_list = newspaper_crawler(url)

    browser.close()
    print("-------------------완료------------------")

    

    # 문장이 담긴 리스트를 반환
    return sentence_separation_result_list

def naver_post_crawler(url):
    st.write("네이버포스트 링크입니다.")

    playwright = sync_playwright().start() # 크롤링용 플레이라이트 시작
    browser = playwright.chromium.launch(headless=True) 
    page = browser.new_page() # 웹브라우저 열기
    page.goto(url) # 파라미터로 받은 url 주소로 이동
    ctx = page.locator('xpath=/html/body') # 오픈한 페이지에서 가져올 html code path 지정(body로 지정해서 전체 가져옴)
    ctx_inner_html = ctx.inner_html() # 조회한 페이지에서 html 코드 가져오기

    # 포스트의 제목을 가져오기 위함
    # 제목 태그명을 선택한 뒤 class 이름 붙여서 접근
    title_tag = page.locator('h3.se_textarea')
    title_text = title_tag.inner_html().replace("  ","")
    st.write("게시글 제목 : ", title_text)

    # 앞을 잘라내기 위해 본문이 시작되는 start 태그 지정
    # 해당 태그 다음부터 본문의 내용이 시작된다(본문)
    under_start_tag_location = ctx_inner_html.find("<div class=\"se_component_wrap sect_dsc __se_component_area")

    # 뒷부분을 잘라내기 위해 end 태그 지정
    # 해당 태그가 위치한 곳부터 내용을 잘라낸다(본문이 끝난 이후 바로 다음에 등장하는 태그로 지정)
    over_end_tag_location = ctx_inner_html.find("<div class=\"state_line") 

    # 조회한 태그의 위치만큼 html 코드를 잘라내기
    tag_data = ctx_inner_html[under_start_tag_location:over_end_tag_location]

    print("잘라낸 html : ", tag_data)

    # re (정규표현식??) 사용해서 원하는 문자열이 전체에서 어느 위치에 있는지 확인
    # get_br_lotation_list = [m.start() for m in re.finditer('<br>', tag_data)]
    get_br_lotation_list = [m.start() for m in re.finditer('<p class="se_textarea">', tag_data)]
    print('<p class=\"se_textarea">태그 위치 확인:' ,get_br_lotation_list)

    # 반복문으로 위의 list를 돌면서, <br> 태그의 위치를 기준으로 텍스트를 잘라냄
    # 자른 데이터를 순수 텍스트로 만들어서 결과 list에 따로 저장
    post_iteration_count = 0
    sentence_separation_result_list = []
    for i in range(len(get_br_lotation_list)):

        # 맨 처음은 0으로 고정
        print("이전 :",get_br_lotation_list[i-1])
        print("현재 :",get_br_lotation_list[i])

        # 첫 번째 반복인 경우에만
        if post_iteration_count == 0:

            present_paragraph_location = get_br_lotation_list[i]

            paragraph_text_data = tag_data[0:present_paragraph_location]
            print("paragraph_text_data:",paragraph_text_data)

            # 잘라낸 html 코드에서 tag 데이터를 지우고 순수 텍스트만 가져오기
            cleantext = BeautifulSoup(paragraph_text_data, "lxml").text
            cleantext = cleantext.replace("\n\n", "")
            cleantext = cleantext.replace("\xa0", " ")
            st.write("분리된 문장:", cleantext) # 불필요한 공백(엔터) 제거 후 출력확인

            post_iteration_count += 1

        # 두번째 반복부터 모두 해당
        else:
            present_paragraph_location = get_br_lotation_list[i]
            previous_paragraph_location = get_br_lotation_list[i-1]

            paragraph_text_data = tag_data[previous_paragraph_location:present_paragraph_location]
            print("paragraph_text_data:",paragraph_text_data)

            # 잘라낸 html 코드에서 tag 데이터를 지우고 순수 텍스트만 가져오기
            cleantext = BeautifulSoup(paragraph_text_data, "lxml").text
            cleantext = cleantext.replace("\n\n", "")
            cleantext = cleantext.replace("\xa0", " ")
            st.write("분리된 문장:", cleantext) # 불필요한 공백(엔터) 제거 후 출력확인
            st.write("분리된 문장 길이:", len(cleantext)) # 불필요한 공백(엔터) 제거 후 출력확인

            post_iteration_count += 1
        
        sentence_separation_result_list.append(cleantext)
        sentence_separation_result_list = list(filter(None, sentence_separation_result_list)) # 빈요소가 있으면 제거
        sentence_separation_result_list = [word.strip('\xa0') for word in sentence_separation_result_list ] # 특정 문자열 '\xa0' 제거
        sentence_separation_result_list = [word.strip('\n') for word in sentence_separation_result_list ] # 특정 문자열 '\n' 제거



    # 잘라낸 html 코드에서 tag 데이터를 지우고 순수 텍스트만 가져오기
    # cleantext = BeautifulSoup(tag_data, "lxml").text
    # print(cleantext.replace("\n\n", "")) # 불필요한 공백(엔터) 제거 후 출력확인
    st.write("문단 분리 결과:",sentence_separation_result_list)
    st.write("문단 분리 결과 리스트의 길이:",len(sentence_separation_result_list))
    browser.close()
    print("-------------------완료------------------")

    return sentence_separation_result_list

# 네이버 블로그 - iframe 제거 후 blog.naver.com 붙이기
def naver_blog_delete_iframe(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"}
    res = requests.get(url, headers=headers)
    res.raise_for_status()  # 문제시 프로그램 종료
    soup = BeautifulSoup(res.text, "lxml")

    src_url = "https://blog.naver.com/" + soup.iframe["src"]

    return src_url

# 네이버블로그 본문 스크래핑 - 스마트 에디터 2.0, ONE 포함
def naver_blog_text_scraping(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"}
    res = requests.get(url, headers=headers)
    res.raise_for_status()  # 문제시 프로그램 종료
    soup = BeautifulSoup(res.text, "lxml")

    pragraph_data = ''
    pragraph_data_list = []
    # clean_pragaph_data = ''
    # clean_pragaph_data_list = []
    result_clean_paragraph_list = []

    print("url:",url)

    if soup.find("div", attrs={"class": "se-main-container"}): # 본문 전체 태그
        text = soup.find(
            "div", attrs={"class": "se-main-container"}).get_text()
        text = text.replace("\n\n", "")


        find_span = soup.select('p > span') # 문단 분리를 위해 text 데이터가 들어가는 span 태그 선택해서 가져오기
        print("span_data:::::",soup.select('p > span')) # p태그 안의 span 태그를 선택해서 가져오기
        print(type(find_span)) # 타입은 bs4에서 제공하는 resultset임 -> set이므로 for문을 통해 요소를 가져오기 가능

        # 결과를 담을 list
        result_pragaph_list = []

        # span 데이터 안의 텍스트를 하나씩 출력하기 위한 반복문
        for text_data in find_span:
          clean_span_text = text_data.get_text() # <span> 태그안의 텍스트 데이터 가져오기
          print(text_data.get_text()) 
          print("타입:",type(text_data.get_text())) #str
          print("텍스트 길이:",len(clean_span_text))  # 최소 1
          
          # 한 문자가 아닌 일반 텍스트의 경우 == 정상적인 문단이라고 판단
          # 한 글자만 치고 엔터친 경우도 포함하나, 유효한 데이터가 거의 아닐 것이므로 우선 글자수로 조건식 진행
          if len(clean_span_text) != 1: 
            print("clean_span_text:",clean_span_text) 

            result_pragaph_list.append(clean_span_text) 
        
          else:
            unicode = ord(clean_span_text)
            print("유니코드:",ord(clean_span_text))  

            # 8203인 경우 == ZeroWidthSpace 문자가 아닌 경우는 정상적으로 결과 리스트에 삽입
            if unicode != 8203: 
                print("한글자 데이터입니다.")

                result_pragaph_list.append(clean_span_text) 

            else:
                print("공백없는 공백문자입니다.")

                result_pragaph_list.append("$$space$$") 


          print("~!~!~!~!~!~!~!~!~!")
        
        
        
        for text in result_pragaph_list:
            print("문장:",text)
        
            if text == "$$space$$": # 공백없는 공백문자인 경우
                pragraph_data = pragraph_data + '\n' # 엔터 추가
                # continue
            else:
                pragraph_data = pragraph_data + text
        
        split_pragraph_data = pragraph_data.split('\n')
        pragraph_data_list.append(split_pragraph_data)
        pragraph_data_list = list(filter(None, pragraph_data_list)) # 빈요소가 있으면 제거
        
        print("pragraph_data_list - type:", type(pragraph_data_list))
        
        # 1차 분리한 문단 데이터 리스트에서 빈 요소(=enter) 제거하기 위함
        for texts in pragraph_data_list:
            print("texts:",texts)
            print("texts:",len(texts))

            for text_data in texts:
                # print("text_data:",text_data)
                # print("text_data:",len(text_data))

                if len(text_data) == 0:
                    continue
                else:
                    result_clean_paragraph_list.append(text_data)






            # if len(i) == 1: # 길이가 1인 경우
                # print("길이:::",len(i))

                # if ord(text) != 8203: # nbsp가 아닌 경우 == 일반 텍스트 데이터
                #     clean_pragaph_data = clean_pragaph_data + text
                #     clean_pragaph_data_list.append(text)        
        
    
        print("문단화 결과:",pragraph_data_list)
        print("문단화 결과 - 클린:",result_clean_paragraph_list)

        # result_pragaph_list = [word.strip('\n') for word in result_pragaph_list ] # 특정 문자열 제거
        # result_pragaph_list = [word.strip('\u200b') for word in result_pragaph_list ] # 특정 문자열 '\u200b' 제거
        # result_pragaph_list = list(filter(None, result_pragaph_list)) # 빈요소가 있으면 제거
        # st.write("result_pragaph_list:",result_pragaph_list)

        # for data in pragraph_data_list:
        #     print("data:",data)

        #     if len(data) == 0:
        #         print("data:",data)
        #         print("길이가 0인 데이터입니다.")

        # st.write("pragraph_data_list:",pragraph_data_list)
        # st.write("result_clean_paragraph_list:",result_clean_paragraph_list) # 빈 요소 제거한 최종 list 화면에 출력
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~!!!!!!!!!!!!!!!!~~~~~~~~~~~~~~~~~~~~~~~~")


        # return text
        # return result_pragaph_list
        return result_clean_paragraph_list

    elif soup.find("div", attrs={"id": "postViewArea"}):  # 스마트 에디터 2.0도 크롤링 해오기 위함
        text = soup.find("div", attrs={"id": "postViewArea"}).get_text()
        text = text.replace("\n\n", "")
        return text

    else:
        st.write("라이브러리 실행")
        print("라이브러리 실행")
        result = newspaper_crawler(url)

        return result  # 크롤링 안되는 글 처리

# 네이버 블로그 크롤링해서 출력
def naver_blog_crawler(url):
    st.write("네이버블로그 링크입니다.")

    blog_text = naver_blog_text_scraping(naver_blog_delete_iframe(url))
    print("네이버 블로그 본문 : ", blog_text)

    # blog_text_split = blog_text.replace('\n', '$$split$$') # 분리용 특수문자 삽입 
    # blog_text_split_list = blog_text_split.split('$$split$$') # paragraph 분리
    # blog_text_split_list = list(filter(None, blog_text_split_list)) # 빈요소가 있으면 제거

    # st.write("문단 분리된 데이터:",blog_text_split_list)
    st.write("문단 분리된 데이터:",blog_text)

    print("-------------------완료------------------")

    return blog_text

# newspaper 라이브러리를 사용한 크롤러
def newspaper_crawler(url):
    # 입력받은 url을 통해 크롤링 해오기 위한 코드 - newspaper Article 사용
    url_str = str(url)
    url_strip = url_str.strip()
    page = Article(url_strip, config=config) # 언어설정을 하지 않으면 자동감지된다.
    page.download()
    page.parse() # 웹페이지 parse 진행

    # 크롤링 된 결과의 본문(text)을 변수에 따로 저장
    url_crawling_text = page.text
    url_crawling_title = page.title

    # 크롤링 원본 결과 출력
    st.write("게시글 제목 : ", url_crawling_title)
    st.write("크롤링 결과 : ")
    st.write(url_crawling_text)

    return url_crawling_text

# 크롤링된 결과의 언어를 자동 감지하고 번역해주는 함수 - 이전버전
def google_translator_old(original_crawling_data):
    # detect 라이브러리를 통해 텍스트의 언어 자동감지
    language_sensing = detect(original_crawling_data)
    st.write("언어감지 : ", language_sensing)

    # 크롤링한 언어가 한국어, 일본어이면 구글 번역기 실행(한글,일본어만 -> 영어)
    if language_sensing == 'ko' or language_sensing == 'ja':
        url_crawling_slice = original_crawling_data[0:5000] # 구글번역기 최대 제한수로 인한 슬라이스

        # 영어로 번역
        crawling_data_translate = translator.translate(url_crawling_slice, dest='en')

        # 프롬프트에 날리기 위해 정리된 크롤링 데이터 변수에 저장
        for_prompt_data = crawling_data_translate.text

        # 번역된 결과 출력
        st.write("----- url1 크롤링 결과 번역(en) -----")
        st.write(for_prompt_data)

    # 크롤링한 언어가 영어면 번역 없이 그대로 진행 - 긴 경우 슬라이스만
    elif language_sensing == 'en':
        for_prompt_data = original_crawling_data[0:10000] # chatgpt 최대 제한수로 인한 슬라이스
    
    # 크롤링 되지 않은 경우 == 데이터가 없는 경우 -> 에러처리
    else:
        st.error("크롤링된 값이 너무 적거나 모호해서 언어를 감지할 수 없습니다. 다른 url을 입력해주세요.", icon="🚨")


    return for_prompt_data

# 크롤링된 결과의 언어를 자동 감지하고 번역해주는 함수
# 언어 자동으로 감지, 영어가 아니라면 번역을 진행하는 형태로 수정
def google_translator(original_text):
    # detect 라이브러리를 통해 텍스트의 언어 자동감지

    if len(original_text) != 0:

        language_sensing = detect(original_text)
        st.write("번역 전!!:",original_text)
        
        # 크롤링한 언어가 영어가 아니면 구글 번역기 실행
        if language_sensing != 'en':
            # 영어로 번역
            crawling_data_translate = translator.translate(original_text, dest='en')
            # 프롬프트에 날리기 위해 정리된 크롤링 데이터(.text) 변수에 저장
            for_prompt_data = crawling_data_translate.text
        
        # 크롤링 되지 않은 경우 == 데이터가 없는 경우 -> 에러처리
        else:
            for_prompt_data = ''
            st.error("크롤링된 값이 너무 적거나, 언어가 섞여있거나, 값이 모호해서 언어를 감지할 수 없습니다. 다른 url을 입력해주세요.", icon="🚨")

        st.write("번역!!: ", for_prompt_data)

    else:
        st.error("detect할 데이터에 오류가 있습니다.")

    return for_prompt_data

# 요청 실패시 재시도 하기 위한 함수
def retry_prompt(answer, prompt):
    # 응답된 결과의 길이가 0이면 (없으면) 3회까지 다시 시도
    if len(answer) == 0:
        st.write("[Error] chatGPT의 답변이 정상적으로 생성되지 않았습니다.")
        st.write("5초 후 자동으로 재요청됩니다. (1/3).")
        time.sleep(5)

        first_re_answer = chatGPT(prompt).strip()
        st.write("chatGPT 1차 재요청 결과의 길이 : ", len(first_re_answer))
        st.write("chatGPT 1차 재요청 결과: ", first_re_answer)

        if len(first_re_answer) == 0:
            st.write("[Error] chatGPT의 답변이 정상적으로 생성되지 않았습니다.")
            st.write("10초 후 자동으로 재요청됩니다. (2/3).")
            time.sleep(10)

            second_re_answer = chatGPT(prompt).strip()
            st.write("chatGPT 2차 재요청 결과의 길이 : ", len(second_re_answer))
            st.write("chatGPT 2차 재요청 결과: ", second_re_answer)

            if len(second_re_answer) == 0:
                st.write("[Error] chatGPT의 답변이 정상적으로 생성되지 않았습니다.")
                st.write("15초 후 자동으로 재요청됩니다. (3/3).")
                time.sleep(15)

                third_re_answer = chatGPT(prompt).strip()
                st.write("chatGPT 3차 재요청 결과의 길이 : ", len(third_re_answer))
                st.write("chatGPT 3차 재요청 결과: ", third_re_answer)

                if len(third_re_answer) == 0:
                    st.write("[Error] 잠시 후 재요청 해주시기 바랍니다.")



# url을 입력하면 크롤링을 실행하는 핵심 함수
def url_crawler(input_prompt, *urls):

    for url in urls:
        # url domain 값으로 어떤 사이트인지 판단
        # 1. 유튜브 url인 경우
        if 'youtube' in url:
            # 확인용 안내문구
            st.write("크롤링 작업 진행중...")
            st.write("유튜브 링크입니다.")
            youtube_url = url
            search_value = "v="
            index_no = youtube_url.find(search_value)
            video_id = youtube_url[index_no+2:]

            print("video_id:",video_id)
            st.write("video_id:",video_id)

            # 유튜브 스크립트 크롤링 후 원본 결과를 변수에 저장
            original_crawling_result = youtube_script_crawling(video_id)
            # 반환값 : 크롤링된 자막 데이터 str

            st.write("-"*50)

        # 2. 브런치인 경우
        elif 'brunch' in url:
            # 확인용 안내문구
            st.write("크롤링 작업 진행중...")
            # 브런치 글 크롤링 후 원본 결과를 변수에 저장
            original_crawling_result = brunch_crawler(url)
            # 반환값 : 문단별 데이터가 담긴 list

        # 3. 네이버 블로그인 경우
        elif 'blog.naver.com' in url:
            # 확인용 안내문구
            st.write("크롤링 작업 진행중...")
            # 네이버블로그 글을 크롤링 후 원본 결과를 변수에 저장
            original_crawling_result = naver_blog_crawler(url)
            # 반환 값 : paragraph별 데이터가 담긴 list

        # 4. 네이버 포스트인 경우
        elif 'post.naver.com' in url:
            # 확인용 안내문구
            st.write("크롤링 작업 진행중...")
            # 네이버 포스트 글을 크롤링 후 원본 결과를 변수에 저장
            original_crawling_result = naver_post_crawler(url)
            # 반환값 : 문단별 데이터가 담긴 list

            # 문장이 담긴 list가 반환됨

        # 5. 이외의 경우
        else:
            
            # 확인용 안내문구
            st.write("기타 링크입니다.")
            st.write("크롤링 작업 진행중...")

            original_crawling_result = newspaper_crawler(url)
            # 반환값 : 크롤링된 본문 데이터 str

            # newspaper로 못 가져오는 경우 대비 코드 필요
        
        # 크롤링한 데이터를 번역(en 이외의 언어인 경우) -> 문장 분리 함수에 번역 포함되어 있음
        # for_prompt_data_one = google_translator(original_crawling_result)

        # 자료형이 list인 경우 - 문단으로 분리한 경우 문단별로 요약 요청
        if isinstance(original_crawling_result, list):
            st.write("자료형이 list입니다.")

            final_result.append(pragraph_to_chatGPT(input_prompt, original_crawling_result))


        else:
            # 크롤링된 데이터를 문장 덩어리로 분리 -> 문단으로 분리 불가능하면 차선책으로 문장으로 분리하게 함
            # 현재는 마침표 기준으로 문장 분리만 가능
            st.write("문단 인식이 어려운 데이터입니다.. 마침표를 기준으로 문장으로 분리합니다.")
            st.write(input_prompt)
            st.write(original_crawling_result)

            # 마침표 기준의 문장으로 분리
            final_result.append(separate_by_crawled_data_sentence(input_prompt, original_crawling_result))

    

    st.write("-"*50)


# 문장을 사용자가 입력한 글자수 기준만큼 나눠서 리스트에 저장하기
# list의 요소를 분리해주는 함수
# lst : 분리하고자 하는 대상 리스트
# n : 몇 개씩 분할할지
def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]


# 크롤링된 전체 데이터를 마침표를 기준으로 문장별로 분리해주는 함수  + chatGPT 에 프롬프트 날리기
# in : 크롤링 된 번역 전의 original 데이터 전부
# out : chatGPT 요청 결과
def separate_by_crawled_data_sentence(input_prompt, crawled_data):
    st.write("크롤링된 데이터를 마침표를 기준으로 문장으로 분리합니다..")

    # 몇번 반복되는지 체크하기 위함
    roop_count = 0
    # 청크별로 합쳐서 요약한 결과를 담을 리스트
    combine_chatGPT_result = [] # 추후 딕셔너리로 바꿔도 좋을듯 

    # 나눌 문장 수, 현재는 하드코딩으로 구현
    sentence_division = 25
    st.write("분리되는 문장의 갯수 :", sentence_division)

    # 문장별로 나눠서 리스트에 넣기
    split_data_remove_enter = crawled_data.replace('\n','') # 엔터 제거
    split_data_token = split_data_remove_enter.split() # 스페이스(띄어쓰기) 기준으로 분리해서 token 갯수 알아내기
    split_data = split_data_remove_enter.split('.') # 온점을 기준으로 문장 분리

    # 사용자가 입력한 값(sentence_division)만큼 문장을 덩어리화 해줌
    list_chunked = list_chunk(split_data, sentence_division)

    # 나뉘어진 청크의 총 길이만큼 반복할 반복문 
    for i in range(len(list_chunked)):
        roop_count = roop_count + 1
        st.write("roop_count: ",roop_count)

        # 합쳐진 문장을 담을 변수 선언 - 리스트 또는 스트링 변수에 담기
        combine_list = []
        combine_string = ''
        
        # 기존의 합치는 과정은 필요 없음
        for j in range(len(list_chunked[i])):
            # combine_list.append(list_chunked[i][j]) # 리스트에 넣어 합치기 (방법 1)
            combine_string = combine_string + list_chunked[i][j] + '.' # 스트링으로 합치기 (방법 2)

        print("combine_list: ", combine_list)
        # st.write("combine_string: ", combine_string)
        # print("번역: ", google_translator(combine_string))

        # 번역 완료된 문장 청크 저장
        combine_string_translate_data = google_translator(combine_string)

        # 프롬프트에 요청하는 부분 
        prompt_command = input_prompt + combine_string_translate_data
        st.write(len(prompt_command))

        # chatGPT에 요청하기
        st.write("chatGPT 실행중...")
        result = chatGPT(prompt_command).strip()
        st.write("chatGPT 요청결과: ", result)
        st.write(len(result))

        # 만약 결과의 길이가 0이라면 == 오류가 나서 결과가 응답되지 않았다면 재실행
        retry_prompt(result, prompt_command)

        # 결과가 0이 아니라면 == 오류가 아니라면 최종으로 한국어로 번역된 결과를 리턴해준다.
        if len(result) != 0:
            result_to_kr = translator.translate(result, dest='ko').text
            st.write("chatGPT 요청결과(번역): ", result_to_kr)
            combine_chatGPT_result.append(result_to_kr)
        st.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # 각 청크의 요약된 결과를 담은 리스트를 출력해줌 -> 최종 한번만 출력되도록 주석처리
    # st.write("____________________________")
    # st.write("전체 결과: ", combine_chatGPT_result)

    # return combine_string_translate_data
    return combine_chatGPT_result

# 문단으로 나뉘어진 데이터를 chatGPT prompt에 날리기 위한 함수
def pragraph_to_chatGPT(input_prompt, paragraph_list):
    # 몇번 반복되는지 체크하기 위함
    roop_count = 0
    # 청크별로 합쳐서 요약한 결과를 담을 리스트
    combine_chatGPT_result = [] # 추후 딕셔너리로 바꿔도 좋을듯 
    paragraph_translate_data = []

    print(paragraph_list)
    
    # 나뉘어진 청크의 총 길이만큼 반복할 반복문 
    for i in range(len(paragraph_list)):
        roop_count = roop_count + 1
        st.write("roop_count: ",roop_count)

        # 합쳐진 문장을 담을 변수 선언 - 리스트 또는 스트링 변수에 담기
        # combine_list = []
        # combine_string = ''
        
        # list가 이미 합쳐져 있으므로 문장끼리 합칠 필요는 없음
        # for j in range(len(paragraph_list[i])):
        #     # combine_list.append(list_chunked[i][j]) # 리스트에 넣어 합치기 (방법 1)
        #     combine_string = combine_string + list_chunked[i][j] + '.' # 스트링으로 합치기 (방법 2)

        # print("combine_list: ", combine_list)
        # st.write("combine_string: ", combine_string)
        # print("번역: ", google_translator(combine_string))

        # 번역 완료된 문장 청크 저장
        paragraph_translate_data = google_translator(paragraph_list[i])

        # 프롬프트에 요청할 데이터 ( 사용자가 입력한 값 + 문단 )
        prompt_command = input_prompt + paragraph_translate_data
        st.write(len(prompt_command))

        # chatGPT에 요청하기
        st.write("chatGPT 실행중...")
        result = chatGPT(prompt_command).strip()
        st.write("chatGPT 요청결과: ", result)
        st.write(len(result))

        # 만약 결과의 길이가 0이라면 == 오류가 나서 결과가 응답되지 않았다면 재실행
        retry_prompt(result, prompt_command)

        # 결과가 0이 아니라면 == 오류가 아니라면 최종으로 한국어로 번역된 결과를 리턴해준다.
        if len(result) != 0:
            result_to_kr = translator.translate(result, dest='ko').text
            st.write("chatGPT 요청결과(번역): ", result_to_kr)
            combine_chatGPT_result.append(result_to_kr)
        st.write("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # 각 청크의 요약된 결과를 담은 리스트를 출력해줌 -> 최종 한번만 출력되도록 주석처리
    # st.write("____________________________")
    # st.write("전체 결과: ", combine_chatGPT_result) # 최종적으로 한 번만 출력될 수 있게끔 수정 필요

    # return paragraph_translate_data
    return combine_chatGPT_result

# 크롤링 프로그램 실행용 함수
def start_button(first_url, second_url, third_url, fourth_url, fifth_url, url_count, input_prompt):

    # 입력값이 아무것도 없으면 버튼 비활성화
    disabled_flag = True

    # 입력값이 생기면 버튼 활성화 - 조건 3개 맞춰서 수정 필요
    if len(first_url) != 0:
        disabled_flag = False

    # 버튼 클릭시 실행
    if st.button("시작", disabled=disabled_flag):
        
        # url 1개만 입력하는 경우
        if url_count == 1:
            st.write("1")
            url_crawler(input_prompt, first_url)

        # url을 2개 입력하는 경우
        elif url_count == 2:
            st.write("2")
            url_crawler(input_prompt, first_url, second_url)

        # url 3개 입력하는 경우
        elif url_count == 3:
            st.write("3")
            url_crawler(input_prompt, first_url, second_url, third_url)

        elif url_count == 4:
            st.write("4")
            url_crawler(input_prompt, first_url, second_url, third_url, fourth_url)

        elif url_count == 5:
            st.write("5")
            url_crawler(input_prompt, first_url, second_url, third_url, fourth_url, fifth_url)
        
        st.write("최종:",final_result)
        st.write("완료되었습니다.")
    
    else:
        st.write(" ")


# UI 구현용 메인 함수
def main():

    # button 함수에 사용하기 위한 변수 초기 선언
    input_prompt = ""
    first_url = ""
    second_url = ""
    third_url = ""
    fourth_url = ""
    fifth_url = ""
    # original_crawling_result = ''

    # 제목
    st.title('URL로 크롤링해서 chatGPT에 명령하기')
    st.write("최종 업데이트 : 2023.02.28 ")

    # 안내 문구
    st.info("""URL 크롤링 프로그램 동작 과정\n
        1. 크롤링하고자 하는 URL 입력
        2. 크롤링 진행
        3. 크롤링 된 데이터가 영어가 아닌 경우 googletrans로 번역
        4. 번역된 결과를 chatGPT prompt에 담아서 전송
        (전송형태 : [사용자가 입력한 prompt] + 크롤링 된 en 데이터 )
        5. 요청 결과 출력 (kr)
        """, icon="ℹ️")

    # 내용
    st.write("1. 크롤링 할 URL의 개수를 선택하고 값을 입력하세요.")

    url_count = st.radio(
        "크롤링할 URL 개수 선택",
        (1, 2, 3, 4, 5))

    if url_count == 1:
        first_url = st.text_input('URL 입력칸 1')
        st.write("입력확인 : ", first_url)

    elif url_count == 2:
        first_url = st.text_input('URL 입력칸 1')
        st.write("입력확인 : ", first_url)

        second_url = st.text_input('url 입력칸 2')
        st.write("입력확인 : ", second_url)

    elif url_count == 3:
        first_url = st.text_input('URL 입력칸 1')
        st.write("입력확인 : ", first_url)

        second_url = st.text_input('url 입력칸 2')
        st.write("입력확인 : ", second_url)

        third_url = st.text_input('url 입력칸 3')
        st.write("입력확인 : ", third_url)

    elif url_count == 4:
        first_url = st.text_input('URL 입력칸 1')
        st.write("입력확인 : ", first_url)

        second_url = st.text_input('url 입력칸 2')
        st.write("입력확인 : ", second_url)

        third_url = st.text_input('url 입력칸 3')
        st.write("입력확인 : ", third_url)

        fourth_url = st.text_input('url 입력칸 4')
        st.write("입력확인 : ", fourth_url)

    else:
        first_url = st.text_input('URL 입력칸 1')
        st.write("입력확인 : ", first_url)

        second_url = st.text_input('url 입력칸 2')
        st.write("입력확인 : ", second_url)

        third_url = st.text_input('url 입력칸 3')
        st.write("입력확인 : ", third_url)

        fourth_url = st.text_input('url 입력칸 4')
        st.write("입력확인 : ", fourth_url)

        fifth_url = st.text_input('url 입력칸 5')
        st.write("입력확인 : ", fifth_url)


    st.write("")

    st.write("2. ChatGPT에 요청할 prompt를 입력하세요. (입력한 prompt 뒤에 크롤링된 결과가 붙습니다.)")
    input_prompt = st.text_input(label='ex) Summarize the following ...', value="Summarize the following ")
    st.write("입력확인 : ", input_prompt)

    start_button(first_url, second_url, third_url, fourth_url, fifth_url, url_count, input_prompt)

# 파일을 실행시키면 main 함수를 실행하도록 함
if __name__ == '__main__':
    main()