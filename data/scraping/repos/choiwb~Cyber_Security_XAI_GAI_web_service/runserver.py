
import os
import pandas as pd, numpy as np
import pyspark
import shap
from setting import *
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from flask import request, Flask, render_template, Markup
import plotly.express as px
import re
import time
import itertools
import geoip2.database
import urllib.parse
import base64
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI



java_location = '/usr/lib/jvm/java-11-openjdk-amd64'
os.environ['JAVA_HOME'] = java_location
    
    
app = Flask(__name__)


# URL Encoding (percent encoding) & Base64 Encoding to decoding
# hex to ascii => 미완성 !!!!!!!!!!!!!!!
# 아래 함수의 경우, 비식별 함수 진행 전 삽입 !!!!!
def payload_decode(raw_data_str):

    segments = re.split('%\w\w', raw_data_str)
    # hex to ascii
    # | ~ | 안에 있는 문자열을 기준으로 split
    segments_2 = re.split(r'\|([0-9a-fA-F]+)\|', raw_data_str)
    # print(segments_2)

    for i in range(len(segments)):
        segment = segments[i]

        try:
            # Base64 디코딩 시도
            base64_decoded_bytes = base64.b64decode(segment)
            base64_decoded_str = base64_decoded_bytes.decode('utf-8')

            if len(base64_decoded_str) > 0:
                raw_data_str = raw_data_str.replace(segment, base64_decoded_str)
            else:
                raw_data_str = raw_data_str.replace(segment, segment + base64_decoded_str)

        except:
            pass


    for i in range(len(segments_2)):
        segments_2 = segments_2[i]
        try:
            ascii_decoded_str = bytes.fromhex(segments_2).decode('ascii')
            raw_data_str = raw_data_str.replace(segments_2, ascii_decoded_str)

        except:
            pass
    
    raw_data_str = urllib.parse.unquote(raw_data_str)
    # 2중 디코딩 필요 (..%252f => ..%2f => ../)
    # 즉, % => %25 로 되어 있는 경우 !!!!!
    raw_data_str = urllib.parse.unquote(raw_data_str)

    return raw_data_str


def payload_anonymize(raw_data_str):
    # IP
    # 단, 검색 엔진 (google, safari 등) 관련 버전은 예외 처리
    # 예, chrome/xx.xx.xx.xx, safari/xx.xx.xx.xx
    '''
    파이어폭스, 엣지, 등 기타 검색엔진 적용을 위해 정규 표현식 간결화
    예 임의의 영단어 \w\/ 인경우 
    '''    
    ip_pattern = r'((?<!\w\/)(?<!\w\/[0-9])(?<!\w\/[0-9][0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3})'
    # ip 비식별 대상
    ip_anony = re.findall(ip_pattern, raw_data_str, flags = re.I)
    ip_anony_str = ', '.join(ip_anony)
    if len(ip_anony) > 0:
        ip_anony_explain = 'IP 관련 비식별 대상이 %s 존재하여 10.10.123.123 으로 비식별 처리 하였습니다.' %(ip_anony_str)
    else:
        ip_anony_explain = 'IP 관련 비식별 대상이 존재하지 않습니다.'
    output_str = re.sub(ip_pattern, '10.10.123.123', raw_data_str, flags = re.I)

    # HOST
    # host: 또는 :// 또는 %3a%2f%2f 또는 www.  ~ .go.kr 또는 .or.kr 또는 .com 또는 .co.kr
    host_pattern = r'(?i)(?:(?<=:\/\/)|(?<=%3a%2f%2f)|(?<=www\.)|(?<=host: ))((?!10\.10\.123\.123).*?)(?=\.go\.kr|\.or\.kr|\.com|\.co\.kr)'
    # host 비식별 대상
    host_anony = re.findall(host_pattern, output_str, flags = re.I)
    host_anony_str = ', '.join(host_anony)
    if len(host_anony) > 0:
        host_anony_explain = 'HOST 관련 비식별 대상이 %s 존재하여 ***** 으로 비식별 처리 하였습니다.' %(host_anony_str)
    else:
        host_anony_explain = 'HOST 관련 비식별 대상이 존재하지 않습니다.'
    output_str = re.sub(host_pattern, '*****', output_str, flags = re.I)

    # E-mail
    # xxxxx@xxx. 형태로 있으면, 아이디 부분만 abc 으로 변경
    email_pattern = r'([a-zA-Z0-9_.-]+)@([a-zA-Z0-9_-]+\.)'

    # e-mail 비식별 대상
    email_anony = re.findall(email_pattern, output_str, flags = re.I)
    # email_anony의 짝수번 째 값 제거
    email_anony = [item[0] for item in email_anony]

    email_anony_str = ', '.join(email_anony)
    if len(email_anony) > 0:
        email_anony_explain = 'E-mail 관련 비식별 대상이 %s 존재하여 abc 으로 비식별 처리 하였습니다.' %(email_anony_str)
    else:
        email_anony_explain = 'E-mail 관련 비식별 대상이 존재하지 않습니다.'
        
    output_str = re.sub(email_pattern, r'abc@\2', output_str, flags=re.I)

    # 카드번호
    # 숫자 4자리-숫자 4자리-숫자 4자리-숫자 4자리 => 숫자 4자리 유지 1234-XXXX-XXXX-XXXX 으로 변경
    card_pattern = r'(\d{4}-\d{4}-\d{4}-\d{4})'

    # card 비식별 대상
    card_anony = re.findall(card_pattern, output_str, flags = re.I)
    card_anony_str = ', '.join(card_anony)
    if len(card_anony) > 0:
        card_anony_explain = '카드번호 관련 비식별 대상이 %s 존재하여 1234-XXXX-XXXX-XXXX 으로 비식별 처리 하였습니다.' %(card_anony_str)
    else:
        card_anony_explain = '카드번호 관련 비식별 대상이 존재하지 않습니다.'
    
    output_str = re.sub(card_pattern, '1234-XXXX-XXXX-XXXX', output_str, flags = re.I)

    # 휴대폰(전화)번호
    # 숫자 2,3자리-숫자 3,4자리-숫자 4자리 => 숫자 2,3자리 유지 010-XXXX-XXXX 으로 변경
    number_pattern = r'(\d{2,3}-\d{3,4}-\d{4})'
    # number 비식별 대상
    number_anony = re.findall(number_pattern, output_str, flags = re.I)
    number_anony_str = ', '.join(number_anony)
    if len(number_anony) > 0:
        number_anony_explain = '휴대폰(전화)번호 관련 비식별 대상이 %s 존재하여 02-XXXX-XXXX 으로 비식별 처리 하였습니다.' %(number_anony_str)
    else:
        number_anony_explain = '휴대폰(전화)번호 관련 비식별 대상이 존재하지 않습니다.'

    output_str = re.sub(number_pattern, '010-XXXX-XXXX', output_str, flags = re.I)

    # 주민등록번호
    # 숫자 6자리-숫자 7자리 => 숫자 6자리 유지 123456-XXXXXXX 으로 변경
    jumin_pattern = r'(\d{6}-\d{7})'
    # jumin 비식별 대상
    jumin_anony = re.findall(jumin_pattern, output_str, flags = re.I)
    jumin_anony_str = ', '.join(jumin_anony)
    if len(jumin_anony) > 0:
        jumin_anony_explain = '주민등록번호 관련 비식별 대상이 %s 존재하여 123456-******* 으로 비식별 처리 하였습니다.' %(jumin_anony_str)
    else:
        jumin_anony_explain = '주민등록번호 관련 비식별 대상이 존재하지 않습니다.'

    output_str = re.sub(jumin_pattern, '123456-*******', output_str, flags = re.I)
    
    # 이름
    # 한글 2~4자리 => 홍길동 으로 변경
    name_pattern = r'[가-힣]{2,4}'

    # name 비식별 대상
    name_anony = re.findall(name_pattern, output_str, flags = re.I)
    name_anony_str = ', '.join(name_anony)
    if len(name_anony) > 0:
        name_anony_explain = '이름 관련 비식별 대상이 %s 존재하여 홍길동 으로 비식별 처리 하였습니다.' %(name_anony_str)
    else:
        name_anony_explain = '이름 관련 비식별 대상이 존재하지 않습니다.'
    
    output_str = re.sub(name_pattern, '홍길동', output_str, flags = re.I)

    print(ip_anony_explain)
    print(host_anony_explain)
    print(email_anony_explain)
    print(card_anony_explain)
    print(number_anony_explain)
    print(jumin_anony_explain)
    print(name_anony_explain)

    return output_str


def web_payload_anonymize(raw_data_str):
    # E-mail
    # xxxxx@xxx. 형태로 있으면, 아이디 부분만 abc 으로 변경
    email_pattern = r'([a-zA-Z0-9_.-]+)@([a-zA-Z0-9_-]+\.)'

    # e-mail 비식별 대상
    email_anony = re.findall(email_pattern, raw_data_str, flags = re.I)
    # email_anony의 짝수번 째 값 제거
    email_anony = [item[0] for item in email_anony]

    email_anony_str = ', '.join(email_anony)
    if len(email_anony) > 0:
        email_anony_explain = 'E-mail 관련 비식별 대상이 %s 존재하여 abc 으로 비식별 처리 하였습니다.' %(email_anony_str)
    else:
        email_anony_explain = 'E-mail 관련 비식별 대상이 존재하지 않습니다.'
        
    output_str = re.sub(email_pattern, r'abc@\2', raw_data_str, flags=re.I)

    # 카드번호
    # 숫자 4자리-숫자 4자리-숫자 4자리-숫자 4자리 => 숫자 4자리 유지 1234-XXXX-XXXX-XXXX 으로 변경
    card_pattern = r'(\d{4}-\d{4}-\d{4}-\d{4})'

    # card 비식별 대상
    card_anony = re.findall(card_pattern, output_str, flags = re.I)
    card_anony_str = ', '.join(card_anony)
    if len(card_anony) > 0:
        card_anony_explain = '카드번호 관련 비식별 대상이 %s 존재하여 1234-XXXX-XXXX-XXXX 으로 비식별 처리 하였습니다.' %(card_anony_str)
    else:
        card_anony_explain = '카드번호 관련 비식별 대상이 존재하지 않습니다.'
    
    output_str = re.sub(card_pattern, '1234-XXXX-XXXX-XXXX', output_str, flags = re.I)

    # 휴대폰(전화)번호
    # 숫자 2,3자리-숫자 3,4자리-숫자 4자리 => 숫자 2,3자리 유지 010-XXXX-XXXX 으로 변경
    number_pattern = r'(\d{2,3}-\d{3,4}-\d{4})'
    # number 비식별 대상
    number_anony = re.findall(number_pattern, output_str, flags = re.I)
    number_anony_str = ', '.join(number_anony)
    if len(number_anony) > 0:
        number_anony_explain = '휴대폰(전화)번호 관련 비식별 대상이 %s 존재하여 02-XXXX-XXXX 으로 비식별 처리 하였습니다.' %(number_anony_str)
    else:
        number_anony_explain = '휴대폰(전화)번호 관련 비식별 대상이 존재하지 않습니다.'

    output_str = re.sub(number_pattern, '010-XXXX-XXXX', output_str, flags = re.I)

    # 주민등록번호
    # 숫자 6자리-숫자 7자리 => 숫자 6자리 유지 123456-XXXXXXX 으로 변경
    jumin_pattern = r'(\d{6}-\d{7})'
    # jumin 비식별 대상
    jumin_anony = re.findall(jumin_pattern, output_str, flags = re.I)
    jumin_anony_str = ', '.join(jumin_anony)
    if len(jumin_anony) > 0:
        jumin_anony_explain = '주민등록번호 관련 비식별 대상이 %s 존재하여 123456-******* 으로 비식별 처리 하였습니다.' %(jumin_anony_str)
    else:
        jumin_anony_explain = '주민등록번호 관련 비식별 대상이 존재하지 않습니다.'

    output_str = re.sub(jumin_pattern, '123456-*******', output_str, flags = re.I)
    
    # 이름
    # 한글 2~4자리 => 홍길동 으로 변경
    name_pattern = r'[가-힣]{2,4}'

    # name 비식별 대상
    name_anony = re.findall(name_pattern, output_str, flags = re.I)
    name_anony_str = ', '.join(name_anony)
    if len(name_anony) > 0:
        name_anony_explain = '이름 관련 비식별 대상이 %s 존재하여 홍길동 으로 비식별 처리 하였습니다.' %(name_anony_str)
    else:
        name_anony_explain = '이름 관련 비식별 대상이 존재하지 않습니다.'
    
    output_str = re.sub(name_pattern, '홍길동', output_str, flags = re.I)

    print(email_anony_explain)
    print(card_anony_explain)
    print(number_anony_explain)
    print(jumin_anony_explain)
    print(name_anony_explain)

    return output_str


def payload_anonymize_highlight(raw_data_str):
    
    # 비식별 하이라이트 처리 - background black & foreground white
    replacement = "\033[40m\033[37m" + "\\1" + "\033[0m"

    ip_anony = '10.10.123.123'
    host_anony = '*****'
    email_anony = 'abc@'
    card_anony = '1234-XXXX-XXXX-XXXX'
    number_anony = '010-XXXX-XXXX'
    jumin_anony = '123456-*******'
    name_anony = '홍길동'
    
    anony_list = [ip_anony, host_anony, email_anony, card_anony, number_anony, jumin_anony, name_anony]
    payload_anonymize_highlight = re.sub("(" + "|".join(map(re.escape, anony_list)) + ")", replacement, raw_data_str, flags=re.I)
    print(payload_anonymize_highlight)
    
    background_black_foreground_white_regex = r'\x1b\[40m\x1b\[37m(.*?)\x1b\[0m'

    payload_anonymize_highlight_html = re.sub(background_black_foreground_white_regex, r'<span style = "background-color:black; color:white">\1</span>', payload_anonymize_highlight)
    return payload_anonymize_highlight_html

def web_payload_anonymize_highlight(raw_data_str):
    
    # 비식별 하이라이트 처리 - background black & foreground white
    replacement = "\033[40m\033[37m" + "\\1" + "\033[0m"

    email_anony = 'abc@'
    card_anony = '1234-XXXX-XXXX-XXXX'
    number_anony = '010-XXXX-XXXX'
    jumin_anony = '123456-*******'
    name_anony = '홍길동'
    
    anony_list = [email_anony, card_anony, number_anony, jumin_anony, name_anony]
    payload_anonymize_highlight = re.sub("(" + "|".join(map(re.escape, anony_list)) + ")", replacement, raw_data_str, flags=re.I)
    print(payload_anonymize_highlight)
    
    background_black_foreground_white_regex = r'\x1b\[40m\x1b\[37m(.*?)\x1b\[0m'

    payload_anonymize_highlight_html = re.sub(background_black_foreground_white_regex, r'<span style = "background-color:black; color:white">\1</span>', payload_anonymize_highlight)
    return payload_anonymize_highlight_html

def IPS_predict_UI_sql_result():
    raw_data_str = request.form['raw_data_str']
    
    # encode to decode
    raw_data_str = payload_decode(raw_data_str)
    
    # 비식별
    raw_data_str = payload_anonymize(raw_data_str)

    conf = pyspark.SparkConf().setAppName('prep_data').setMaster('local')
    sc = pyspark.SparkContext.getOrCreate(conf = conf)

    # 세션 수행
    session = SparkSession(sc)
    payload = raw_data_str
    domain_one_row_df = pd.DataFrame(data = [payload], columns = ['payload'])
    schema = StructType([StructField("payload", StringType(), True)])
    # 데이터 프레임 등록
    domain_df = session.createDataFrame(domain_one_row_df, schema=schema)
    # 현재 스키마 정보 확인
    domain_df.printSchema()
    # 데이터 프레임 'table'이라는 이름으로 SQL테이블 생성
    domain_df.createOrReplaceTempView("table")

    # output_df = session.sql(query)
    output_df = session.sql(ips_query)

    sql_result_df = output_df.toPandas()
    # sql_result_df['ips_00014_payload_logscaled_length_value'] = sql_result_df['ips_00014_payload_logscaled_length_value'].astype(int)

    print('전처리 데이터 크기: ', sql_result_df.shape)
    print('전처리 데이터 feature 명: ', sql_result_df.columns)
    print('전처리 데이터 feature 타입 명: ', sql_result_df.dtypes)
    sql_result_df_array = np.array(sql_result_df)
    print('전처리 데이터 feature 값: ', sql_result_df_array)

    return sql_result_df



def WAF_predict_UI_sql_result():
    raw_data_str = request.form['raw_data_str']
    
    # encode to decode
    raw_data_str = payload_decode(raw_data_str)
    
    # 비식별
    raw_data_str = payload_anonymize(raw_data_str)

    conf = pyspark.SparkConf().setAppName('prep_data').setMaster('local')
    sc = pyspark.SparkContext.getOrCreate(conf = conf)

    # 세션 수행
    session = SparkSession(sc)
    payload = raw_data_str
    domain_one_row_df = pd.DataFrame(data = [payload], columns = ['payload'])
    schema = StructType([StructField("payload", StringType(), True)])
    # 데이터 프레임 등록
    domain_df = session.createDataFrame(domain_one_row_df, schema=schema)
    # 현재 스키마 정보 확인
    domain_df.printSchema()
    # 데이터 프레임 'table'이라는 이름으로 SQL테이블 생성
    domain_df.createOrReplaceTempView("table")

    output_df = session.sql(waf_query)

    sql_result_df = output_df.toPandas()

    print('전처리 데이터 크기: ', sql_result_df.shape)
    print('전처리 데이터 feature 명: ', sql_result_df.columns)
    print('전처리 데이터 feature 타입 명: ', sql_result_df.dtypes)
    sql_result_df_array = np.array(sql_result_df)
    print('전처리 데이터 feature 값: ', sql_result_df_array)

    # TF-IDF 기반 전처리 피처 생성 및 SQL 피처와 통합
    valud_tfidf_feature = vectorizer_waf.fit_transform(domain_one_row_df['payload']).toarray()
    valid_tfidf_df = pd.DataFrame(valud_tfidf_feature, columns=vectorizer_waf.get_feature_names_out())
    # TF * IDF 도출
    valid_tfidf_df = valid_tfidf_df * tfidf_value_waf
    valid_tfidf_df.columns = tfidf_feature_waf

    ''''''
    valid_tfidf_extract = valid_tfidf_df.loc[:, (valid_tfidf_df != 0).any(axis=0)]
    print(valid_tfidf_extract)
    ''''''
    
    # sql_result_df와 valid_tfidf_df 통합
    sql_tfidf_df = pd.concat([sql_result_df, valid_tfidf_df], axis=1)

    return sql_tfidf_df


def WEB_predict_UI_sql_result():
    raw_data_str = request.form['raw_data_str']
    
    # encode to decode
    raw_data_str = payload_decode(raw_data_str)
    
    # 비식별
    raw_data_str = web_payload_anonymize(raw_data_str)

    conf = pyspark.SparkConf().setAppName('prep_data').setMaster('local')
    sc = pyspark.SparkContext.getOrCreate(conf = conf)

    # 세션 수행
    session = SparkSession(sc)
    web_log = raw_data_str
    domain_one_row_df = pd.DataFrame(data = [web_log], columns = ['web_log'])
    schema = StructType([StructField("web_log", StringType(), True)])
    # 데이터 프레임 등록
    domain_df = session.createDataFrame(domain_one_row_df, schema=schema)
    # 현재 스키마 정보 확인
    domain_df.printSchema()
    # 데이터 프레임 'table'이라는 이름으로 SQL테이블 생성
    domain_df.createOrReplaceTempView("table")

    output_df = session.sql(web_query)

    sql_result_df = output_df.toPandas()

    print('전처리 데이터 크기: ', sql_result_df.shape)
    print('전처리 데이터 feature 명: ', sql_result_df.columns)
    print('전처리 데이터 feature 타입 명: ', sql_result_df.dtypes)
    sql_result_df_array = np.array(sql_result_df)
    print('전처리 데이터 feature 값: ', sql_result_df_array)

    return sql_result_df

'''
def IPS_web_UI_preprocess():   
    payload_df = IPS_predict_UI_sql_result()
    # payload_arr = np.array(payload_df)

    return payload_df

def WAF_web_UI_preprocess():   
    payload_df = WAF_predict_UI_sql_result()
    # payload_arr = np.array(payload_df)

    return payload_df

def WEB_web_UI_preprocess():   
    payload_df = WEB_predict_UI_sql_result()
    # payload_arr = np.array(payload_df)

    return payload_df
'''

@app.route('/')
def input():
    return render_template('user_input.html')




import openai
import multiprocessing

# (개인) 유료 API 키!!!!!!!!
os.environ['OPENAI_API_KEY'] = "YOUR OPEN AI API KEY !!!!!!!"

tactics_path = 'chat_gpt_context/tactics.txt'
waf_parsing_path = 'chat_gpt_context/waf_parsing_desc.txt'
web_parsing_path = 'chat_gpt_context/web_parsing_desc.txt'
sigmarule_yaml_sample_path = 'chat_gpt_context/sample_sigma_rule_yaml.txt'
snortrule_sample_path = 'chat_gpt_context/sample_snort_rule.txt'


def load_context(file_path):
    with open(file_path, "r") as f:
       context = f.read()

    return context

base_template = """You are a cyber security analyst. about user question, answering specifically in korean.
            Use the following pieces of payload to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            For questions, related to Mitre Att&ck, Enterprise Tactics ID consist of 1TA0001 (Initial Access), TA0002 (Execution), TA0003 (Persistence), 
            TA0004 (Privilege Escalation), TA0005 (Defense Evasion), TA0006 (Credential Access), TA0007 (Discovery), 
            TA0008 (Lateral Movement), TA0009 (Collection), TA0010 (Exfiltration), TA0011 (Command and Control),
            TA0040 (Impact), TA0042 (Resource Development), TA0043 (Reconnaissance).
            For questions, related to CVE (Common Vulnerabilities and Exposures), Please only search for CVEs released in 2015.
            For questions, related to Cyber Kill Chain Model, Please write only the names in order of all steps.
            For question, related to Attack type associated with payload, there are SQL Injection, Command Injection, XSS (Cross Site Scripting), Attempt access admin page, RCE (Remote Code Execution), WordPress vulnerability, malicious bot.
            Respond don't know to questions not related to cyber security.
            Use three sentences maximum and keep the answer as concise as possible. 
            payload: {payload}
            question: {question}
            answer: """
            
            
continue_template = """You are a cyber security analyst. about user question, answering specifically in korean.
            Use the following pieces of payload to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            For questions, related to Mitre Att&ck, Enterprise Tactics ID consist of 1TA0001 (Initial Access), TA0002 (Execution), TA0003 (Persistence), 
            TA0004 (Privilege Escalation), TA0005 (Defense Evasion), TA0006 (Credential Access), TA0007 (Discovery), 
            TA0008 (Lateral Movement), TA0009 (Collection), TA0010 (Exfiltration), TA0011 (Command and Control),
            TA0040 (Impact), TA0042 (Resource Development), TA0043 (Reconnaissance).
            For questions, related to CVE (Common Vulnerabilities and Exposures), Please only search for CVEs released in 2015.
            For questions, related to Cyber Kill Chain Model, Please write only the names in order of all steps.
            For question, related to Attack type associated with payload, there are SQL Injection, Command Injection, XSS (Cross Site Scripting), Attempt access admin page, RCE (Remote Code Execution), WordPress vulnerability, malicious bot.
            Respond don't know to questions not related to cyber security.
            Use three sentences maximum and keep the answer as concise as possible. 
            context: {context}
            payload: {payload}
            question: {question}
            answer: """
            
snort_sigma_template = """You are a cyber security analyst. about user question, answering specifically in english.
            Use the following pieces of payload to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            For questions, related to Snort Rule or Sigma Rule, Just write rule related to Attack type associated with payload, there are SQL Injection, Command Injection, XSS (Cross Site Scripting), Attempt access admin page, RCE (Remote Code Execution), WordPress vulnerability, malicious bot, following sample Snort rule or Sigma Rule.
            Respond don't know to questions not related to cyber security.
            Keep the answer as concise as possible. 
            context: {context}
            sample snort rule or sigma rule: {sample_rule}
            payload: {payload}
            question: {question}
            answer: """
            
xai_template = """You are a cyber security analyst. about user question, answering specifically in korean.
            Use the following pieces of XAI feature importance to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer. 
            For questions, related to feature importance explanation of xai, Just write based on AI detection keywords, it is associated with major attack types such as SQL Injection, Command Injection, XSS (Cross Site Scripting), Attempt access admin page, RCE (Remote Code Execution), WordPress vulnerability, malicious bot following xai feature importance.
            Respond don't know to questions not related to cyber security.
            Use three sentences maximum and keep the answer as concise as possible. 
            xai feature importance: {xai_result}
            question: {question}
            answer: """


BASE_CHAIN_PROMPT = PromptTemplate(input_variables=["payload", "question"],template=base_template)
CONTINUE_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "payload", "question"],template=continue_template)
SNORT_SIGMA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "sample_rule", "payload", "question"],template=snort_sigma_template)
XAI_CHAIN_PROMPT = PromptTemplate(input_variables=["xai_result", "question"],template=xai_template)



callbacks = [StreamingStdOutCallbackHandler()]
# gpt35_llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0.7, max_tokens=512, 
#                   callbacks=callbacks, streaming=True)
# gpt40_llm = ChatOpenAI(model_name='gpt-4', temperature=0.7, max_tokens=512, 
#                   callbacks=callbacks, streaming=True)

# gpt-3.5-turbo & gpt-4 => gpt-3.5-turbo-instruct 변경 !!!!!!!!!!!!!!!
gpt35_instruct_llm = OpenAI(model_name='gpt-3.5-turbo-instruct', temperature=0, max_tokens=512,
                            callbacks=callbacks, streaming=True)

# base_llmchain = LLMChain(llm=gpt35_llm, prompt=BASE_CHAIN_PROMPT)
# continue_llmchain = LLMChain(llm=gpt35_llm, prompt=CONTINUE_CHAIN_PROMPT)
# snort_sigma_llmchain = LLMChain(llm=gpt40_llm, prompt=SNORT_SIGMA_CHAIN_PROMPT)
# xai_llmchain = LLMChain(llm=gpt40_llm, prompt=XAI_CHAIN_PROMPT)
base_llmchain = LLMChain(llm=gpt35_instruct_llm, prompt=BASE_CHAIN_PROMPT)
continue_llmchain = LLMChain(llm=gpt35_instruct_llm, prompt=CONTINUE_CHAIN_PROMPT)
snort_sigma_llmchain = LLMChain(llm=gpt35_instruct_llm, prompt=SNORT_SIGMA_CHAIN_PROMPT)
xai_llmchain = LLMChain(llm=gpt35_instruct_llm, prompt=XAI_CHAIN_PROMPT)

sigmarule_file = load_context(sigmarule_yaml_sample_path)
snortrule_file = load_context(snortrule_sample_path)

default_ques1 = '입력된 payload완 연관된 공격 명 1개와 판단 근거를 작성해주세요.'
default_ques2 = '피처 중요도의 AI 탐지 키워드 기반으로 설명을 작성해주세요.'
ques1 = '입력된 payload와 연관된 CVE 1개와 판단 근거를 작성해주세요.'
ques2 = '입력된 payload와 연관된 Tactics ID 1개와 판단 근거를 작성해주세요.'
ques3_0 = '입력된 payload의 Cyber Kill Chain Model을 작성해주세요.'
ques3 = '입력된 payload와 연관된 Cyber Kill Chain 대응 단계 1개 및 대응 방안을 작성해주세요.'
ques4 = '입력된 payload의 Snort Rule을 작성해주세요.'
ques5 = '입력된 payload의 Sigma Rule을 작성해주세요.'


'''
@app.route('/IPS_web_UI_predict', methods=['POST'])
def IPS_web_UI_predict():

    payload_df = IPS_web_UI_preprocess()
    payload_arr = np.array(payload_df)

    pred = IPS_model.predict(payload_arr)
    pred_proba = IPS_model.predict_proba(payload_arr)

    normal_proba = int(np.round(pred_proba[:, 0], 2) * 100)
    anomalies_proba = int(np.round(pred_proba[:, 1], 2) * 100)

    return render_template('IPS_server_output.html', data = [pred, normal_proba, anomalies_proba],
                                            # method_str = method_str
                                            )


@app.route('/WAF_web_UI_predict', methods=['POST'])
def WAF_web_UI_predict():

    payload_df = WAF_web_UI_preprocess()
    payload_arr = np.array(payload_df)

    pred = WAF_model.predict(payload_arr)
    pred_proba = WAF_model.predict_proba(payload_arr)

    normal_proba = int(np.round(pred_proba[:, 0], 2) * 100)
    anomalies_proba = int(np.round(pred_proba[:, 1], 2) * 100)
                                                                                                                                                                                                                                                            
    return render_template('WAF_server_output.html', data = [pred, normal_proba, anomalies_proba],
                                                # method_str = method_str
                                                )


@app.route('/WEB_web_UI_predict', methods=['POST'])
def WEB_web_UI_predict():

    payload_df = WEB_web_UI_preprocess()
    payload_arr = np.array(payload_df)

    pred = WEB_model.predict(payload_arr)
    pred_proba = WEB_model.predict_proba(payload_arr)

    cmd_proba = int(np.round(pred_proba[:, 0], 2) * 100)
    sql_proba = int(np.round(pred_proba[:, 2], 2) * 100)
    xss_proba = int(np.round(pred_proba[:, 3], 2) * 100)
    # normal_proba = int(np.round(pred_proba[:, 1], 2) * 100)
    normal_proba = 100 - cmd_proba - sql_proba - xss_proba

    total_proba_list = [cmd_proba, normal_proba, sql_proba, xss_proba]
    
    # pred = 0 이면, CMD Injection
    # 1이면, 정상
    # 2이면, SQL Injection
    # 3이면, XSS
    
    if pred == 0:
        pred_label = 'CMD Injection'
    elif pred == 1:
        pred_label = '정상'
    elif pred == 2:
        pred_label = 'SQL Injection'
    else:
        pred_label = 'XSS'
    print(pred_label)
    
    pred_max_proba = max(total_proba_list)
    print(pred_max_proba)


    return render_template('WEB_server_output.html', data = [pred, cmd_proba, normal_proba, sql_proba, xss_proba],
                                                # method_str = method_str
                                                )
'''

###############################################
# log odds 형태 라벨 값을 확률 값으로 변환
def shap_logit(x):
    logit_result = 1 / (1 + np.exp(-x))
    normalized_result = logit_result / np.sum(logit_result)
    return normalized_result
###############################################



# 보안 시그니처 패턴 리스트 => highlight 처리
# 시그니처 패턴 리스트 csv 호출 => 사용자 정의 & Web CGI 공격 & 패턴 블럭  sheet 참조! (단, Snort(사용자 패턴) 시트 제외!)
sig_pattern_csv_path = 'save_model'
df = pd.read_csv(os.path.join(sig_pattern_csv_path, 'signature_pattern_list_20230804.csv'))

signature_list = df['탐지 패턴'].tolist()
# 전체 시그니처 패턴 리스트 소문자화
signature_list = [x.lower() for x in signature_list]
# 시그니처 패턴 리스트에서 XSS 창 실행 예외 처리를 위해 소문자 이스케이프 처리
signature_list = [re.sub(r'[\<]' , '%26lt%3B', x) for x in signature_list] 
signature_list = [re.sub(r'[\>]' , '%26gt%3B', x) for x in signature_list]

df['제조사'] = df.apply(lambda x: 'W사' if x['제조사'] == 'SPECIFIC VENDOR !!!!!!!' 
                                    else 'S사' if x['제조사'] == 'SPECIFIC VENDOR !!!!!!!'
                                    else 'I사' if x['제조사'] == 'SPECIFIC VENDOR !!!!!!!'
                                    else '', axis = 1)
df['장비 명'] = df.apply(lambda x: 'S제품' if x['장비 명'] == 'SPECIFIC PRODUCT !!!!!!!' 
                                    else 'M제품' if x['장비 명'] == 'SPECIFIC PRODUCT !!!!!!!'
                                    else 'N제품' if x['장비 명'] == 'SPECIFIC PRODUCT !!!!!!!'
                                    else '', axis = 1)

vendor_list = df['제조사'].tolist()
equip_list = df['장비 명'].tolist()
method_list = df['탐지 명'].tolist()
descrip_list = df['설명'].tolist()
response_list = df['대응 방안'].tolist()



# IPS & WAF & WEB 피처 설명 테이블 생성
ips_feature_file = 'save_model/IPS_KISA_feature_def_202311.xlsx'
ips_feature_df = pd.read_excel(ips_feature_file)
ips_feature_df = ips_feature_df[['피처 명', '피처 설명']]

waf_feature_file = 'save_model/WAF_KISA_feature_def_202311.xlsx'
waf_feature_df = pd.read_excel(waf_feature_file)
waf_feature_df = waf_feature_df[['피처 명', '피처 설명']]

web_feature_file = 'save_model/WEB_KISA_feature_def_202311.xlsx'
web_feature_df = pd.read_excel(web_feature_file)
web_feature_df = web_feature_df[['피처 명', '피처 설명']]


def highlight_text(text, signature, ai_field):

    # background yellow - 시그니처 패턴
    replacement = "\033[103m" + "\\1" + "\033[49m"
    # foreground red - AI 생성 필드
    replacement_2 = "\033[91m" + "\\1" + "\033[39m"       
    
    # 시그니처 패턴 또는 AI 생성 필드 인 경우, highlight 처리
    # re.escape() : 특수문자를 이스케이프 처리
    text = re.sub("(" + "|".join(map(re.escape, signature)) + ")", replacement, text, flags=re.I)      
    
    # IPS
    if ai_field == ips_ai_field:
        # ai_field에서 'ips' element 제외
        ai_field = [i for i in ai_field if i != 'ips']
        
        # ai_field에서 useragent 제외
        not_ips_useragent_field = [i for i in ai_field if i not in ips_useragent_field]

        text = re.sub("(" + "|".join(not_ips_useragent_field) + ")", replacement_2, text, flags=re.I)

        # text.spliyt('HTTP/1.[01]')[1]에 user-agent가 있는 경우, highlight 처리
        if 'HTTP/1.1' in text and text.count('HTTP/1.1') == 1:
            text = text.split('HTTP/1.1')[0] + 'HTTP/1.1' + re.sub("(" + "|".join(ips_useragent_field) + ")", replacement_2, text.split('HTTP/1.1')[1], flags=re.I)
        elif 'HTTP/1.0' in text and text.count('HTTP/1.0') == 1:
            text = text.split('HTTP/1.0')[0] + 'HTTP/1.0' + re.sub("(" + "|".join(ips_useragent_field) + ")", replacement_2, text.split('HTTP/1.0')[1], flags=re.I)
    
    # WAF
    elif ai_field == waf_ai_field:
        # ai_field에서 'waf' element 제외
        ai_field = [i for i in ai_field if i != 'waf']
        
        # ai_field에서 useragent 제외
        not_waf_useragent_field = [i for i in ai_field if i not in waf_useragent_field]

        text = re.sub("(" + "|".join(not_waf_useragent_field) + ")", replacement_2, text, flags=re.I)

        # text.spliyt('HTTP/1.[01]')[1]에 user-agent가 있는 경우, highlight 처리
        if 'HTTP/1.1' in text and text.count('HTTP/1.1') == 1:
            text = text.split('HTTP/1.1')[0] + 'HTTP/1.1' + re.sub("(" + "|".join(waf_useragent_field) + ")", replacement_2, text.split('HTTP/1.1')[1], flags=re.I)
        elif 'HTTP/1.0' in text and text.count('HTTP/1.0') == 1:
            text = text.split('HTTP/1.0')[0] + 'HTTP/1.0' + re.sub("(" + "|".join(waf_useragent_field) + ")", replacement_2, text.split('HTTP/1.0')[1], flags=re.I)

    # WEB
    else:
        # ai_field에서 'web' element 제외
        ai_field = [i for i in ai_field if i != 'web']
        
        text = re.sub("(" + "|".join(ai_field) + ")", replacement_2, text, flags=re.I)
        
    # 39m 91m이 붙어 있는 경우 제거 !!!!!!!! 왜냐하면 단일처리를 위해, 빨간색 폰트 끝과 시작이 붙어 있으면 연속 키워드로 인식하기 위함.
    # text = re.sub(r'\x1b\[39m\x1b\[91m', '', text)    

    regex = re.compile('\x1b\[103m(.*?)\x1b\[49m')

    matches = [regex.match(text[i:]) for i in range(len(text))] 
    sig_pattern_prep = [m.group(0) for m in matches if m] 

    sig_pattern = [re.sub(r'\x1b\[103m|\x1b\[49m', '', i) for i in sig_pattern_prep]
    sig_pattern = [re.sub(r'\x1b\[91m|\x1b\[39m', '', i) for i in sig_pattern]

    sig_pattern_df = pd.DataFrame(columns = ['탐지 순서', '제조사', '장비 명', '탐지 명', '설명', '대응 방안'])
    count = 0
    for i in sig_pattern:
        # 시그니처 탐지 패턴의 경우, 소문자화
        i = i.lower()
        count = count + 1

        if i in signature_list:
            j = signature_list.index(i)
            # print('%d 번째 시그니처 패턴 공격명: %s' %(count, method_list[j]))
            one_row_df = pd.DataFrame([[count, vendor_list[j], equip_list[j], method_list[j], descrip_list[j], response_list[j]]], 
                                columns = ['탐지 순서', '제조사', '장비 명', '탐지 명', '설명', '대응 방안'])
            sig_pattern_df = pd.concat([sig_pattern_df, one_row_df], axis = 0)
        
    return text, sig_pattern_df

'''
def web_highlight_text(text, signature, web_ai_field):

    # background yellow - 시그니처 패턴
    replacement = "\033[103m" + "\\1" + "\033[49m"
    # foreground red - AI 생성 필드
    replacement_2 = "\033[91m" + "\\1" + "\033[39m"

    # 시그니처 패턴 또는 AI 생성 필드 인 경우, highlight 처리
    # re.escape() : 특수문자를 이스케이프 처리
    text = re.sub("(" + "|".join(map(re.escape, signature)) + ")", replacement, text, flags=re.I)

    # ai_field에서 cmd, sql 제외
    not_cmd_sql_field = [i for i in web_ai_field if i not in web_cmd_1 and i not in web_cmd_2 and i not in web_cmd_3 and i not in web_sql_1 and i not in web_sql_2 and i not in web_sql_3 and i not in web_sql_4 and i not in web_sql_5]
    # ai_field에서 cmd, sql 인 경우
    cmd_sql = [i for i in web_ai_field if i in web_cmd_1 or i in web_cmd_2 or i in web_cmd_3 or i in web_sql_1 or i in web_sql_2 or i in web_sql_3 or i in web_sql_4 or i in web_sql_5]

    text = re.sub("(" + "|".join(not_cmd_sql_field) + ")", replacement_2, text, flags=re.I)

    # test.split('HTTP/1.[01]')[0]에 cmd, sql가 있는 경우, highlight 처리
    if 'HTTP/1.1' in text and text.count('HTTP/1.1') == 1:
        text = re.sub("(" + "|".join(cmd_sql) + ")", replacement_2, text.split('HTTP/1.1')[0], flags=re.I) + 'HTTP/1.1' + text.split('HTTP/1.1')[1]
    elif 'HTTP/1.0' in text and text.count('HTTP/1.0') == 1:
        text = re.sub("(" + "|".join(cmd_sql) + ")", replacement_2, text.split('HTTP/1.0')[0], flags=re.I) + 'HTTP/1.0' + text.split('HTTP/1.0')[1]

    # 39m 91m이 붙어 있는 경우 제거 !!!!!!!! 왜냐하면 단일처리를 위해, 빨간색 폰트 끝과 시작이 붙어 있으면 연속 키워드로 인식하기 위함.
    # text = re.sub(r'\x1b\[39m\x1b\[91m', '', text)
        
    regex = re.compile('\x1b\[103m(.*?)\x1b\[49m')

    matches = [regex.match(text[i:]) for i in range(len(text))] 
    sig_pattern_prep = [m.group(0) for m in matches if m] 

    sig_pattern = [re.sub(r'\x1b\[103m|\x1b\[49m', '', i) for i in sig_pattern_prep]
    sig_pattern = [re.sub(r'\x1b\[91m|\x1b\[39m', '', i) for i in sig_pattern]

    sig_pattern_df = pd.DataFrame(columns = ['탐지 순서', '제조사', '장비 명', '탐지 명', '설명', '대응 방안'])
    count = 0
    for i in sig_pattern:
        # 시그니처 탐지 패턴의 경우, 소문자화
        i = i.lower()
        count = count + 1

        if i in signature_list:
            j = signature_list.index(i)
            # print('%d 번째 시그니처 패턴 공격명: %s' %(count, method_list[j]))
            one_row_df = pd.DataFrame([[count, vendor_list[j], equip_list[j], method_list[j], descrip_list[j], response_list[j]]], 
                                columns = ['탐지 순서', '제조사', '장비 명', '탐지 명', '설명', '대응 방안'])
            sig_pattern_df = pd.concat([sig_pattern_df, one_row_df], axis = 0)

    return text, sig_pattern_df
'''

def dl_highlight_text(text, signature, dl_ai_field):

    # ai_field의 경우 상위10개 키워드에 컬럼에 대한 선언
    # 특수문자 => [\\특수문자] 로 변경
    dl_ai_field = [re.sub(r'([^\w\s])', r'[\\\1]', x) for x in dl_ai_field]

    # background yellow - 시그니처 패턴
    replacement = "\033[103m" + "\\1" + "\033[49m"
    # foreground red - AI 생성 필드
    replacement_2 = "\033[91m" + "\\1" + "\033[39m"

    # 시그니처 패턴 또는 AI 생성 필드 인 경우, highlight 처리
    # re.escape() : 특수문자를 이스케이프 처리
    text = re.sub("(" + "|".join(dl_ai_field) + ")", replacement_2, text, flags=re.I)
    # 39m 91m이 붙어 있는 경우 제거 !!!!!!!! 왜냐하면 단일처리를 위해, 빨간색 폰트 끝과 시작이 붙어 있으면 연속 키워드로 인식하기 위함.
    text = re.sub(r'\x1b\[39m\x1b\[91m', '', text)

    text = re.sub("(" + "|".join(map(re.escape, signature)) + ")", replacement, text, flags=re.I)
    
    regex = re.compile('\x1b\[103m(.*?)\x1b\[49m')

    matches = [regex.match(text[i:]) for i in range(len(text))] 
    sig_pattern_prep = [m.group(0) for m in matches if m] 

    sig_pattern = [re.sub(r'\x1b\[103m|\x1b\[49m', '', i) for i in sig_pattern_prep]
    sig_pattern = [re.sub(r'\x1b\[91m|\x1b\[39m', '', i) for i in sig_pattern]

    sig_pattern_df = pd.DataFrame(columns = ['탐지 순서', '제조사', '장비 명', '탐지 명', '설명', '대응 방안'])
    count = 0
    for i in sig_pattern:
        # 시그니처 탐지 패턴의 경우, 소문자화
        i = i.lower()
        count = count + 1

        if i in signature_list:
            j = signature_list.index(i)
            # print('%d 번째 시그니처 패턴 공격명: %s' %(count, method_list[j]))
            one_row_df = pd.DataFrame([[count, vendor_list[j], equip_list[j], method_list[j], descrip_list[j], response_list[j]]], 
                                columns = ['탐지 순서', '제조사', '장비 명', '탐지 명', '설명', '대응 방안'])
            sig_pattern_df = pd.concat([sig_pattern_df, one_row_df], axis = 0)
    
    return text, sig_pattern_df
    
    
@app.route('/WAF_payload_parsing', methods = ['POST'])
def WAF_payload_parsing():
    raw_data_str = request.form['raw_data_str']

    ##############################################
    # raw_data_str이 " 으로 시작하는 경우 '' 처리
    if raw_data_str[0] == '"':
        raw_data_str = raw_data_str[1:]
    ##############################################

    # encode to decode
    raw_data_str = payload_decode(raw_data_str)

    # 비식별
    raw_data_str = payload_anonymize(raw_data_str)

    pre_df = pd.DataFrame([raw_data_str], columns = ['payload'])
    pre_df['http_method'] = [str(x).split(' ')[0] for x in pre_df['payload']]


    mtd = [str(x).split(' ')[0] for x in pre_df['payload']]
    for i, m in enumerate(mtd):
        if len(m) > 10 or len(m) == 1 or not m.isalpha():
            mtd[i] = ''

    method_list = ['', 'upload', 'get', 'profind', 'put', 'options', 'head', 'trace', 'connect', 'delete', 'post', 'patch']
 
    m_idx = []
    not_m_idx = []

    for i, m in enumerate(pre_df['http_method']):
        # if m in method_list:
        if m.lower() in method_list:
            m_idx.append(i)
        else:
            not_m_idx.append(i)

    df_m = pre_df.iloc[m_idx].reset_index(drop=True)
    df_nm = pre_df.iloc[not_m_idx].reset_index(drop=True)

    # payload_0: payload에서 ' ' (공백) 첫번째를 기준으로 나누엇, 2번째 값을 반환하므로, http_url 부터 끝 임.
    # 따라서, http_url + http_query + http_body
    df_m['payload_0'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['payload']]
    # 'HTTP/1.' 의 앞에 있는 공백 ' ' 을 기준으로 split
    # 따라서, http_url + http_query
    df_m['url_query'] = [str(x).split(' HTTP/1.', maxsplit=1)[0] for x in df_m['payload_0']]

    http_body = []
    for i in df_m['payload_0']:
        if 'HTTP/1.1' in i:
            # payload_0에서 'HTTP/1.1' 이 있는 경우, http_body
            http_body.append(i.split('HTTP/1.1', maxsplit=1)[1])
            http_body = ['HTTP/1.1' + x for x in http_body]

        elif 'HTTP/1.0' in i:
            # payload_0에서 'HTTP/1.0' 이 있는 경우, http_body
            http_body.append(i.split('HTTP/1.0', maxsplit=1)[1])
            http_body = ['HTTP/1.0' + x for x in http_body]

        else:
            http_body.append('')

    df_m['http_body'] = http_body
    # url_query에서 ? 가 있는 경우, 1번째 값을 반환하므로, http_url 임.
    df_m['http_url'] = [str(x).split('?', maxsplit=1)[0] for x in df_m['url_query']]

    query = []
    for i in df_m['url_query']:
        if '?' in i:
            # url_query에서, ?가 있는 경우, 2번째 값을 반환하므로, http_query 임.
            query.append('?'+i.split('?', maxsplit=1)[1])
        else:
            query.append('')
    df_m['http_query'] = query

    

    df_res = df_m[['payload', 'http_method', 'http_url', 'http_query', 'http_body']]

    a = []
    a.append('')
    df_nm['http_method'] = a
    df_nm['http_url'] = a
    df_nm['http_query'] = a                                                         
    df_nm['http_body'] = a
    df_nm['uri'] = list(df_nm['payload'])

    if str(df_nm['http_url'][0:1]) == 'nan' and str(df_nm['http_query'][0:1]) == 'nan' and str(df_nm['http_body'][0:1]) == 'nan':
        df_nm['http_body'][0:1] = df_nm['uri'][0:1]

    if df_nm['uri'][0:1].isna().sum() == 0:
        df_nm = df_nm.fillna('-')
        df_nm_np = np.where(df_nm.iloc[:, :] == '', '-', df_nm.iloc[:, :])
        df_nm = pd.DataFrame(df_nm_np, columns = df_nm.columns.tolist())
        df_nm['http_body'] = df_nm['uri']
        df_nm = df_nm.drop(['payload', 'uri'], axis = 1)
        df_nm['http_version'] = '-'
        final_df = df_nm[['http_method', 'http_url', 'http_query', 'http_version', 'http_body']]

        # http_query 필드의 첫 글자가 '?' 인 경우, '' 처리
        if final_df.iloc[0,2].startswith('?') == True:
            final_df['http_query'] = final_df['http_query'].str[1:]

        # FLASK 적용
        flask_html = final_df.to_html(index = False, justify = 'center')
        # print(flask_df)
        # CTI 적용
        cti_json = final_df.to_json(orient = 'records')
        # print(ctf_df)
        warning_statement = '비정상적인 Payload 입력 형태 입니다. (예, payload 의 시작이 특수문자 등)'


    else:
        # http_version => HTTP/1.1 OR HTTP/1.0 OR HTTP/2.0
        df_res['http_version'] = '-'
        # df_res.iloc[0,4]) ' '  로 시작하는 경우 '' 처리
        if df_res.iloc[0,4].startswith(' ') == True:
            df_res['http_body'] = df_res['http_body'].str[1:]

        # print('##############')
        # print(df_res['http_body'][0])

        if df_res.iloc[0,4].lower().startswith('http/') == True:
            df_res['http_version'][0:1] = df_res['http_body'][0:1].str[0:8]
            df_res['http_body'] = df_res['http_body'].str[8:]
            
        final_df = df_res[['payload', 'http_method', 'http_url', 'http_query', 'http_version', 'http_body']]
        final_df = final_df.drop('payload', axis = 1)

        final_np = np.where(final_df.iloc[:, :] == '', '-', final_df.iloc[:, :])
        final_df = pd.DataFrame(final_np, columns = final_df.columns.tolist())

        # http_query 필드의 첫 글자가 '?' 인 경우, '' 처리
        if final_df.iloc[0,2].startswith('?') == True:
            final_df['http_query'] = final_df['http_query'].str[1:]


        # FLASK 적용
        flask_html = final_df.to_html(index = False, justify = 'center')
        # print(flask_df)
        # CTI 적용
        cti_json = final_df.to_json(orient = 'records')
        # print(ctf_df)

        warning_statement = '정상적인 Payload 입력 형태 입니다.'

    return final_df, warning_statement


@app.route('/WEB_payload_parsing', methods = ['POST'])
def WEB_payload_parsing():
    raw_data_str = request.form['raw_data_str']

    # raw_data_str이 "" 인 경우, " " 처리
    raw_data_str = raw_data_str.replace('""', '" "')

    # encode to decode
    # raw_data_str = payload_decode(raw_data_str)

    # 비식별
    raw_data_str = web_payload_anonymize(raw_data_str)

    # raw_data_str에 '"'가 4개 이상 (2쌍) 인 경우, APACHE, 아니면, IIS
    if raw_data_str.count('"') >= 4:
        pre_df = pd.DataFrame([raw_data_str], columns = ['payload'])
        pre_df['payload_prep'] = [str(x).split('"', maxsplit=1)[1] for x in pre_df['payload']]

        pre_df['http_method'] = [str(x).split(' ', maxsplit=1)[0] for x in pre_df['payload_prep']]

        mtd = [str(x).split(' ', maxsplit=1)[0] for x in pre_df['payload_prep']]
        for i, m in enumerate(mtd):
            if len(m) > 10 or len(m) == 1 or not m.isalpha():
                mtd[i] = ''

        method_list = ['', 'upload', 'get', 'profind', 'put', 'options', 'head', 'trace', 'connect', 'delete', 'post', 'patch']

        m_idx = []
        not_m_idx = []

        for i, m in enumerate(pre_df['http_method']):
            # if m in method_list:
            if m.lower() in method_list:
                m_idx.append(i)
            else:
                not_m_idx.append(i)

        df_m = pre_df.iloc[m_idx].reset_index(drop=True)
        df_nm = pre_df.iloc[not_m_idx].reset_index(drop=True)

        # payload_0: payload에서 ' ' (공백) 첫번째를 기준으로 나누엇, 2번째 값을 반환하므로, http_url 부터 끝 임.
        # 따라서, http_url + http_query + 끝
        df_m['payload_0'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['payload_prep']]

        # url_query: payload_0에서, ' ' (공백) 첫번째를 기준으로 나누어, 1번째 값을 반환하므로, http_url ~ http_query 임.
        # 따라서, http_url + http_query
        df_m['url_query'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['payload_0']]

        except_url_query = []
        for i in df_m['payload_0']:
            if ' ' in i:
                # payload_0에서 공백이 있는 경우, http_body
                except_url_query.append(i.split(' ', maxsplit=1)[1])
                # print(except_url_query)
            else:
                except_url_query.append('')

        df_m['except_url_query'] = except_url_query
        # url_query에서 ? 가 있는 경우, 1번째 값을 반환하므로, http_url 임.
        df_m['http_url'] = [str(x).split('?', maxsplit=1)[0] for x in df_m['url_query']]

        query = []
        for i in df_m['url_query']:
            if '?' in i:
                # url_query에서, ?가 있는 경우, 2번째 값을 반환하므로, http_query 임.
                query.append('?'+i.split('?', maxsplit=1)[1])
            else:
                query.append('')
        df_m['http_query'] = query

        # except_url_query 여기서 9번째 글자가 공백이 아닌 경우 공백 추가
        # HTTP/1.1 또는 HTTP/1.0 다음 9번째 글자가 공백이 아닌 경우, 해당 문자열 제거
        for i, x in enumerate(df_m['except_url_query']):
            if x[8] != ' ':
                df_m['except_url_query'][i] = x[:8] + x[9:]

        df_m['http_version'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['except_url_query']]

        df_m['except_version'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['except_url_query']]
        df_m['http_status'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['except_version']]

        df_m['except_status'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['except_version']]
        df_m['pkt_bytes'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['except_status']]

        df_m['except_bytes'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['except_status']]
        df_m['referer'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['except_bytes']]

        df_m['except_referer'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['except_bytes']]
        df_m['agent_etc'] = [str(x).split('"', maxsplit=1)[1] for x in df_m['except_referer']]

        # df_m['agent_etc'] 가 ' "'로 시작하는 경우, '' 처리
        if df_m['agent_etc'][0].startswith(' "'):
            df_m['agent_etc'][0] = df_m['agent_etc'][0][2:]

        df_m['user_agent'] = [str(x).split('"', maxsplit=1)[0] for x in df_m['agent_etc']]

        # xforwarded_for 및 request_body 있는 경우, NGINX 임.
        # if df_m.iloc[0,-1].count('"') >= 1:
        # 2022/11/14 기준 APACHE & NGINX 구분 로직 TO DO
        # 1. NGINX 처럼 APACHE, IIS에 xforwarded_for 및 request_body 필드 추가 (null 값으로)
        # 2. APACHE 이면서, SIEM RAW 필드에 'nginx' 문자열 있는 경우,  NGINX 아니면, APACHE => 이 경우, NGINX에 xforwarded_for, request_body 추가하지 않음.
        # 3. http_version 이후를, http_body 필드를 생성하여 필드 통합.

        # df_m['except_agent'] = [str(x).split('"', maxsplit=1)[1] for x in df_m['except_agent']]
        # df_m['xforwarded_for'] = [str(x).split('"', maxsplit=1)[0] for x in df_m['except_agent']]

        # df_m['except_xforwarded'] = [str(x).split('"', maxsplit=1)[1] for x in df_m['except_agent']]

        # df_m['request_body'] = [str(x).split('"', maxsplit=1)[1] for x in df_m['except_xforwarded']]
        
        final_df = df_m[['http_method', 'http_url', 'http_query', 'http_version', 'user_agent']]
    
        final_np = np.where(final_df.iloc[:,:] == '', '-', final_df.iloc[:,:])
        final_df = pd.DataFrame(final_np, columns = final_df.columns)

        # http_query 필드의 첫 글자가 '?' 인 경우, '' 처리
        if final_df.iloc[0,2].startswith('?') == True:
            final_df['http_query'] = final_df['http_query'].str[1:]
        
        # final_df의 컬럼별 값에서 '"' 가 있는 경우, '' 처리
        final_df = final_df.apply(lambda x: x.str.replace('"', ''))
        # final_df의 '' 값은 '-' 로 변경
        final_df = final_df.replace('', '-')
        
        # FLASK 적용
        flask_html = final_df.to_html(index = False, justify = 'center')
        # print(flask_df)
        # CTI 적용
        cti_json = final_df.to_json(orient = 'records')
        # print(ctf_df)
        # warning_statement = 'WEB_NGINX 로그 입니다.'
        warning_statement = 'WEB 로그 입니다.'
        
        # else:
        # final_df = df_m[['http_method', 'http_url', 'http_query', 'http_version', 'http_status', 'pkt_bytes', 'referer', 'user_agent']]

        # final_np = np.where(final_df.iloc[:,:] == '', '-', final_df.iloc[:,:])
        # final_df = pd.DataFrame(final_np, columns = final_df.columns)

        # http_query 필드의 첫 글자가 '?' 인 경우, '' 처리
        # if final_df.iloc[0,2].startswith('?') == True:
        #    final_df['http_query'] = final_df['http_query'].str[1:]
        
        # final_df의 컬럼별 값에서 '"' 가 있는 경우, '' 처리
        # final_df = final_df.apply(lambda x: x.str.replace('"', ''))
        # final_df의 '' 값은 '-' 로 변경
        # final_df = final_df.replace('', '-')

        # SIEM 기반 이 아닌 일반적인 WEB 로그 포괄을 위해 아래와 같은 주요 필드 5개 파싱.
        # final_df = final_df[['http_method', 'http_url', 'http_query', 'http_version', 'user_agent']]

        # FLASK 적
        # flask_html = final_df.to_html(index = False, justify = 'center')
        # print(flask_df)
        # CTI 적용
        # cti_json = final_df.to_json(orient = 'records')
        # print(ctf_df)
        # warning_statement = 'WEB_APACHE 로그 입니다.'
        # warning_statement = 'WEB 로그 입니다.'

    else:
        try:
            pre_df = pd.DataFrame([raw_data_str], columns = ['payload'])
            pre_df['payload_prep'] = [str(x).split(' ', maxsplit=4)[4] for x in pre_df['payload']]
            # payload_prep 이 'http/' 부터 시작
            pre_df['start_version'] = re.findall(r'http/(.*)', pre_df.iloc[0,1], flags=re.I)
            pre_df['http_method'] = [str(x).split(' ', maxsplit=1)[0] for x in pre_df['payload_prep']]
            pre_df['start_version'] = 'HTTP/' + pre_df.iloc[0,2]
            mtd = [str(x).split(' ', maxsplit=1)[0] for x in pre_df['payload_prep']]
            for i, m in enumerate(mtd):
                if len(m) > 10 or len(m) == 1 or not m.isalpha():
                    mtd[i] = ''

            method_list = ['', 'upload', 'get', 'profind', 'put', 'options', 'head', 'trace', 'connect', 'delete', 'post', 'patch']

            m_idx = []
            not_m_idx = []

            for i, m in enumerate(pre_df['http_method']):
                # if m in method_list:
                if m.lower() in method_list:
                    m_idx.append(i)
                else:
                    not_m_idx.append(i)


            df_m = pre_df.iloc[m_idx].reset_index(drop=True)
            df_nm = pre_df.iloc[not_m_idx].reset_index(drop=True)

            # payload_0: payload에서 ' ' (공백) 첫번째를 기준으로 나누엇, 2번째 값을 반환하므로, http_url 부터 끝 임.
            # 따라서, http_url + http_query + 끝
            df_m['payload_0'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['payload_prep']]
            # url_query: payload_0에서, ' ' (공백) 첫번째를 기준으로 나누어, 1번째 값을 반환하므로, http_url ~ http_query 임.
            # 따라서, http_url + http_query
            df_m['url_query'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['payload_0']]

            except_url_query = []
            for i in df_m['payload_0']:
                if ' ' in i:
                    # payload_0에서 공백이 있는 경우, http_body
                    except_url_query.append(i.split(' ', maxsplit=1)[1])
                    # print(except_url_query)
                else:
                    except_url_query.append('')

            df_m['except_url_query'] = except_url_query
            # url_query에서 ? 가 있는 경우, 1번째 값을 반환하므로, http_url 임.
            df_m['http_url'] = [str(x).split('?', maxsplit=1)[0] for x in df_m['url_query']]

            query = []
            for i in df_m['url_query']:
                if '?' in i:
                    # url_query에서, ?가 있는 경우, 2번째 값을 반환하므로, http_query 임.
                    query.append('?'+i.split('?', maxsplit=1)[1])
                else:
                    query.append('')
            df_m['http_query'] = query

            df_m['http_version'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['start_version']]
            print(df_m['http_version'])

            df_m['except_version'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['start_version']]
            df_m['user_agent'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['except_version']]
            
            # df_m['except_agent'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['except_version']]
            # df_m['referer'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['except_agent']]

            # df_m['except_referer'] =  [str(x).split(' ', maxsplit=1)[1] for x in df_m['except_agent']]
            # df_m['http_status'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['except_referer']]

            # df_m['except_status'] = [str(x).split(' ', maxsplit=1)[1] for x in df_m['except_referer']]
            # df_m['sent_bytes'] = [str(x).split(' ', maxsplit=1)[0] for x in df_m['except_status']]

            final_df = df_m[['http_method', 'http_url', 'http_query', 'http_version', 'user_agent']]

            # http_query 필드의 첫 글자가 '?' 인 경우, '' 처리
            if final_df.iloc[0,2].startswith('?') == True:
                final_df['http_query'] = final_df['http_query'].str[1:]

            # final_df의 컬럼별 값에서 '"' 가 있는 경우, '' 처리
            final_df = final_df.apply(lambda x: x.str.replace('"', ''))
            # final_df의 '' 값은 '-' 로 변경
            final_df = final_df.replace('', '-')

            # FLASK 적용
            flask_html = final_df.to_html(index = False, justify = 'center')
            # print(flask_df)
            # CTI 적용
            cti_json = final_df.to_json(orient = 'records')
            # print(ctf_df)
            # warning_statement = 'WEB_IIS 로그 입니다.'
            warning_statement = 'WEB 로그 입니다.'
        except:
            # flask_html = 'WEB 로그가 아닙니다.'
            # cti_json = 'WEB 로그가 아닙니다.'
            final_df = pd.DataFrame([raw_data_str], columns = ['web_log'])
            warning_statement = 'WEB 로그가 아닙니다.'

    return final_df, warning_statement



@app.route('/IPS_XAI_result', methods = ['POST'])
def IPS_XAI_result(): 
   # payload의 raw data 입력 값!
    raw_data_str = request.form['raw_data_str']

    # payload의 개행문자 (\n|\r|\t) => ' ' 로 변경
    raw_data_str = re.sub(r'(\n|\r|\t)', ' ', raw_data_str)

    ##########################################################
    # raw_data_str 변수에 XSS 관련 문구가 있어서 창이 나오는 이슈 해결 
    raw_data_str = re.sub(r'[\<]' , '%26lt%3B', raw_data_str)
    raw_data_str = re.sub(r'[\>]' , '%26gt%3B', raw_data_str)
    ##########################################################
    
    # encode to decode
    raw_data_str = payload_decode(raw_data_str)

    # 비식별
    raw_data_str = payload_anonymize(raw_data_str)

    # 비식별 하이라이트
    payload_anonymize_highlight_html = payload_anonymize_highlight(raw_data_str)
    
    # payload_df = IPS_web_UI_preprocess()
    payload_df = IPS_predict_UI_sql_result()

    payload_arr = np.array(payload_df)
    
    pred = IPS_model.predict(payload_arr)
    pred_proba = IPS_model.predict_proba(payload_arr)

    if pred == 1:
        db_ai = '공격'
        pred_proba = int(np.round(pred_proba[:, 1], 2) * 100)
    else:
        db_ai = '정상'
        pred_proba = int(np.round(pred_proba[:, 0], 2) * 100)

    ml_pred_comment = '인공지능에서 예측한 결과 %s일 확률이 %s%%입니다.' %(db_ai, pred_proba)
    
    IPS_total_explainer = pickle.load(open(IPS_explainer_path, 'rb'))

    # anomalies : shap_values[1], normal: shap_values[0]
    shap_values_sql = IPS_total_explainer.shap_values(payload_arr)
    shap_values_sql = np.array(shap_values_sql)

    # shap_values_sql[1] 이 0 이상인 경우, 공격, 미만인 경우, 정상으로 판단
    shap_values_sql_direction = np.where(shap_values_sql[1] >= 0, '공격', '정상')
    print(shap_values_sql_direction)
    shap_values_sql_2 = np.abs(shap_values_sql[1]).mean(0)
    shap_values_sql_2_ratio = shap_logit(shap_values_sql_2)
    shap_values_sql_2_ratio = np.round(shap_values_sql_2_ratio, 4)
    print(shap_values_sql_2_ratio)

    shap_values_sql_direction = np.array(shap_values_sql_direction).flatten()
    mean_shap_value_df = pd.DataFrame(list(zip(payload_df.columns, shap_values_sql_2_ratio, shap_values_sql_direction)),
                                   columns=['피처 명','피처 중요도', 'AI 예측 방향'])
    
    train_mean_df = pd.DataFrame([['모델 평균', expected_value_sql_logit, '기준'], ['예측', attack_proba, attack_proba - expected_value_sql_logit]], 
                        columns = ['모델 평균/예측', '위험도(%)', '위험도(%) 증감'])
    train_mean_df['위험도(%) 증감'][1] = np.round(train_mean_df['위험도(%) 증감'][1], 2)

    if train_mean_df['위험도(%) 증감'][1] < 0:
        train_mean_df['위험도(%) 증감'][1] = train_mean_df['위험도(%) 증감'][1]
    else:
        train_mean_df['위험도(%) 증감'] = train_mean_df['위험도(%) 증감'].astype(str)
        train_mean_df['위험도(%) 증감'][1] = '+' +  train_mean_df['위험도(%) 증감'][1]

    ################################################################
    # expected_value_sql_logit 기반 plotly bar chart 생성 !!!! (기준 100%)
    
    train_mean_proba_plot = px.bar(train_mean_df, x = '위험도(%)',  y = '모델 평균/예측',  
                                        orientation = 'h',
                                        text = '위험도(%)',
                                        hover_data = {'모델 평균/예측': True, '위험도(%)': True, '위험도(%) 증감': True},
                                        color = '모델 평균/예측', 
                                        color_discrete_map = {'모델 평균': '#0000FF', '예측': '#FF0000'},
                                        template = 'plotly_white')

    train_mean_proba_plot.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True,   
                        legend_itemclick = False, legend_itemdoubleclick = False,
                        showlegend = False,
                        title_text='모델 평균/예측 위험도', title_x=0.5,
                        yaxis_title = None,
                        # xaxis_title = None,
                        width = 900,
                        height = 250
                        )
    
    train_mean_proba_html = train_mean_proba_plot.to_html(full_html=False, include_plotlyjs=True,
                            config = {'displaylogo': False,
                            'modeBarButtonsToRemove': ['zoom', 'pan', 'zoomin', 'zoomout', 'autoscale', 'select2d', 'lasso2d',
                            'resetScale2d', 'toImage']
                            }
                            )
    

    # mean_shap_value_df 의 피처 중요도를 기준으로 내림차순 정렬
    mean_shap_value_df = mean_shap_value_df.sort_values(by=['피처 중요도'], ascending = False)
    top10_shap_values = mean_shap_value_df.iloc[0:10, :]
    top10_shap_values = top10_shap_values.reset_index(drop = True)

    top10_shap_values['순위'] = top10_shap_values.index + 1

    # 피처 설명 테이블과 join
    top10_shap_values = pd.merge(top10_shap_values, ips_feature_df, how = 'left', on = '피처 명')
    top10_shap_values = top10_shap_values[['순위', '피처 명', '피처 설명', '피처 중요도', 'AI 예측 방향']]

    payload_df_t = payload_df.T
    payload_df_t.columns = ['피처 값']
    # payload_df_t에 피처 명 컬럼 추가
    payload_df_t['피처 명'] = payload_df_t.index
    top10_shap_values = pd.merge(top10_shap_values, payload_df_t, how = 'left', on = '피처 명')
    top10_shap_values = top10_shap_values[['순위', '피처 명', '피처 설명', '피처 값', '피처 중요도', 'AI 예측 방향']]

    # top10_shap_values['피처 명'] 에서 'ips_' 제거
    top10_shap_values['피처 명'] = top10_shap_values['피처 명'].apply(lambda x: x[4:] if x.startswith('ips_') else x)

    top10_shap_values['순위'] = top10_shap_values.index + 1
    top10_shap_values  = top10_shap_values[['순위', '피처 명', '피처 설명', '피처 값', '피처 중요도', 'AI 예측 방향']]
    top10_shap_values['피처 중요도'] = top10_shap_values['피처 중요도'].apply(lambda x: round(x, 4))

    # print(top10_shap_values)

    # 보안 시그니처 패턴 리스트 highlight
    sig_ai_pattern, sig_df = highlight_text(raw_data_str, signature_list, ips_ai_field)
    print(sig_ai_pattern)

    ai_detect_regex = r'\x1b\[91m(.*?)\x1b\[39m'
    ai_detect_list = re.findall(ai_detect_regex, sig_ai_pattern)
    ai_detect_list = [re.sub(r'\x1b\[103m|\x1b\[49m', '', x) for x in ai_detect_list]

    ###################################################################
    # raw_adta_str 변수에 XSS 관련 문구 떼문에 변경한 부분 원복
    ai_detect_list = [re.sub('%26lt%3B', '<', x) for x in ai_detect_list]
    ai_detect_list = [re.sub('%26gt%3B', '>', x) for x in ai_detect_list]
    ###################################################################

    ai_feature_list = []
    ai_pattern_list = []

    '''
    for x in ai_detect_list:
        for y in ips_auth_field: 
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[0])
                ai_pattern_list.append(y)
        for y in ips_bof_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[1])
                ai_pattern_list.append(y)
        for y in ips_cmd_1_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[2])
                ai_pattern_list.append(y)
        for y in ips_cmd_2_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[3])
                ai_pattern_list.append(y)
        for y in ips_code_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[4])
                ai_pattern_list.append(y)
        for y in ips_dir_1_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[5])
                ai_pattern_list.append(y)
        for y in ips_dir_2_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[6])
                ai_pattern_list.append(y)
        for y in ips_dir_count_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[7])
                ai_pattern_list.append(y)
        for y in ips_cgi_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[8])
                ai_pattern_list.append(y)
        for y in ips_wp_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[9])
                ai_pattern_list.append(y)
        for y in ips_error_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[10])
                ai_pattern_list.append(y)
        for y in ips_file_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[11])
                ai_pattern_list.append(y)
        for y in ips_http_method_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[12])
                ai_pattern_list.append(y)
        for y in ips_malware_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[13])
                ai_pattern_list.append(y)
        for y in ips_rce_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[14])
                ai_pattern_list.append(y)
        for y in ips_sql_1_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[15])
                ai_pattern_list.append(y)
        for y in ips_sql_2_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[16])
                ai_pattern_list.append(y)
        for y in ips_useragent_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[17])
                ai_pattern_list.append(y)
        for y in ips_php_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[18])
                ai_pattern_list.append(y)
        for y in ips_xss_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[19])
                ai_pattern_list.append(y)
        for y in ips_and_count_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[20])
                ai_pattern_list.append(y)
        for y in ips_semico_count_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(ips_attack_feature_list[21])
                ai_pattern_list.append(y)
    '''

    for x in ai_detect_list:
        for i, attack_field in enumerate(ips_attack_feature_value_list):
            for y in attack_field:
                if re.findall(y, x.lower()):
                    ai_feature_list.append(ips_feature_list[i])
                    ai_pattern_list.append(y)

    # ai_feature_list = list(itertools.chain(*ai_feature_list))
    # ai_pattern_list = list(itertools.chain(*ai_pattern_list))
    # ai_pattern_list에사 (.*?) => [~] 로 변경, [%%] => % 로 변경
    ai_pattern_list = [x.replace('(.*?)', '[~]').replace('[%%]', '%') for x in ai_pattern_list]
    # [\\특수문자] => 특수문자 로 변경
    ai_pattern_list = [re.sub(r'(\[\\)(.*?)(\])', r'\2', x, flags = re.I) for x in ai_pattern_list]
    ai_pattern_list = [x.replace('[^a-zA-Z0-9가-힣]', '') for x in ai_pattern_list]

    # ai_feature_list, ai_detect_list 를 이용하여 2개 컬럼 기반 data frame 생성
    print(ai_detect_list)
    print(ai_feature_list)
    print(ai_pattern_list)

    ai_feature_df = pd.DataFrame({'피처 명': ai_feature_list, 'AI 탐지 키워드': ai_pattern_list})
    # ai_feature_df['피처 명'] 중복된 행이 있다면, ',' 기준 concat
    ai_feature_df = ai_feature_df.groupby('피처 명')['AI 탐지 키워드'].apply(', '.join).reset_index()


    # print(ai_feature_df)
    top10_shap_values = top10_shap_values.merge(ai_feature_df, how='left', on='피처 명')
    top10_shap_values['AI 탐지 키워드'] = top10_shap_values['AI 탐지 키워드'].fillna('-')

    top10_shap_values['피처 중요도'] = np.round(top10_shap_values['피처 중요도'] * 100, 2)
    top10_shap_values = top10_shap_values.rename(columns = {'피처 중요도': '피처 중요도(%)'})

    # top10_shap_values의 피처 중요도 합계 
    top10_shap_values_sum = top10_shap_values['피처 중요도(%)'].sum()
    # top10_shap_values_sum_etc = 1 - top10_shap_values_sum
    # etc_df = pd.DataFrame([['기타', '상위 10개 이외 피처', '-', top10_shap_values_sum_etc, '기타']], columns = ['피처 명', '피처 설명', '피처 값', '피처 중요도', 'AI 예측 방향'])
    # top10_shap_values = pd.concat([top10_shap_values, etc_df], axis=0)
    # top10_shap_values = top10_shap_values.sort_values(by='피처 중요도', ascending=False)
    # top10_shap_values = top10_shap_values.reset_index(drop = True)


    ##################################################
    # 학습 데이터 기반 피처 중요도 요약 (상위 3개 피처)
    ##################################################

    first_feature = top10_shap_values.iloc[0, 1]
    first_fv = top10_shap_values.iloc[0, 3]
    first_word = top10_shap_values.iloc[0,-1]
    second_feature = top10_shap_values.iloc[1, 1]
    second_fv = top10_shap_values.iloc[1, 3]
    second_word = top10_shap_values.iloc[1,-1]
    third_feature = top10_shap_values.iloc[2, 1]
    third_fv = top10_shap_values.iloc[2, 3]
    third_word = top10_shap_values.iloc[2,-1]


   
    if first_feature != ips_attack_dir_conti_feature and first_feature != ips_attack_and_conti_feature and first_feature != ips_attack_semico_conti_feature:
        if first_fv == 1:
            first_fv_result = '공격 탐지'
            first_statement = '%s 가 %s 하였고 AI 탐지 키워드는 %s 입니다.'  %(first_feature, first_fv_result, first_word)
        else:
            first_fv_result = '정상 인식'
            first_statement = '%s 가 %s 하였습니다.' %(first_feature, first_fv_result)
    # elif first_feature == ips_whitelist_feature:
    #     if first_fv == 1:
    #         first_statement = '로그 전송이 1건 이하 임에 따라 공격일 가능성이 있습니다.'     
    #     else:
    #         first_statement = '로그 전송이 2건 이상 임에 따라 정상 입니다.'
    elif first_feature == ips_attack_and_conti_feature or first_feature == ips_attack_semico_conti_feature:
        if 'ampersand' in first_feature.lower():
            first_statement = '& 특수문자가 총 %d건 입니다.' %(first_fv)
        else:
            first_statement = '; 특수문자가 총 %d건 입니다.' %(first_fv)
    else:
        first_statement = '상위 디렉토리 접근이 총 %s건 입니다.' % first_fv       

    if second_feature != ips_attack_dir_conti_feature and second_feature != ips_attack_and_conti_feature and second_feature != ips_attack_semico_conti_feature:
        if second_fv == 1:
            second_fv_result = '공격 탐지'
            second_statement = '%s 가 %s 하였고 AI 탐지 키워드는 %s 입니다.'  %(second_feature, second_fv_result, second_word)
        else:
            second_fv_result = '정상 인식'
            second_statement = '%s 가 %s 하였습니다.' %(second_feature, second_fv_result)
    # elif second_feature == ips_whitelist_feature:
    #     if second_fv == 1:
    #         second_statement = '로그 전송이 1건 이하 임에 따라 공격일 가능성이 있습니다.'     
    #     else:
    #         second_statement = '로그 전송이 2건 이상 임에 따라 정상 입니다.'
    elif second_feature == ips_attack_and_conti_feature or second_feature == ips_attack_semico_conti_feature:
        if 'ampersand' in second_feature.lower():
            second_statement = '& 특수문자가 총 %d건 입니다.' %(second_fv)
        else:
            second_statement = '; 특수문자가 총 %d건 입니다.' %(second_fv)
    else:
        second_statement = '상위 디렉토리 접근이 총 %s건 입니다.' % second_fv      

    if third_feature != ips_attack_dir_conti_feature and third_feature != ips_attack_and_conti_feature and third_feature != ips_attack_semico_conti_feature:
        if third_fv == 1:
            third_fv_result = '공격 탐지'
            third_statement = '%s 가 %s 하였고 AI 탐지 키워드는 %s 입니다.'  %(third_feature, third_fv_result, third_word)
        else:
            third_fv_result = '정상 인식'
            third_statement = '%s 가 %s 하였습니다.' %(third_feature, third_fv_result)
    # elif third_feature == ips_whitelist_feature:
    #     if third_fv == 1:
    #         third_statement = '로그 전송이 1건 이하 임에 따라 공격일 가능성이 있습니다.'     
    #     else:
    #         third_statement = '로그 전송이 2건 이상 임에 따라 정상 입니다.'
    elif third_feature == ips_attack_and_conti_feature or third_feature == ips_attack_semico_conti_feature:
        if 'ampersand' in third_feature.lower():
            third_statement = '& 특수문자가 총 %d건 입니다.' %(third_fv)
        else:
            third_statement = '; 특수문자가 총 %d건 입니다.' %(third_fv)
    else:
        third_statement = '상위 디렉토리 접근이 총 %s건 입니다.' % third_fv

    # top10_shap_values to html
    top10_shap_values_html = top10_shap_values.to_html(index=False, justify='center')


    # top10_shap_values to plotly                         
    # 피처 중요도에 커서 올리면 피처 설명 나오도록 표시
    # background color = white
    # 피처 중요도 기준 0.5 이상은 '공격' 미만은 '정상'
    # top10_shap_values['AI 예측 방향'] = ['공격' if x >= 0.5 else '정상' for x in top10_shap_values['피처 중요도']]

    summary_plot = px.bar(top10_shap_values, x="피처 중요도(%)", y="피처 명", 
                color = 'AI 예측 방향', color_discrete_map = {'공격': '#FF0000', '정상': '#00FF00', '기타': '#0000FF'},
                text = '피처 중요도(%)', orientation='h', hover_data = {'피처 명': False, '피처 설명': True, '피처 값': True, '피처 중요도(%)': False, 'AI 예측 방향': False,
                                                                    'AI 탐지 키워드': True},
                template = 'plotly_white',
                )
    
    # 피처 중요도에 따른 sort reverse !!!!!
    # 피처 중요도 기준 내림 차순 정렬
    summary_plot.update_traces(textposition="outside")
    
    summary_plot.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True,
                            yaxis = dict(autorange="reversed"),
                            yaxis_categoryorder = 'total descending',
                            legend_itemclick = False, legend_itemdoubleclick = False,
                            title_text='AI 예측 상위 10개 피처 중요도', title_x=0.5,
                            yaxis_title = None
                            )
    
    # plotly to html and all config false
    summary_html = summary_plot.to_html(full_html=False, include_plotlyjs=True,
                                config = {'displaylogo': False,
                                'modeBarButtonsToRemove': ['zoom', 'pan', 'zoomin', 'zoomout', 'autoscale', 'select2d', 'lasso2d',
                                'resetScale2d', 'toImage']
                                }
                                )

    ###################################
    # 1. 전체 피처 중 공격/정상 예측에 영향을 준 상위 10개 피처 비율은 몇 % 이다.
    summary_statement_1 = "전체 피처 중 공격/정상 예측에 영향을 준 상위 10개 피처 비율은 {:.2f}%를 차지.".format(top10_shap_values_sum)
    # 2. 상위 10개 피처 중 공격 예측에 영향을 준 피처는 전체 피처 중 몇 % 이다.
    summary_statement_2 = "상위 10개 피처 중 공격 예측에 영향을 준 피처는 전체 피처 중 {:.2f}%를 차지.".format(top10_shap_values[top10_shap_values['AI 예측 방향'] == '공격']['피처 중요도(%)'].sum())
    ###################################

    
    pie_plot = px.pie(top10_shap_values, values='피처 중요도(%)', names='피처 명',
                                                color = 'AI 예측 방향',
                                                color_discrete_map = {'공격': '#FF0000', '정상': '#00FF00', '기타': '#0000FF'},
                                                template = 'plotly_white',
                                                custom_data = ['피처 설명', '피처 값', 'AI 예측 방향', 'AI 탐지 키워드'],
                                                labels = ['피처 명']
                                                )
    
    # print(top10_shap_values.dtypes)

    # custom_data 에서 피처 설명, 피처 값, AI 예측 방향을 가져와서 ',' 기준 split 하여 표시
    pie_plot.update_traces(textposition='inside', textinfo='label+percent',
                           hovertemplate = '피처 명: %{label}<br>' +
                                            '피처 중요도(%): %{value:.2f}<br>' +
                                            '피처 설명: %{customdata[0][0]}<br>' +
                                            '피처 값: %{customdata[0][1]}<br>' +
                                            'AI 예측 방향: %{customdata[0][2]}<br>' +
                                            'AI 탐지 키워드: %{customdata[0][3]}<br>',

                           hole = 0.3,
                           # hoverinfo = 'label+value'
                            )

    pie_plot.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True,
                           legend_itemclick = False, legend_itemdoubleclick = False,
                            title_text='AI 예측 피처 중요도', title_x=0.5,
                            annotations = [dict(text = '위험도: %d%%<br>%s' %(attack_proba, db_ai),
                            x = 0.5, y = 0.5, 
                            font_color = '#FF0000' if db_ai == '공격' else '#00FF00',
                            font_size = 12, showarrow = False)]
                            )

    pie_plot.update(layout_showlegend=True)
    


    pie_html = pie_plot.to_html(full_html=False, include_plotlyjs=True,
                                config = {'displaylogo': False,
                                'modeBarButtonsToRemove': ['zoom', 'pan', 'zoomin', 'zoomout', 'autoscale', 'select2d', 'lasso2d',
                                'resetScale2d', 'toImage']
                                }
                                )   
    

    # higher: red, lower: green
    shap_cols = payload_df.columns.tolist()
    # payload_df.columns startswith 'ips_' 인 경우, ''로 변경
    shap_cols = [x.replace('ips_', '') for x in shap_cols]cols]

    # force_plot = plt.figure()
    force_plot = shap.force_plot(expected_value_sql[0], shap_values_sql[1], payload_df, link = 'logit',
                        plot_cmap = ['#FF0000', '#00FF00'],
                        feature_names = shap_cols,
                        out_names = '공격',
                        matplotlib = False)

    

    # plt.savefig('static/force_plot.png', bbox_inches = 'tight', dpi = 500)
    force_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"

    # HTML 형태 payload 의 경우, 소괄호 치환 필요
    sig_ai_pattern = re.sub(r'[\\<]', r'%26lt%3B', sig_ai_pattern)
    sig_ai_pattern = re.sub(r'[\\>]', r'%26gt%3B', sig_ai_pattern)

    foreground_regex = r'\x1b\[91m(.*?)\x1b\[39m'
    background_regex = r'\x1b\[103m(.*?)\x1b\[49m'
    
    ################################################################################
    sig_ai_pattern = re.sub(foreground_regex, r'<font color = "red">\1</font>', sig_ai_pattern)
    sig_ai_pattern = re.sub(background_regex, r'<span style = "background-color:yellow;">\1</span>', sig_ai_pattern)

    # </font> ~ </span> 사이를 background-color:yello 추가
    # 단, <font, <span 이 있는 경우 예외 처리
    '''
    </font>
    (?:
    (?<!<font)(?<!<span)
    |
    (?<=<span)
    |
    (?<=<font)
    )
    [^<]*
    (?!<font)(?!<span)
    (?=</span>)
    '''
    # CSS 버전 이슈로 XAI에선 적용 안하기로 함 - 20230308
    # sig_ai_pattern = re.sub(r'</font>(?:(?<!<font)(?<!<span)|(?<=<span)|(?<=<font))[^<]*(?!<font)(?!<span)(?=</span>)',
    #                   r'</font><span style="background-color:yellow;">\g<0></span>', sig_ai_pattern)
    sig_ai_pattern = re.sub(r'\<\/font\>(?:(?<!\<font)(?<!\<span)|(?<=\<span)|(?<=\<font))[^\<]*(?!\<font)(?!\<span)(?=\<\/span\>)',
                       r'</font><span style="background-color:yellow;">\g<0></span>', sig_ai_pattern)

    
    ################################################################################
    
    sig_pattern_html = f"<head>{sig_ai_pattern}</head>"        
    sig_df_html = sig_df.to_html(index=False, justify='center')

    ####################################################################################
    # 딥러닝 기반 XAI
    payload_text_df = pd.DataFrame([raw_data_str], columns = ['payload'])

    try:
        # 512개 이하 토큰
        truncated_text = payload_text_df.iloc[0,0]
        pipe_result = ips_dl_pipe(truncated_text)
    except:
        # error 발생 시, 512개 이하 토큰으로 자르기
        truncated_text = ips_truncate_text(payload_text_df.iloc[0,0])
        pipe_result = ips_dl_pipe(truncated_text)
        
    pipe_result_label = pipe_result[0]['label']
    
    if pipe_result_label == 'POSITIVE':
        pipe_result_label = '공격'
    else:
        pipe_result_label = '정상'
        
    pipe_result_score = pipe_result[0]['score']
    # 정수 표시
    pipe_result_score = int(np.round(pipe_result_score, 2) * 100)
    dl_pred_comment = '인공지능에서 예측한 결과 %s일 확률이 %s%%입니다.' %(pipe_result_label, pipe_result_score)

    masker_check_pattern =  r"\s|%20|\+|\/|%2f|HTTP/1.1|\?|\n|\r|\t"

    if re.search(masker_check_pattern, payload_text_df['payload'][0]):
        IPS_DL_shap_values = IPS_DL_XAI(payload_text_df['payload'], fixed_context=1, batch_size=1)
    else:
        # payload 와 유사하지 않는 이상치에 대한 XAI 재생성
        IPS_DL_XAI_2 = shap.Explainer(ips_bert_predict, IPS_DL_tokenizer)
        IPS_DL_shap_values = IPS_DL_XAI_2(payload_text_df['payload'], fixed_context=1, batch_size=1)

    text_html = shap.text_plot(IPS_DL_shap_values, display = False)

    IPS_DL_shap_values_data = IPS_DL_shap_values.data[0]
    IPS_DL_shap_values_values = IPS_DL_shap_values.values[0]

    dl_xai_df = pd.DataFrame({'AI 탐지 키워드': IPS_DL_shap_values_data,
                        'shap_values': IPS_DL_shap_values_values})

    dl_shap_values_direction = np.where(IPS_DL_shap_values_values >= 0, '공격', '정상')
    
    IPS_DL_shap_values_values_2 = np.abs(IPS_DL_shap_values_values)
    
    IPS_DL_shap_values_values_2_ratio = shap_logit(IPS_DL_shap_values_values_2)
    IPS_DL_shap_values_values_2_ratio = IPS_DL_shap_values_values_2_ratio * 100
    IPS_DL_shap_values_values_2_ratio = np.round(IPS_DL_shap_values_values_2_ratio, 2)
    
    dl_xai_df['피처 중요도(%)'] = IPS_DL_shap_values_values_2_ratio
    dl_xai_df['AI 예측 방향'] = dl_shap_values_direction

    dl_xai_df = dl_xai_df.sort_values(ascending = False, by = '피처 중요도(%)')
    top10_dl_xai = dl_xai_df.head(10)
        
    top10_dl_xai = top10_dl_xai[['AI 탐지 키워드', 'AI 예측 방향', '피처 중요도(%)']]
    # print(top10_dl_xai)
    
    top10_dl_xai_html = top10_dl_xai.to_html(index=False, justify='center')

    
    dl_summary_plot = px.bar(top10_dl_xai, x="피처 중요도(%)", y="AI 탐지 키워드", 
                color='AI 예측 방향', color_discrete_map={'공격': '#FF0000', '정상': '#00FF00', '기타': '#0000FF'},
                text = '피처 중요도(%)',
                orientation='h', hover_data={'피처 중요도(%)': False, 'AI 예측 방향': False,
                                             'AI 탐지 키워드': False},
                template='plotly_white',
            )

    dl_summary_plot.update_traces(textposition="outside")

    dl_summary_plot.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True,
                            yaxis = dict(autorange="reversed"),
                            yaxis_categoryorder = 'total descending',
                            legend_itemclick = False, legend_itemdoubleclick = False,
                            title_text='AI 예측 상위 10개 딥러닝 피처 중요도', title_x=0.5,
                            yaxis_title = None,
                            hovermode = False
                            )
    
    # plotly to html and all config false
    dl_summary_html = dl_summary_plot.to_html(full_html=False, include_plotlyjs=True,
                                config = {'displaylogo': False,
                                'modeBarButtonsToRemove': ['zoom', 'pan', 'zoomin', 'zoomout', 'autoscale', 'select2d', 'lasso2d',
                                'resetScale2d', 'toImage']
                                }
                                )
    
    # 보안 시그니처 패턴 리스트 highlight
    dl_ai_field = top10_dl_xai['AI 탐지 키워드'].tolist()
    print(dl_ai_field)
    dl_sig_ai_pattern, dl_sig_df = dl_highlight_text(raw_data_str, signature_list, dl_ai_field)
    print(dl_sig_ai_pattern)

    # HTML 형태 payload 의 경우, 소괄호 치환 필요
    dl_sig_ai_pattern = re.sub(r'[\\<]', r'%26lt%3B', dl_sig_ai_pattern)
    dl_sig_ai_pattern = re.sub(r'[\\>]', r'%26gt%3B', dl_sig_ai_pattern)

    foreground_regex = r'\x1b\[91m(.*?)\x1b\[39m'
    background_regex = r'\x1b\[103m(.*?)\x1b\[49m'
    
    dl_sig_ai_pattern = re.sub(foreground_regex, r'<font color = "red">\1</font>', dl_sig_ai_pattern)
    dl_sig_ai_pattern = re.sub(background_regex, r'<span style = "background-color:yellow;">\1</span>', dl_sig_ai_pattern)

    dl_sig_pattern_html = f"<head>{dl_sig_ai_pattern}</head>"        
    dl_sig_df_html = dl_sig_df.to_html(index=False, justify='center')
    ####################################################################################

    try:
        # IGLOO XAI 리포트 작성
        start = time.time()
        xai_report_html = xai_llmchain.run({'xai_result': top10_shap_values_html, 'question': default_ques2})
        end = time.time()
        print('IGLOO XAI 리포트 작성: %.2f (초)' %(end - start))

        # 질의 1단계
        # 공격 판단 근거, Tactics ID, 사이버 킬 체인 모델
        start = time.time()
        init_answer_string_1 = base_llmchain.run({'payload': raw_data_str, 'question': default_ques1})
        end = time.time()
        print('공격 판단 근거: %.2f (초)' % (end - start))
        
        start = time.time()
        init_answer_string_2 = base_llmchain.run({'payload': raw_data_str, 'question': ques2})
        end = time.time()
        print('Tactics 추천: %.2f (초)' % (end - start))
        
        start = time.time()
        init_answer_string_3 = base_llmchain.run({'payload': raw_data_str, 'question': ques3_0})
        end = time.time()
        print('사이버 킬 체인 모델: %.2f (초)' % (end - start))

        # 질의 2단계
        # Sigma Rule 추천, 사이버 킬 체인 대응 단계 추천        
        start = time.time()
        continue_answer_string_1 = snort_sigma_llmchain.run({'context': init_answer_string_1, 'sample_rule': sigmarule_file, 'payload': raw_data_str, 'question': ques5})
        end = time.time()
        print('Sigma Rule 추천: %.2f (초)' % (end - start))

        start = time.time()
        continue_answer_string_2 = continue_llmchain.run({'context': init_answer_string_3, 'payload': raw_data_str, 'question': ques3})
        end = time.time()
        print('사이버 킬 체인 대응 단계 추천: %.2f (초)' % (end - start))

        # Snort Rule 추천, CVE 추천
        start = time.time()
        continue_answer_string_3 = snort_sigma_llmchain.run({'context': init_answer_string_1, 'sample_rule': snortrule_file, 'payload': raw_data_str, 'question': ques4})
        end = time.time()
        print('Snort Rule 추천: %.2f (초)' % (end - start))
        
        start = time.time()
        continue_answer_string_4 = continue_llmchain.run({'context': init_answer_string_1, 'payload': raw_data_str, 'question': ques1})
        end = time.time()
        print('CVE 추천: %.2f (초)' % (end - start))
        
        # IGLOO XAI 딥러닝 리포트 작성
        start = time.time()
        dl_xai_report_html = xai_llmchain.run({'xai_result': top10_dl_xai_html, 'question': default_ques2})
        end = time.time()
        print('IGLOO XAI 딥러닝 리포트 작성: %.2f (초)' %(end - start))


        q_and_a_1_df = pd.DataFrame([
                ['공격 판단 근거', init_answer_string_1],
            ], columns=['Question', 'Answer'])
        
        q_and_a_2_df = pd.DataFrame([
                ['Tactics 추천', init_answer_string_2],
                ['Sigma Rule 추천', continue_answer_string_1],
                ['Snort Rule 추천', continue_answer_string_3],
                ['CVE 추천', continue_answer_string_4],
                ['사이버 킬 체인 대응 단계 추천', continue_answer_string_2]
            ], columns=['Question', 'Answer'])
        
        q_and_a_1_html = q_and_a_1_df.to_html(index=False, justify='center')
        q_and_a_2_html = q_and_a_2_df.to_html(index=False, justify='center')
        q_and_a_2_html = q_and_a_2_html.replace('description:', '<br>description:').replace('logsource:', '<br>logsource:').replace('detection:', '<br>detection:').replace('falsepositives:', '<br>falsepositives:').replace('level:', '<br>level:')
    except:
        xai_report_html = '질의 응답 과정에서 오류가 발생했습니다.'
        q_and_a_1_html = '질의에 대한 답변을 생성하는데 실패했습니다.'
        q_and_a_2_html = '질의에 대한 답변을 생성하는데 실패했습니다.'
        dl_xai_report_html = '질의에 대한 답변을 생성하는데 실패했습니다.'


    return render_template('IPS_XAI_output.html', payload_raw_data = request.form['raw_data_str'],  
                                payload_anonymize_highlight_html = payload_anonymize_highlight_html,
                                anony_raw_data = raw_data_str,
                                ml_pred_comment = ml_pred_comment,
                                train_mean_proba_html = train_mean_proba_html,
                                force_html = force_html,
                                summary_html = summary_html,
                                pie_html = pie_html,
                                first_statement = first_statement,
                                second_statement = second_statement,
                                third_statement = third_statement,
                                summary_statement_1 = summary_statement_1,
                                summary_statement_2 = summary_statement_2,
                                sig_pattern_html = sig_pattern_html,
                                sig_df_html = sig_df_html,
                                xai_report_html = xai_report_html,
                                q_and_a_1_html = q_and_a_1_html,
                                q_and_a_2_html = q_and_a_2_html,
                                text_html = text_html,
                                dl_pred_comment = dl_pred_comment,
                                dl_summary_html = dl_summary_html,
                                dl_xai_report_html = dl_xai_report_html,
                                dl_sig_pattern_html = dl_sig_pattern_html,
                                dl_sig_df_html = dl_sig_df_html
                                )


@app.route('/WAF_XAI_result', methods = ['POST'])
def WAF_XAI_result(): 
   # payload의 raw data 입력 값!
    raw_data_str = request.form['raw_data_str']

    # payload의 개행문자 (\n|\r|\t) => ' ' 로 변경
    raw_data_str = re.sub(r'(\n|\r|\t)', ' ', raw_data_str)

    ##########################################################
    # raw_data_str 변수에 XSS 관련 문구가 있어서 창이 나오는 이슈 해결 
    raw_data_str = re.sub(r'[\<]' , '%26lt%3B', raw_data_str)
    raw_data_str = re.sub(r'[\>]' , '%26gt%3B', raw_data_str)
    ##########################################################
    
    # encode to decode
    raw_data_str = payload_decode(raw_data_str)

    # 비식별
    raw_data_str = payload_anonymize(raw_data_str)

    # 비식별 하이라이트
    payload_anonymize_highlight_html = payload_anonymize_highlight(raw_data_str)

    # payload_df = WAF_web_UI_preprocess()
    payload_df = WAF_predict_UI_sql_result()

    payload_arr = np.array(payload_df)

    pred = WAF_model.predict(payload_arr)
    pred_proba = WAF_model.predict_proba(payload_arr)

    if pred == 1:
        db_ai = '공격'
        pred_proba = int(np.round(pred_proba[:, 1], 2) * 100)
    else:
        db_ai = '정상'
        pred_proba = int(np.round(pred_proba[:, 0], 2) * 100)

    ml_pred_comment = '인공지능에서 예측한 결과 %s일 확률이 %s%%입니다.' %(db_ai, pred_proba)
    
    WAF_total_explainer = pickle.load(open(WAF_explainer_path, 'rb'))

    # anomalies : shap_values[1], normal: shap_values[0]
    shap_values_sql = WAF_total_explainer.shap_values(payload_arr)
    shap_values_sql = np.array(shap_values_sql)

    # shap_values_sql[1] 이 0 이상인 경우, 공격, 미만인 경우, 정상으로 판단
    shap_values_sql_direction = np.where(shap_values_sql[1] >= 0, '공격', '정상')
    print(shap_values_sql_direction)
    shap_values_sql_2 = np.abs(shap_values_sql[1]).mean(0)
    shap_values_sql_2_ratio = shap_logit(shap_values_sql_2)
    shap_values_sql_2_ratio = np.round(shap_values_sql_2_ratio, 4)
    print(shap_values_sql_2_ratio)

    shap_values_sql_direction = np.array(shap_values_sql_direction).flatten()
    mean_shap_value_df = pd.DataFrame(list(zip(payload_df.columns, shap_values_sql_2_ratio, shap_values_sql_direction)),
                                   columns=['피처 명','피처 중요도', 'AI 예측 방향'])

    train_mean_df = pd.DataFrame([['모델 평균', expected_value_sql_logit, '기준'], ['예측', attack_proba, attack_proba - expected_value_sql_logit]], 
                        columns = ['모델 평균/예측', '위험도(%)', '위험도(%) 증감'])
    train_mean_df['위험도(%) 증감'][1] = np.round(train_mean_df['위험도(%) 증감'][1], 2)

    if train_mean_df['위험도(%) 증감'][1] < 0:
        train_mean_df['위험도(%) 증감'][1] = train_mean_df['위험도(%) 증감'][1]
    else:
        train_mean_df['위험도(%) 증감'] = train_mean_df['위험도(%) 증감'].astype(str)
        train_mean_df['위험도(%) 증감'][1] = '+' +  train_mean_df['위험도(%) 증감'][1]

    ################################################################
    # expected_value_sql_logit 기반 plotly bar chart 생성 !!!! (기준 100%)
    
    train_mean_proba_plot = px.bar(train_mean_df, x = '위험도(%)',  y = '모델 평균/예측',  
                                        orientation = 'h',
                                        text = '위험도(%)',
                                        hover_data = {'모델 평균/예측': True, '위험도(%)': True, '위험도(%) 증감': True},
                                        color = '모델 평균/예측', 
                                        color_discrete_map = {'모델 평균': '#0000FF', '예측': '#FF0000'},
                                        template = 'plotly_white')



    train_mean_proba_plot.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True,   
                        legend_itemclick = False, legend_itemdoubleclick = False,
                        showlegend = False,
                        title_text='모델 평균/예측 위험도', title_x=0.5,
                        yaxis_title = None,
                        # xaxis_title = None,
                        width = 900,
                        height = 250
                        )
    
    train_mean_proba_html = train_mean_proba_plot.to_html(full_html=False, include_plotlyjs=True,
                            config = {'displaylogo': False,
                            'modeBarButtonsToRemove': ['zoom', 'pan', 'zoomin', 'zoomout', 'autoscale', 'select2d', 'lasso2d',
                            'resetScale2d', 'toImage']
                            }
                            )
    
    
    # mean_shap_value_df 의 피처 중요도를 기준으로 내림차순 정렬
    mean_shap_value_df = mean_shap_value_df.sort_values(by=['피처 중요도'], ascending = False)
    top10_shap_values = mean_shap_value_df.iloc[0:10, :]
    top10_shap_values = top10_shap_values.reset_index(drop = True)

    top10_shap_values['순위'] = top10_shap_values.index + 1

    # 피처 설명 테이블과 join
    top10_shap_values = pd.merge(top10_shap_values, waf_feature_df, how = 'left', on = '피처 명')
    top10_shap_values = top10_shap_values[['순위', '피처 명', '피처 설명', '피처 중요도', 'AI 예측 방향']]

    payload_df_t = payload_df.T
    payload_df_t.columns = ['피처 값']
    # payload_df_t에 피처 명 컬럼 추가
    payload_df_t['피처 명'] = payload_df_t.index
    top10_shap_values = pd.merge(top10_shap_values, payload_df_t, how = 'left', on = '피처 명')
    top10_shap_values = top10_shap_values[['순위', '피처 명', '피처 설명', '피처 값', '피처 중요도', 'AI 예측 방향']]

    # top10_shap_values['피처 명'] 에서 'waf_' 제거
    top10_shap_values['피처 명'] = top10_shap_values['피처 명'].apply(lambda x: x[4:] if x.startswith('waf_') else x)
    
    top10_shap_values['순위'] = top10_shap_values.index + 1
    top10_shap_values  = top10_shap_values[['순위', '피처 명', '피처 설명', '피처 값', '피처 중요도', 'AI 예측 방향']]
    top10_shap_values['피처 중요도'] = top10_shap_values['피처 중요도'].apply(lambda x: round(x, 4))

    # top10_shap_values의 TF-IDF 피처에 대한 피처 설명 추가
    top10_shap_values['피처 설명'] = top10_shap_values['피처 설명'].fillna('AI가 자동 생성한 피처')
    
    # 보안 시그니처 패턴 리스트 highlight
    sig_ai_pattern, sig_df = highlight_text(raw_data_str, signature_list, waf_ai_field)
    print(sig_ai_pattern)

    ai_detect_regex = r'\x1b\[91m(.*?)\x1b\[39m'
    ai_detect_list = re.findall(ai_detect_regex, sig_ai_pattern)
    ai_detect_list = [re.sub(r'\x1b\[103m|\x1b\[49m', '', x) for x in ai_detect_list]

    ###################################################################
    # raw_adta_str 변수에 XSS 관련 문구 떼문에 변경한 부분 원복
    ai_detect_list = [re.sub('%26lt%3B', '<', x) for x in ai_detect_list]
    ai_detect_list = [re.sub('%26gt%3B', '>', x) for x in ai_detect_list]
    ###################################################################

    ai_feature_list = []
    ai_pattern_list = []

    '''
    for x in ai_detect_list:
        for y in waf_auth_field: 
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[0])
                ai_pattern_list.append(y)
        for y in waf_bof_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[1])
                ai_pattern_list.append(y)
        for y in waf_cmd_1_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[2])
                ai_pattern_list.append(y)
        for y in waf_cmd_2_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[3])
                ai_pattern_list.append(y)
        for y in waf_code_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[4])
                ai_pattern_list.append(y)
        for y in waf_dir_1_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[5])
                ai_pattern_list.append(y)
        for y in waf_dir_2_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[6])
                ai_pattern_list.append(y)
        for y in waf_dir_count_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[7])
                ai_pattern_list.append(y)
        for y in waf_cgi_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[8])
                ai_pattern_list.append(y)
        for y in waf_wp_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[9])
                ai_pattern_list.append(y)
        for y in waf_error_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[10])
                ai_pattern_list.append(y)
        for y in waf_file_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[11])
                ai_pattern_list.append(y)
        for y in waf_http_method_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[12])
                ai_pattern_list.append(y)
        for y in waf_malware_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[13])
                ai_pattern_list.append(y)
        for y in waf_rce_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[14])
                ai_pattern_list.append(y)
        for y in waf_sql_1_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[15])
                ai_pattern_list.append(y)
        for y in waf_sql_2_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[16])
                ai_pattern_list.append(y)
        for y in waf_useragent_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[17])
                ai_pattern_list.append(y)
        for y in waf_php_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[18])
                ai_pattern_list.append(y)
        for y in waf_xss_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[19])
                ai_pattern_list.append(y)
        for y in waf_and_count_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[20])
                ai_pattern_list.append(y)
        for y in waf_semico_count_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(waf_feature_list[21])
                ai_pattern_list.append(y)
    '''

    for x in ai_detect_list:
        for i, attack_field in enumerate(waf_feature_value_list):
            for y in attack_field:
                if re.findall(y, x.lower()):
                    ai_feature_list.append(waf_feature_list[i])
                    ai_pattern_list.append(y)

    # ai_feature_list = list(itertools.chain(*ai_feature_list))
    # ai_pattern_list = list(itertools.chain(*ai_pattern_list))
    # ai_pattern_list에사 (.*?) => [~] 로 변경, [%%] => % 로 변경
    ai_pattern_list = [x.replace('(.*?)', '[~]').replace('[%%]', '%') for x in ai_pattern_list]
    # [\\특수문자] => 특수문자 로 변경
    ai_pattern_list = [re.sub(r'(\[\\)(.*?)(\])', r'\2', x, flags = re.I) for x in ai_pattern_list]
    ai_pattern_list = [x.replace('[^a-zA-Z0-9가-힣]', '') for x in ai_pattern_list]

    # ai_feature_list, ai_detect_list 를 이용하여 2개 컬럼 기반 data frame 생성
    print(ai_detect_list)
    print(ai_feature_list)
    print(ai_pattern_list)

    ai_feature_df = pd.DataFrame({'피처 명': ai_feature_list, 'AI 탐지 키워드': ai_pattern_list})

    # ai_feature_df['피처 명'] 중복된 행이 있다면, ',' 기준 concat
    ai_feature_df = ai_feature_df.groupby('피처 명')['AI 탐지 키워드'].apply(', '.join).reset_index()

    # print(ai_feature_df)
    top10_shap_values = top10_shap_values.merge(ai_feature_df, how='left', on='피처 명')

    top10_shap_values['TF-IDF 피처 등장 횟수'] = '-'
    
    # top10_shap_values의 TF-IDF 피처에 AI 공격 탐지 키워드 추가
    for i in range(top10_shap_values.shape[0]):
        if top10_shap_values['피처 명'][i].startswith('token_'):
            # top10_shap_values['피처 명'] 값을 WAF_tfidf_word['feature'] 에서 찾아, WAF_tfidf_word['word'] 값을 가져옴
            top10_shap_values['AI 탐지 키워드'][i] = WAF_tfidf_word.loc[WAF_tfidf_word['feature'] == top10_shap_values['피처 명'][i], 'word'].values[0]
            # TF-IDF 피처의 등장 횟수 가져오기
            top10_shap_values['TF-IDF 피처 등장 횟수'][i] = WAF_tfidf_word.loc[WAF_tfidf_word['feature'] == top10_shap_values['피처 명'][i], 'IDF'].values[0]
            top10_shap_values['TF-IDF 피처 등장 횟수'][i] = top10_shap_values['피처 값'][i] / top10_shap_values['TF-IDF 피처 등장 횟수'][i]

    top10_shap_values['AI 탐지 키워드'] = top10_shap_values['AI 탐지 키워드'].fillna('-')

    top10_shap_values['피처 중요도'] = np.round(top10_shap_values['피처 중요도'] * 100, 2)
    top10_shap_values = top10_shap_values.rename(columns = {'피처 중요도': '피처 중요도(%)'})

    # top10_shap_values의 피처 중요도 합계 
    top10_shap_values_sum = top10_shap_values['피처 중요도(%)'].sum()
    # top10_shap_values_sum_etc = 1 - top10_shap_values_sum
    # etc_df = pd.DataFrame([['기타', '상위 10개 이외 피처', '-', top10_shap_values_sum_etc, '기타']], columns = ['피처 명', '피처 설명', '피처 값', '피처 중요도', 'AI 예측 방향'])
    # top10_shap_values = pd.concat([top10_shap_values, etc_df], axis=0)
    # top10_shap_values = top10_shap_values.sort_values(by='피처 중요도', ascending=False)
    # top10_shap_values = top10_shap_values.reset_index(drop = True)


    ##################################################
    # 학습 데이터 기반 피처 중요도 요약 (상위 3개 피처)
    ##################################################

    first_feature = top10_shap_values.iloc[0, 1]
    first_fv = top10_shap_values.iloc[0, 3]
    first_word = top10_shap_values.iloc[0,-1]
    second_feature = top10_shap_values.iloc[1, 1]
    second_fv = top10_shap_values.iloc[1, 3]
    second_word = top10_shap_values.iloc[1,-1]
    third_feature = top10_shap_values.iloc[2, 1]
    third_fv = top10_shap_values.iloc[2, 3]
    third_word = top10_shap_values.iloc[2,-1]


    if first_feature != waf_attack_dir_conti_feature and first_feature != waf_attack_and_conti_feature and first_feature != waf_attack_semico_conti_feature:
        if first_fv == 1:
            first_fv_result = '공격 탐지'
            first_statement = '%s 가 %s 하였고 AI 탐지 키워드는 %s 입니다.'  %(first_feature, first_fv_result, first_word)
        else:
            first_fv_result = '정상 인식'
            first_statement = '%s 가 %s 하였습니다.' %(first_feature, first_fv_result)  
    elif first_feature == waf_attack_and_conti_feature or first_feature == waf_attack_semico_conti_feature:
        if 'ampersand' in first_feature.lower():
            first_statement = '& 특수문자가 총 %d건 입니다.' %(first_fv)
        else:
            first_statement = '; 특수문자가 총 %d건 입니다.' %(first_fv)
    else:
        first_statement = '상위 디렉토리 접근이 총 %s건 입니다.' % first_fv       

    if second_feature != waf_attack_dir_conti_feature and second_feature != waf_attack_and_conti_feature and second_feature != waf_attack_semico_conti_feature:
        if second_fv == 1:
            second_fv_result = '공격 탐지'
            second_statement = '%s 가 %s 하였고 AI 탐지 키워드는 %s 입니다.'  %(second_feature, second_fv_result, second_word)
        else:
            second_fv_result = '정상 인식'
            second_statement = '%s 가 %s 하였습니다.' %(second_feature, second_fv_result)
    elif second_feature == waf_attack_and_conti_feature or second_feature == waf_attack_semico_conti_feature:
        if 'ampersand' in second_feature.lower():
            second_statement = '& 특수문자가 총 %d건 입니다.' %(second_fv)
        else:
            second_statement = '; 특수문자가 총 %d건 입니다.' %(second_fv)
    else:
        second_statement = '상위 디렉토리 접근이 총 %s건 입니다.' % second_fv      

    if third_feature != waf_attack_dir_conti_feature and third_feature != waf_attack_and_conti_feature and third_feature != waf_attack_semico_conti_feature:
        if third_fv == 1:
            third_fv_result = '공격 탐지'
            third_statement = '%s 가 %s 하였고 AI 탐지 키워드는 %s 입니다.'  %(third_feature, third_fv_result, third_word)
        else:
            third_fv_result = '정상 인식'
            third_statement = '%s 가 %s 하였습니다.' %(third_feature, third_fv_result)
    elif third_feature == waf_attack_and_conti_feature or third_feature == waf_attack_semico_conti_feature:
        if 'ampersand' in third_feature.lower():
            third_statement = '& 특수문자가 총 %d건 입니다.' %(third_fv)
        else:
            third_statement = '; 특수문자가 총 %d건 입니다.' %(third_fv)
    else:
        third_statement = '상위 디렉토리 접근이 총 %s건 입니다.' % third_fv


    # top10_shap_values to html
    top10_shap_values_html = top10_shap_values.to_html(index=False, justify='center')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(top10_shap_values)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')


    # top10_shap_values to plotly                         
    # 피처 중요도에 커서 올리면 피처 설명 나오도록 표시
    # background color = white
    # 피처 중요도 기준 0.5 이상은 '공격' 미만은 '정상'
    # top10_shap_values['AI 예측 방향'] = ['공격' if x >= 0.5 else '정상' for x in top10_shap_values['피처 중요도']]

    summary_plot = px.bar(top10_shap_values, x="피처 중요도(%)", y="피처 명", 
                color = 'AI 예측 방향', color_discrete_map = {'공격': '#FF0000', '정상': '#00FF00', '기타': '#0000FF'},
                text = '피처 중요도(%)', orientation='h', hover_data = {'피처 명': False, '피처 설명': True, '피처 값': True, '피처 중요도(%)': False, 'AI 예측 방향': False,
                                                                    'AI 탐지 키워드': True, 'TF-IDF 피처 등장 횟수': False},
                template = 'plotly_white',
                )
    
    # 피처 중요도에 따른 sort reverse !!!!!
    # 피처 중요도 기준 내림 차순 정렬
    summary_plot.update_traces(textposition="outside")
    
    summary_plot.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True,
                            yaxis = dict(autorange="reversed"),
                            yaxis_categoryorder = 'total descending',
                            legend_itemclick = False, legend_itemdoubleclick = False,
                            title_text='AI 예측 상위 10개 피처 중요도', title_x=0.5,
                            yaxis_title = None
                            )
    
    # plotly to html and all config false
    summary_html = summary_plot.to_html(full_html=False, include_plotlyjs=True,
                                config = {'displaylogo': False,
                                'modeBarButtonsToRemove': ['zoom', 'pan', 'zoomin', 'zoomout', 'autoscale', 'select2d', 'lasso2d',
                                'resetScale2d', 'toImage']
                                }
                                )

    ###################################
    # 1. 전체 피처 중 공격/정상 예측에 영향을 준 상위 10개 피처 비율은 몇 % 이다.
    summary_statement_1 = "전체 피처 중 공격/정상 예측에 영향을 준 상위 10개 피처 비율은 {:.2f}%를 차지.".format(top10_shap_values_sum)
    # 2. 상위 10개 피처 중 공격 예측에 영향을 준 피처는 전체 피처 중 몇 % 이다.
    summary_statement_2 = "상위 10개 피처 중 공격 예측에 영향을 준 피처는 전체 피처 중 {:.2f}%를 차지.".format(top10_shap_values[top10_shap_values['AI 예측 방향'] == '공격']['피처 중요도(%)'].sum())
    ###################################

    
    pie_plot = px.pie(top10_shap_values, values='피처 중요도(%)', names='피처 명',
                                                color = 'AI 예측 방향',
                                                color_discrete_map = {'공격': '#FF0000', '정상': '#00FF00', '기타': '#0000FF'},
                                                template = 'plotly_white',
                                                custom_data = ['피처 설명', '피처 값', 'AI 예측 방향', 'AI 탐지 키워드'],
                                                labels = ['피처 명']
                                                )
    
    # print(top10_shap_values.dtypes)

    # custom_data 에서 피처 설명, 피처 값, AI 예측 방향을 가져와서 ',' 기준 split 하여 표시
    pie_plot.update_traces(textposition='inside', textinfo='label+percent',
                           hovertemplate = '피처 명: %{label}<br>' +
                                            '피처 중요도(%): %{value:.2f}<br>' +
                                            '피처 설명: %{customdata[0][0]}<br>' +
                                            '피처 값: %{customdata[0][1]}<br>' +
                                            'AI 예측 방향: %{customdata[0][2]}<br>' +
                                            'AI 탐지 키워드: %{customdata[0][3]}<br>',
                           hole = 0.3,
                           # hoverinfo = 'label+value'
                            )

    pie_plot.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True,
                           legend_itemclick = False, legend_itemdoubleclick = False,
                            title_text='AI 예측 피처 중요도', title_x=0.5,
                            annotations = [dict(text = '위험도: %d%%<br>%s' %(attack_proba, db_ai),
                            x = 0.5, y = 0.5, 
                            font_color = '#FF0000' if db_ai == '공격' else '#00FF00',
                            font_size = 12, showarrow = False)]
                            )

    pie_plot.update(layout_showlegend=True)
    


    pie_html = pie_plot.to_html(full_html=False, include_plotlyjs=True,
                                config = {'displaylogo': False,
                                'modeBarButtonsToRemove': ['zoom', 'pan', 'zoomin', 'zoomout', 'autoscale', 'select2d', 'lasso2d',
                                'resetScale2d', 'toImage']
                                }
                                )   
    

    # higher: red, lower: green
    shap_cols = payload_df.columns.tolist()
    # payload_df.columns startswith 'waf_' 인 경우, ''로 변경
    shap_cols = [x.replace('waf_', '') for x in shap_cols]p_cols]

    # force_plot = plt.figure()
    force_plot = shap.force_plot(expected_value_sql[0], shap_values_sql[1], payload_df, link = 'logit',
                        plot_cmap = ['#FF0000', '#00FF00'],
                        feature_names = shap_cols,
                        out_names = '공격',
                        matplotlib = False)

    

    # plt.savefig('static/force_plot.png', bbox_inches = 'tight', dpi = 500)
    force_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    
    # HTML 형태 payload 의 경우, 소괄호 치환 필요
    sig_ai_pattern = re.sub(r'[\\<]', r'%26lt%3B', sig_ai_pattern)
    sig_ai_pattern = re.sub(r'[\\>]', r'%26gt%3B', sig_ai_pattern)

    foreground_regex = r'\x1b\[91m(.*?)\x1b\[39m'
    background_regex = r'\x1b\[103m(.*?)\x1b\[49m'
    
    sig_ai_pattern = re.sub(foreground_regex, r'<font color = "red">\1</font>', sig_ai_pattern)
    sig_ai_pattern = re.sub(background_regex, r'<span style = "background-color:yellow;">\1</span>', sig_ai_pattern)

    # </font> ~ </span> 사이를 background-color:yello 추가
    # 단, <font, <span 이 있는 경우 예외 처리
    '''
    </font>
    (?:
    (?<!<font)(?<!<span)
    |
    (?<=<span)
    |
    (?<=<font)
    )
    [^<]*
    (?!<font)(?!<span)
    (?=</span>)
    '''
    
    # CSS 버전 이슈로 XAI에선 적용 안하기로 함 - 20230308
    # sig_ai_pattern = re.sub(r'</font>(?:(?<!<font)(?<!<span)|(?<=<span)|(?<=<font))[^<]*(?!<font)(?!<span)(?=</span>)',
    #                   r'</font><span style="background-color:yellow;">\g<0></span>', sig_ai_pattern)
    sig_ai_pattern = re.sub(r'\<\/font\>(?:(?<!\<font)(?<!\<span)|(?<=\<span)|(?<=\<font))[^\<]*(?!\<font)(?!\<span)(?=\<\/span\>)',
                     r'</font><span style="background-color:yellow;">\g<0></span>', sig_ai_pattern)
    
    sig_pattern_html = f"<head>{sig_ai_pattern}</head>"        
    sig_df_html = sig_df.to_html(index=False, justify='center')
    
    # waf_payload_parsing 함수에서 파싱 진행
    payload_parsing_result_df, payload_parsing_comment = WAF_payload_parsing()
    payload_parsing_result_html = payload_parsing_result_df.to_html(index = False, justify = 'center')

    
    ####################################################################################
    # 딥러닝 기반 XAI
    payload_text_df = pd.DataFrame([raw_data_str], columns = ['payload'])
    
    try:
        # 512개 이하 토큰
        truncated_text = payload_text_df.iloc[0,0]
        pipe_result = waf_dl_pipe(truncated_text)
    except:
        # error 발생 시, 512개 이하 토큰으로 자르기
        truncated_text = waf_truncate_text(payload_text_df.iloc[0,0])
        pipe_result = waf_dl_pipe(truncated_text)
        
    pipe_result_label = pipe_result[0]['label']
    
    if pipe_result_label == 'POSITIVE':
        pipe_result_label = '공격'
    else:
        pipe_result_label = '정상'
        
    pipe_result_score = pipe_result[0]['score']
    # 정수 표시
    pipe_result_score = int(np.round(pipe_result_score, 2) * 100)
    dl_pred_comment = '인공지능에서 예측한 결과 %s일 확률이 %s%%입니다.' %(pipe_result_label, pipe_result_score)

    masker_check_pattern =  r"\s|%20|\+|\/|%2f|HTTP/1.1|\?|\n|\r|\t"
    
    if re.search(masker_check_pattern, payload_text_df['payload'][0]):
        WAF_DL_shap_values = WAF_DL_XAI(payload_text_df['payload'], fixed_context=1, batch_size=1)
    else:
        # payload 와 유사하지 않는 이상치에 대한 XAI 재생성
        WAF_DL_XAI_2 = shap.Explainer(waf_bert_predict, WAF_DL_tokenizer)
        WAF_DL_shap_values = WAF_DL_XAI_2(payload_text_df['payload'], fixed_context=1, batch_size=1)
    
    text_html = shap.text_plot(WAF_DL_shap_values, display = False)

    WAF_DL_shap_values_data = WAF_DL_shap_values.data[0]
    WAF_DL_shap_values_values = WAF_DL_shap_values.values[0]

    dl_xai_df = pd.DataFrame({'AI 탐지 키워드': WAF_DL_shap_values_data,
                        'shap_values': WAF_DL_shap_values_values})

    dl_shap_values_direction = np.where(WAF_DL_shap_values_values >= 0, '공격', '정상')
    
    WAF_DL_shap_values_values_2 = np.abs(WAF_DL_shap_values_values)
    
    WAF_DL_shap_values_values_2_ratio = shap_logit(WAF_DL_shap_values_values_2)
    WAF_DL_shap_values_values_2_ratio = WAF_DL_shap_values_values_2_ratio * 100
    WAF_DL_shap_values_values_2_ratio = np.round(WAF_DL_shap_values_values_2_ratio, 2)
    
    dl_xai_df['피처 중요도(%)'] = WAF_DL_shap_values_values_2_ratio
    dl_xai_df['AI 예측 방향'] = dl_shap_values_direction

    dl_xai_df = dl_xai_df.sort_values(ascending = False, by = '피처 중요도(%)')
    top10_dl_xai = dl_xai_df.head(10)
        
    top10_dl_xai = top10_dl_xai[['AI 탐지 키워드', 'AI 예측 방향', '피처 중요도(%)']]
    # print(top10_dl_xai)
    
    top10_dl_xai_html = top10_dl_xai.to_html(index=False, justify='center')

    
    dl_summary_plot = px.bar(top10_dl_xai, x="피처 중요도(%)", y="AI 탐지 키워드", 
                color='AI 예측 방향', color_discrete_map={'공격': '#FF0000', '정상': '#00FF00', '기타': '#0000FF'},
                text = '피처 중요도(%)',
                orientation='h', hover_data={'피처 중요도(%)': False, 'AI 예측 방향': False,
                                             'AI 탐지 키워드': False},
                template='plotly_white',
            )

    dl_summary_plot.update_traces(textposition="outside")
    
    dl_summary_plot.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True,
                            yaxis = dict(autorange="reversed"),
                            yaxis_categoryorder = 'total descending',
                            legend_itemclick = False, legend_itemdoubleclick = False,
                            title_text='AI 예측 상위 10개 딥러닝 피처 중요도', title_x=0.5,
                            yaxis_title = None,
                            hovermode = False
                            )
    
    # plotly to html and all config false
    dl_summary_html = dl_summary_plot.to_html(full_html=False, include_plotlyjs=True,
                                config = {'displaylogo': False,
                                'modeBarButtonsToRemove': ['zoom', 'pan', 'zoomin', 'zoomout', 'autoscale', 'select2d', 'lasso2d',
                                'resetScale2d', 'toImage']
                                }
                                )
    
    # 보안 시그니처 패턴 리스트 highlight
    dl_ai_field = top10_dl_xai['AI 탐지 키워드'].tolist()
    print(dl_ai_field)
    dl_sig_ai_pattern , dl_sig_df = dl_highlight_text(raw_data_str, signature_list, dl_ai_field)
    print(dl_sig_ai_pattern)

    # HTML 형태 payload 의 경우, 소괄호 치환 필요
    dl_sig_ai_pattern = re.sub(r'[\\<]', r'%26lt%3B', dl_sig_ai_pattern)
    dl_sig_ai_pattern = re.sub(r'[\\>]', r'%26gt%3B', dl_sig_ai_pattern)

    foreground_regex = r'\x1b\[91m(.*?)\x1b\[39m'
    background_regex = r'\x1b\[103m(.*?)\x1b\[49m'
    
    dl_sig_ai_pattern = re.sub(foreground_regex, r'<font color = "red">\1</font>', dl_sig_ai_pattern)
    dl_sig_ai_pattern = re.sub(background_regex, r'<span style = "background-color:yellow;">\1</span>', dl_sig_ai_pattern)

    dl_sig_pattern_html = f"<head>{dl_sig_ai_pattern}</head>"        
    dl_sig_df_html = dl_sig_df.to_html(index=False, justify='center')
    ####################################################################################

    try:
        # IGLOO XAI 리포트 작성
        start = time.time()
        xai_report_html = xai_llmchain.run({'xai_result': top10_shap_values_html, 'question': default_ques2})
        end = time.time()
        print('IGLOO XAI 리포트 작성: %.2f (초)' %(end - start))

        # 질의 1단계
        # 공격 판단 근거, Tactics ID, 사이버 킬 체인 모델
        start = time.time()
        init_answer_string_1 = base_llmchain.run({'payload': raw_data_str, 'question': default_ques1})
        end = time.time()
        print('공격 판단 근거: %.2f (초)' % (end - start))
        
        start = time.time()
        init_answer_string_2 = base_llmchain.run({'payload': raw_data_str, 'question': ques2})
        end = time.time()
        print('Tactics 추천: %.2f (초)' % (end - start))
        
        start = time.time()
        init_answer_string_3 = base_llmchain.run({'payload': raw_data_str, 'question': ques3_0})
        end = time.time()
        print('사이버 킬 체인 모델: %.2f (초)' % (end - start))

        # 질의 2단계
        # Sigma Rule 추천, 사이버 킬 체인 대응 단계 추천        
        start = time.time()
        continue_answer_string_1 = snort_sigma_llmchain.run({'context': init_answer_string_1, 'sample_rule': sigmarule_file, 'payload': raw_data_str, 'question': ques5})
        end = time.time()
        print('Sigma Rule 추천: %.2f (초)' % (end - start))

        start = time.time()
        continue_answer_string_2 = continue_llmchain.run({'context': init_answer_string_3, 'payload': raw_data_str, 'question': ques3})
        end = time.time()
        print('사이버 킬 체인 대응 단계 추천: %.2f (초)' % (end - start))

        # Snort Rule 추천, CVE 추천
        start = time.time()
        continue_answer_string_3 = snort_sigma_llmchain.run({'context': init_answer_string_1, 'sample_rule': snortrule_file, 'payload': raw_data_str, 'question': ques4})
        end = time.time()
        print('Snort Rule 추천: %.2f (초)' % (end - start))
        
        start = time.time()
        continue_answer_string_4 = continue_llmchain.run({'context': init_answer_string_1, 'payload': raw_data_str, 'question': ques1})
        end = time.time()
        print('CVE 추천: %.2f (초)' % (end - start))
        
        # IGLOO XAI 딥러닝 리포트 작성
        start = time.time()
        dl_xai_report_html = xai_llmchain.run({'xai_result': top10_dl_xai_html, 'question': default_ques2})
        end = time.time()
        print('IGLOO XAI 딥러닝 리포트 작성: %.2f (초)' %(end - start))

        
        q_and_a_1_df = pd.DataFrame([
                ['공격 판단 근거', init_answer_string_1],
            ], columns=['Question', 'Answer'])
        
        # q_and_a_2_df = pd.DataFrame([
        #        ['WAF 구문 분석', init_answer_string_4],
        #    ], columns=['Question', 'Answer'])
        
        q_and_a_3_df = pd.DataFrame([
                ['Tactics 추천', init_answer_string_2],
                ['Sigma Rule 추천', continue_answer_string_1],
                ['Snort Rule 추천', continue_answer_string_3],
                ['CVE 추천', continue_answer_string_4],
                ['사이버 킬 체인 대응 단계 추천', continue_answer_string_2]
            ], columns=['Question', 'Answer'])
        
        q_and_a_1_html = q_and_a_1_df.to_html(index=False, justify='center')
        # q_and_a_2_html = q_and_a_2_df.to_html(index=False, justify='center')
        q_and_a_3_html = q_and_a_3_df.to_html(index=False, justify='center')
        q_and_a_3_html = q_and_a_3_html.replace('description:', '<br>description:').replace('logsource:', '<br>logsource:').replace('detection:', '<br>detection:').replace('falsepositives:', '<br>falsepositives:').replace('level:', '<br>level:')
    except:
        xai_report_html = '질의 응답 과정에서 오류가 발생했습니다.'
        q_and_a_1_html = '질의에 대한 답변을 생성하는데 실패했습니다.'
        # q_and_a_2_html = '질의에 대한 답변을 생성하는데 실패했습니다.'
        q_and_a_3_html = '질의에 대한 답변을 생성하는데 실패했습니다.'
        dl_xai_report_html = '질의에 대한 답변을 생성하는데 실패했습니다.'


    return render_template('WAF_XAI_output.html', payload_raw_data = request.form['raw_data_str'],  
                                payload_anonymize_highlight_html = payload_anonymize_highlight_html,
                                anony_raw_data = raw_data_str,
                                ml_pred_comment = ml_pred_comment,
                                train_mean_proba_html = train_mean_proba_html,
                                force_html = force_html,
                                summary_html = summary_html,
                                pie_html = pie_html,
                                first_statement = first_statement,
                                second_statement = second_statement,
                                third_statement = third_statement,
                                summary_statement_1 = summary_statement_1,
                                summary_statement_2 = summary_statement_2,
                                sig_pattern_html = sig_pattern_html,
                                sig_df_html = sig_df_html,
                                xai_report_html = xai_report_html,
                                q_and_a_1_html = q_and_a_1_html,
                                # q_and_a_2_html = q_and_a_2_html,
                                q_and_a_3_html = q_and_a_3_html,                                
                                payload_parsing_result_html = payload_parsing_result_html,
                                payload_parsing_comment = payload_parsing_comment,
                                text_html = text_html,
                                dl_pred_comment = dl_pred_comment,
                                dl_summary_html = dl_summary_html,
                                dl_xai_report_html = dl_xai_report_html,
                                dl_sig_pattern_html = dl_sig_pattern_html,
                                dl_sig_df_html = dl_sig_df_html
                                )


@app.route('/WEB_XAI_result', methods = ['POST'])
def WEB_XAI_result(): 
   # payload의 raw data 입력 값!
    raw_data_str = request.form['raw_data_str']

    # payload의 개행문자 (\n|\r|\t) => ' ' 로 변경
    raw_data_str = re.sub(r'(\n|\r|\t)', ' ', raw_data_str)

    ##########################################################
    # raw_data_str 변수에 XSS 관련 문구가 있어서 창이 나오는 이슈 해결 
    raw_data_str = re.sub(r'[\<]' , '%26lt%3B', raw_data_str)
    raw_data_str = re.sub(r'[\>]' , '%26gt%3B', raw_data_str)
    ##########################################################
    
    # 출발지 IP 의 경우, nginx: 다음 IP 부터 시작
    # NGINX 로그
    '''
    if 'nginx:' in raw_data_str.lower():
        start_ip = re.findall(r'(?<=nginx: )\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', raw_data_str.lower())
    # APACHE 또는 IIS 로그
    else: 
        start_ip = re.findall(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', raw_data_str)
    '''

    ip_pattern = r'((?<!\w\/)(?<!\w\/[0-9])(?<!\w\/[0-9][0-9])(?:[0-9]{1,3}\.){3}[0-9]{1,3})'
    start_ip = re.findall(ip_pattern, raw_data_str)
        
    # encode to decode
    raw_data_str = payload_decode(raw_data_str)

    # 비식별
    raw_data_str = web_payload_anonymize(raw_data_str)

    # 비식별 하이라이트
    payload_anonymize_highlight_html = web_payload_anonymize_highlight(raw_data_str)
    
    # payload_df = WEB_web_UI_preprocess()
    payload_df = WEB_predict_UI_sql_result()

    payload_arr = np.array(payload_df)

    pred = WEB_model.predict(payload_arr)
    pred = pred[0]

    pred_proba = WEB_model.predict_proba(payload_arr)

    cmd_proba = int(np.round(pred_proba[:, 0], 2) * 100)
    sql_proba = int(np.round(pred_proba[:, 2], 2) * 100)
    xss_proba = int(np.round(pred_proba[:, 3], 2) * 100)
    normal_proba = 100 - cmd_proba - sql_proba - xss_proba
    
    total_proba_list = [cmd_proba, normal_proba, sql_proba, xss_proba]
    
    '''
    shap_values_sql 총 4개를 아래 라벨 인덱스별로 파라미터화 !!!!!!!
    변경 필요 !!!!!!!!
    0: CMD_Inj
    1: normal
    2: SQL_Inj
    3: XSS
    '''

    if pred == 0:
        db_ai = 'CMD Injection'
        not_db_ai = '기타 (SQL, XSS, 정상)'
    elif pred == 1:
        db_ai = '정상'
        not_db_ai = '기타 (CMD, SQL, XSS)'
    elif pred == 2:
        db_ai = 'SQL Injection'
        not_db_ai = '기타 (CMD, XSS, 정상)'
    else:
        db_ai = 'XSS'
        not_db_ai = '기타 (CMD, SQL, 정상)'

    pred_max_proba = max(total_proba_list)
    # print(pred_max_proba)

    if pred != 1:
        ml_pred_comment = '인공지능에서 예측한 결과 %s 공격일 확률이 %s%%입니다.' %(db_ai, pred_max_proba)
    else:
        ml_pred_comment = '인공지능에서 예측한 결과 %s일 확률이 %s%%입니다.' %(db_ai, pred_max_proba)

    WEB_total_explainer = pickle.load(open(WEB_explainer_path, 'rb'))

    # 다중 분류 모델의 경우, expected_value 를 TreeExplainer를 모델 구조상 알 수가 없으므로, None 으로 지정 !!!!!!!    
    shap_values_sql = WEB_total_explainer.shap_values(payload_arr)
    shap_values_sql = np.array(shap_values_sql)

    shap_values_sql_direction = np.where(shap_values_sql[pred] >= 0, db_ai, not_db_ai)
    shap_values_sql_2 = np.abs(shap_values_sql[pred]).mean(0)

    shap_values_sql_2_ratio = shap_logit(shap_values_sql_2)
    shap_values_sql_2_ratio = np.round(shap_values_sql_2_ratio, 4)

    shap_values_sql_direction = np.array(shap_values_sql_direction).flatten()
    mean_shap_value_df = pd.DataFrame(list(zip(payload_df.columns, shap_values_sql_2_ratio, shap_values_sql_direction)),
                                   columns=['피처 명','피처 중요도', 'AI 예측 방향'])

    
    # mean_shap_value_df 의 피처 중요도를 기준으로 내림차순 정렬
    mean_shap_value_df = mean_shap_value_df.sort_values(by=['피처 중요도'], ascending = False)
    top10_shap_values = mean_shap_value_df.iloc[0:10, :]
    top10_shap_values = top10_shap_values.reset_index(drop = True)

    top10_shap_values['순위'] = top10_shap_values.index + 1

    # 피처 설명 테이블과 join
    top10_shap_values = pd.merge(top10_shap_values, web_feature_df, how = 'left', on = '피처 명')
    top10_shap_values = top10_shap_values[['순위', '피처 명', '피처 설명', '피처 중요도', 'AI 예측 방향']]

    payload_df_t = payload_df.T
    payload_df_t.columns = ['피처 값']
    # payload_df_t에 피처 명 컬럼 추가
    payload_df_t['피처 명'] = payload_df_t.index
    top10_shap_values = pd.merge(top10_shap_values, payload_df_t, how = 'left', on = '피처 명')
    top10_shap_values = top10_shap_values[['순위', '피처 명', '피처 설명', '피처 값', '피처 중요도', 'AI 예측 방향']]

    top10_shap_values['순위'] = top10_shap_values.index + 1
    top10_shap_values  = top10_shap_values[['순위', '피처 명', '피처 설명', '피처 값', '피처 중요도', 'AI 예측 방향']]
    top10_shap_values['피처 중요도'] = top10_shap_values['피처 중요도'].apply(lambda x: round(x, 4))

    # print(top10_shap_values)

    # 보안 시그니처 패턴 리스트 highlight
    # sig_ai_pattern, sig_df = web_highlight_text(raw_data_str, signature_list, web_ai_field)
    sig_ai_pattern, sig_df = highlight_text(raw_data_str, signature_list, web_ai_field)

    print(sig_ai_pattern)

    ai_detect_regex = r'\x1b\[91m(.*?)\x1b\[39m'
    ai_detect_list = re.findall(ai_detect_regex, sig_ai_pattern)
    ai_detect_list = [re.sub(r'\x1b\[103m|\x1b\[49m', '', x) for x in ai_detect_list]

    ###################################################################
    # raw_adta_str 변수에 XSS 관련 문구 떼문에 변경한 부분 원복
    ai_detect_list = [re.sub('%26lt%3B', '<', x) for x in ai_detect_list]
    ai_detect_list = [re.sub('%26gt%3B', '>', x) for x in ai_detect_list]
    ###################################################################


    ai_feature_list = []
    ai_pattern_list = []

    '''
    for x in ai_detect_list:
        for y in web_sql_1_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(web_feature_list[0])
                ai_pattern_list.append(y)
        for y in web_sql_2_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(web_feature_list[1])
                ai_pattern_list.append(y)
        for y in web_sql_3_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(web_feature_list[2])
                ai_pattern_list.append(y)
        for y in web_sql_4_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(web_feature_list[3])
                ai_pattern_list.append(y)
        for y in web_sql_5_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(web_feature_list[4])
                ai_pattern_list.append(y)
        for y in web_xss_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(web_feature_list[5])
                ai_pattern_list.append(y)
        for y in web_cmd_1_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(web_feature_list[6])
                ai_pattern_list.append(y)
        for y in web_cmd_2_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(web_feature_list[7])
                ai_pattern_list.append(y)
        for y in web_cmd_3_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(web_feature_list[8])
                ai_pattern_list.append(y)
        for y in web_dir_1_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(web_feature_list[9])
                ai_pattern_list.append(y)
        for y in web_dir_count_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(web_feature_list[10])
                ai_pattern_list.append(y)
        for y in web_and_count_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(web_feature_list[11])
                ai_pattern_list.append(y)
        for y in web_semico_count_field:
            if re.findall(y, x.lower()):
                ai_feature_list.append(web_feature_list[12])
                ai_pattern_list.append(y)
    '''

    for x in ai_detect_list:
        for i, attack_field in enumerate(web_feature_value_list):
            for y in attack_field:
                if re.findall(y, x.lower()):
                    ai_feature_list.append(web_feature_list[i])
                    ai_pattern_list.append(y)


    # ai_feature_list = list(itertools.chain(*ai_feature_list))
    # ai_pattern_list = list(itertools.chain(*ai_pattern_list))
    # ai_pattern_list에사 (.*?) => [~] 로 변경, [%%] => % 로 변경
    ai_pattern_list = [x.replace('(.*?)', '[~]').replace('[%%]', '%') for x in ai_pattern_list]
    # [\\특수문자] => 특수문자 로 변경
    ai_pattern_list = [re.sub(r'(\[\\)(.*?)(\])', r'\2', x, flags = re.I) for x in ai_pattern_list]
    ai_pattern_list = [x.replace('[^a-zA-Z0-9가-힣]', '') for x in ai_pattern_list]

    # ai_feature_list, ai_detect_list 를 이용하여 2개 컬럼 기반 data frame 생성
    print(ai_detect_list)
    print(ai_feature_list)
    print(ai_pattern_list)

    ai_feature_df = pd.DataFrame({'피처 명': ai_feature_list, 'AI 탐지 키워드': ai_pattern_list})

    # ai_feature_df['피처 명'] 중복된 행이 있다면, ',' 기준 concat
    ai_feature_df = ai_feature_df.groupby('피처 명')['AI 탐지 키워드'].apply(', '.join).reset_index()


    # print(ai_feature_df)
    top10_shap_values = top10_shap_values.merge(ai_feature_df, how='left', on='피처 명')
    top10_shap_values['AI 탐지 키워드'] = top10_shap_values['AI 탐지 키워드'].fillna('-')

    top10_shap_values['피처 중요도'] = np.round(top10_shap_values['피처 중요도'] * 100, 2)
    top10_shap_values = top10_shap_values.rename(columns = {'피처 중요도': '피처 중요도(%)'})

    # top10_shap_values의 피처 중요도 합계 
    top10_shap_values_sum = top10_shap_values['피처 중요도(%)'].sum()
    # top10_shap_values_sum_etc = 1 - top10_shap_values_sum
    # etc_df = pd.DataFrame([['기타', '상위 10개 이외 피처', '-', top10_shap_values_sum_etc, '기타']], columns = ['피처 명', '피처 설명', '피처 값', '피처 중요도', 'AI 예측 방향'])
    # top10_shap_values = pd.concat([top10_shap_values, etc_df], axis=0)
    # top10_shap_values = top10_shap_values.sort_values(by='피처 중요도', ascending=False)
    # top10_shap_values = top10_shap_values.reset_index(drop = True)


    ##################################################
    # 학습 데이터 기반 피처 중요도 요약 (상위 3개 피처)
    ##################################################

    first_feature = top10_shap_values.iloc[0, 1]
    first_fv = top10_shap_values.iloc[0, 3]
    first_word = top10_shap_values.iloc[0,-1]
    second_feature = top10_shap_values.iloc[1, 1]
    second_fv = top10_shap_values.iloc[1, 3]
    second_word = top10_shap_values.iloc[1,-1]
    third_feature = top10_shap_values.iloc[2, 1]
    third_fv = top10_shap_values.iloc[2, 3]
    third_word = top10_shap_values.iloc[2,-1]


    if first_feature != web_attack_dir_conti_feature and first_feature != web_attack_and_conti_feature and first_feature != web_attack_semico_conti_feature:
        if first_fv == 1:
            first_fv_result = '공격 탐지'
            first_statement = '%s 가 %s 하였고 AI 탐지 키워드는 %s 입니다.'  %(first_feature, first_fv_result, first_word)
        else:
            first_fv_result = '정상 인식'
            first_statement = '%s 가 %s 하였습니다.' %(first_feature, first_fv_result)  
    elif first_feature == web_attack_and_conti_feature or first_feature == web_attack_semico_conti_feature:
        if 'ampersand' in first_feature.lower():
            first_statement = '& 특수문자가 총 %d건 입니다.' %(first_fv)
        else:
            first_statement = '; 특수문자가 총 %d건 입니다.' %(first_fv)
    else:
        first_statement = '상위 디렉토리 접근이 총 %s건 입니다.' % first_fv       

    if second_feature != web_attack_dir_conti_feature and second_feature != web_attack_and_conti_feature and second_feature != web_attack_semico_conti_feature:
        if second_fv == 1:
            second_fv_result = '공격 탐지'
            second_statement = '%s 가 %s 하였고 AI 탐지 키워드는 %s 입니다.'  %(second_feature, second_fv_result, second_word)
        else:
            second_fv_result = '정상 인식'
            second_statement = '%s 가 %s 하였습니다.' %(second_feature, second_fv_result)
    elif second_feature == web_attack_and_conti_feature or second_feature == web_attack_semico_conti_feature:
        if 'ampersand' in second_feature.lower():
            second_statement = '& 특수문자가 총 %d건 입니다.' %(second_fv)
        else:
            second_statement = '; 특수문자가 총 %d건 입니다.' %(second_fv)
    else:
        second_statement = '상위 디렉토리 접근이 총 %s건 입니다.' % second_fv      

    if third_feature != web_attack_dir_conti_feature and third_feature != web_attack_and_conti_feature and third_feature != web_attack_semico_conti_feature:
        if third_fv == 1:
            third_fv_result = '공격 탐지'
            third_statement = '%s 가 %s 하였고 AI 탐지 키워드는 %s 입니다.'  %(third_feature, third_fv_result, third_word)
        else:
            third_fv_result = '정상 인식'
            third_statement = '%s 가 %s 하였습니다.' %(third_feature, third_fv_result)
    elif third_feature == web_attack_and_conti_feature or third_feature == web_attack_semico_conti_feature:
        if 'ampersand' in third_feature.lower():
            third_statement = '& 특수문자가 총 %d건 입니다.' %(third_fv)
        else:
            third_statement = '; 특수문자가 총 %d건 입니다.' %(third_fv)
    else:
        third_statement = '상위 디렉토리 접근이 총 %s건 입니다.' % third_fv


    # top10_shap_values to html
    top10_shap_values_html = top10_shap_values.to_html(index=False, justify='center')
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print(top10_shap_values)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')


    # pred = 1 일 때, 정상, 0, 2, 3 일 때, CMD, SQL, XSS
    if pred != 1:
        summary_plot = px.bar(top10_shap_values, x="피처 중요도(%)", y="피처 명", 
                    color = 'AI 예측 방향', color_discrete_map = {db_ai: '#FF0000', not_db_ai: '#0000FF'},
                    text = '피처 중요도(%)', orientation='h', hover_data = {'피처 명': False, '피처 설명': True, '피처 값': True, '피처 중요도(%)': False, 'AI 예측 방향': False,
                                                                        'AI 탐지 키워드': True},
                    template = 'plotly_white',
                    )
    
    else:
        summary_plot = px.bar(top10_shap_values, x="피처 중요도(%)", y="피처 명", 
            color = 'AI 예측 방향', color_discrete_map = {db_ai: '#00FF00', not_db_ai: '#0000FF'},
            text = '피처 중요도(%)', orientation='h', hover_data = {'피처 명': False, '피처 설명': True, '피처 값': True, '피처 중요도(%)': False, 'AI 예측 방향': False,
                                                                'AI 탐지 키워드': True},
            template = 'plotly_white',
            )


    
    # 피처 중요도에 따른 sort reverse !!!!!
    # 피처 중요도 기준 내림 차순 정렬
    summary_plot.update_traces(textposition="outside")
    
    summary_plot.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True,
                            yaxis = dict(autorange="reversed"),
                            yaxis_categoryorder = 'total descending',
                            legend_itemclick = False, legend_itemdoubleclick = False,
                            title_text='AI 예측 상위 10개 피처 중요도', title_x=0.5,
                            yaxis_title = None
                            )
    
    # plotly to html and all config false
    summary_html = summary_plot.to_html(full_html=False, include_plotlyjs=True,
                                config = {'displaylogo': False,
                                'modeBarButtonsToRemove': ['zoom', 'pan', 'zoomin', 'zoomout', 'autoscale', 'select2d', 'lasso2d',
                                'resetScale2d', 'toImage']
                                }
                                )

    ###################################
    # %s의 경우, db_ai 변수 값이 들어감
    # summary_statement_1 = "상위 10개 피처 중 %s 예측에 영향을 준 피처는 전체 피처 중 {:.2f}%를 차지." .format(top10_shap_values[top10_shap_values['AI 예측 방향'] == db_ai]['피처 중요도(%)'].sum())
    summary_statement_1 = "상위 10개 피처 중 %s 예측에 영향을 준 피처는 전체 피처 중 %.2f%%를 차지." %(db_ai, top10_shap_values[top10_shap_values['AI 예측 방향'] == db_ai]['피처 중요도(%)'].sum())
    ###################################

    '''
    shap_cols = payload_df.columns.tolist()
    if pred != 1:
        force_plot = shap.force_plot(shap_values_sql[pred], payload_df, link = 'logit',
                            plot_cmap = ['#FF0000', '#0000FF'],
                            feature_names = shap_cols,
                            out_names = db_ai,
                            matplotlib = False)
    else:
        force_plot = shap.force_plot(shap_values_sql[pred], payload_df, link = 'logit',
                            plot_cmap = ['#00FF00', '#0000FF'],
                            feature_names = shap_cols,
                            out_names = db_ai,
                            matplotlib = False)
    # plt.savefig('static/force_plot.png', bbox_inches = 'tight', dpi = 500)
    force_html = f"<head>{shap.getjs()}</head><body>{force_plot.html()}</body>"
    '''

    # HTML 형태 payload 의 경우, 소괄호 치환 필요
    sig_ai_pattern = re.sub(r'[\\<]', r'%26lt%3B', sig_ai_pattern)
    sig_ai_pattern = re.sub(r'[\\>]', r'%26gt%3B', sig_ai_pattern)

    foreground_regex = r'\x1b\[91m(.*?)\x1b\[39m'
    background_regex = r'\x1b\[103m(.*?)\x1b\[49m'
    
    sig_ai_pattern = re.sub(foreground_regex, r'<font color = "red">\1</font>', sig_ai_pattern)
    sig_ai_pattern = re.sub(background_regex, r'<span style = "background-color:yellow;">\1</span>', sig_ai_pattern)

    # </font> ~ </span> 사이를 background-color:yello 추가
    # 단, <font, <span 이 있는 경우 예외 처리
    '''
    </font>
    (?:
    (?<!<font)(?<!<span)
    |
    (?<=<span)
    |
    (?<=<font)
    )
    [^<]*
    (?!<font)(?!<span)
    (?=</span>)
    '''
    
    # CSS 버전 이슈로 IGLOO XAI에선 적용 안하기로 함 - 20230308
    # sig_ai_pattern = re.sub(r'</font>(?:(?<!<font)(?<!<span)|(?<=<span)|(?<=<font))[^<]*(?!<font)(?!<span)(?=</span>)',
    #                   r'</font><span style="background-color:yellow;">\g<0></span>', sig_ai_pattern)
    sig_ai_pattern = re.sub(r'\<\/font\>(?:(?<!\<font)(?<!\<span)|(?<=\<span)|(?<=\<font))[^\<]*(?!\<font)(?!\<span)(?=\<\/span\>)',
                     r'</font><span style="background-color:yellow;">\g<0></span>', sig_ai_pattern)
    
    sig_pattern_html = f"<head>{sig_ai_pattern}</head>"        
    sig_df_html = sig_df.to_html(index=False, justify='center')

    ###################################
    # User-Agent 의 browser-type 분류
    # web_payload_parsing 함수에서 user_agent 추출 !!!!!!!!
    web_parsing_result, weblog_type_comment = WEB_payload_parsing()
    # FLASK 적용
    web_parsing_result_html = web_parsing_result.to_html(index = False, justify = 'center')
    
    if weblog_type_comment != 'WEB 로그가 아닙니다.':
        useragent_parsing_result_ori = web_parsing_result['user_agent'][0]

        # 영어 이외의 모든 문자열 제거
        web_parsing_result['user_agent'] = web_parsing_result.apply(lambda x: re.sub(r'[^a-zA-Z]+', ' ', x['user_agent']), axis = 1)
        useragent_parsing_result = web_parsing_result['user_agent'][0]

        useragent_raw_data_df = pd.DataFrame([useragent_parsing_result], columns=['user_agent'])

        valud_tfidf_feature = vectorizer.fit_transform(useragent_raw_data_df['user_agent']).toarray()
        valid_tfidf_df = pd.DataFrame(valud_tfidf_feature, columns=vectorizer.get_feature_names_out())
        # TF * IDF 도출
        valid_tfidf_df = valid_tfidf_df * tfidf_value
        valid_tfidf_df.columns = tfidf_feature
        
        # TF-IDF 피처 값이 0이 아닌 경우, 피처 추출
        valid_tfidf_extract = valid_tfidf_df.loc[:, (valid_tfidf_df != 0).any(axis=0)]
        print(valid_tfidf_extract)

        useragent_pred = WEB_useragent_model.predict(valid_tfidf_df)
        print(useragent_pred)

        if useragent_pred[0] == 'bad_bot_crawler':
            useragent_pred[0] = '악성 봇 크롤러'
        elif useragent_pred[0] == 'normal_bot_crawler':
            useragent_pred[0] = '정상 봇 크롤러'
        else:
            useragent_pred[0] = '애플리케이션'

        useragent_pred_explain = '입력된 WEB Log의 User-Agent는 %s이고, %s에 해당합니다.' %(useragent_parsing_result_ori, useragent_pred[0])

        print('출발지 IP: ', start_ip[0])

        # GeoLite2-Country.mmdb 사용법
        country_reader = geoip2.database.Reader(geoip_country_db_path)
        try:
            country_response = country_reader.country(start_ip[0])
            print(country_response.country.name) # 국가명 조회 (한글은 지원 안함)
            start_ip_country = country_response.country.name
            start_ip_country_explain = '입력된 WEB Log의 출발지 IP 국가 명은 %s 입니다.' %(start_ip_country)
        # GeoIP DB에 없는 경우, 한국으로 지정 !!!!!!!
        except:
            start_ip_country_explain = '입력된 WEB Log의 출발지 IP 국가 명은 조회 결과 존재하지 않습니다.'

    else:
        useragent_pred_explain = 'WEB 로그가 아닙니다.'
        start_ip_country_explain = 'WEB 로그가 아닙니다.'

    ####################################################################################
    # 딥러닝 기반 XAI
    
    # web log 파싱 후, http_method 컬럼부터 추출하여 딥러닝 
    # 학습 데이터 상에서, http_method부터 있었기 때문에, 최대한 예측에 대한 편차를 줄이기 위함 임.
    # after_method_raw_data_str는 raw_data_str에서 web_parsing_result 데이터 프레임의 http_method 컬럼의 값부터 시작하는 것을 split하여 추출
    
    # web_parsing_result에 'http_method' 컬럼이 있는 경우, 즉 정상적으로 파싱 된 경우,
    if 'http_method' in web_parsing_result.columns:
        after_method_raw_data_str = web_parsing_result['http_method'][0] + raw_data_str.split(web_parsing_result['http_method'][0])[1]
    else:
        after_method_raw_data_str = raw_data_str

    payload_text_df = pd.DataFrame([after_method_raw_data_str], columns = ['payload'])

    try:
        # 512개 이하 토큰
        truncated_text = payload_text_df.iloc[0,0]
        pipe_result = web_dl_pipe(truncated_text)
    except:
        # error 발생 시, 512개 이하 토큰으로 자르기
        truncated_text = web_truncate_text(payload_text_df.iloc[0,0])
        pipe_result = web_dl_pipe(truncated_text)
        
    pipe_result_label = pipe_result[0]['label']

    if pipe_result_label == 'LABEL_0':
        pipe_result_label = 'CMD Injection'
        not_pipe_result_label = '기타 (SQL, XSS, 정상)'
    elif pipe_result_label == 'LABEL_1':
        pipe_result_label = '정상'
        not_pipe_result_label = '기타 (CMD, SQL, XSS)'
    elif pipe_result_label == 'LABEL_2':
        pipe_result_label = 'SQL Injection'
        not_pipe_result_label = '기타 (CMD, XSS, 정상)'
    else: 
        # LABEL_3
        pipe_result_label = 'XSS'
        not_pipe_result_label = '기타 (CMD, SQL, 정상)'

    pipe_result_score = pipe_result[0]['score']
    # 정수 표시
    pipe_result_score = int(np.round(pipe_result_score, 2) * 100)
    if pipe_result_label != '정상':
        dl_pred_comment = '인공지능에서 예측한 결과 %s 공격일 확률이 %s%%입니다.' %(pipe_result_label, pipe_result_score)
    else:
        dl_pred_comment = '인공지능에서 예측한 결과 %s일 확률이 %s%%입니다.' %(pipe_result_label, pipe_result_score)

    masker_check_pattern =  r"\s|%20|\+|\/|%2f|HTTP/1.1|\?|\n|\r|\t"

    if re.search(masker_check_pattern, payload_text_df['payload'][0]):
        ####################################################################################
        WEB_DL_XAI = shap.Explainer(lambda x: web_bert_predict(x, pipe_result_label), web_masker)
        ####################################################################################
        WEB_DL_shap_values = WEB_DL_XAI(payload_text_df['payload'], fixed_context=1, batch_size=1)

    else:
        # payload 와 유사하지 않는 이상치에 대한 XAI 재생성
        ####################################################################################
        WEB_DL_XAI_2 = shap.Explainer(lambda x: web_bert_predict(x, pipe_result_label), WEB_DL_tokenizer)
        ####################################################################################
        WEB_DL_shap_values = WEB_DL_XAI_2(payload_text_df['payload'], fixed_context=1, batch_size=1)

    text_html = shap.text_plot(WEB_DL_shap_values, display = False)
        
    WEB_DL_shap_values_data = WEB_DL_shap_values.data[0]
    WEB_DL_shap_values_values = WEB_DL_shap_values.values[0]

    dl_xai_df = pd.DataFrame({'AI 탐지 키워드': WEB_DL_shap_values_data,
                        'shap_values': WEB_DL_shap_values_values})

    dl_shap_values_direction = np.where(WEB_DL_shap_values_values >= 0, pipe_result_label, not_pipe_result_label)

    WEB_DL_shap_values_values_2 = np.abs(WEB_DL_shap_values_values)
    
    WEB_DL_shap_values_values_2_ratio = shap_logit(WEB_DL_shap_values_values_2)
    WEB_DL_shap_values_values_2_ratio = WEB_DL_shap_values_values_2_ratio * 100
    WEB_DL_shap_values_values_2_ratio = np.round(WEB_DL_shap_values_values_2_ratio, 2)
    
    dl_xai_df['피처 중요도(%)'] = WEB_DL_shap_values_values_2_ratio
    dl_xai_df['AI 예측 방향'] = dl_shap_values_direction

    dl_xai_df = dl_xai_df.sort_values(ascending = False, by = '피처 중요도(%)')
    top10_dl_xai = dl_xai_df.head(10)
        
    top10_dl_xai = top10_dl_xai[['AI 탐지 키워드', 'AI 예측 방향', '피처 중요도(%)']]
    # print(top10_dl_xai)
    
    top10_dl_xai_html = top10_dl_xai.to_html(index=False, justify='center')

        
    if pipe_result_label != '정상':
        dl_summary_plot = px.bar(top10_dl_xai, x="피처 중요도(%)", y="AI 탐지 키워드", 
                    color = 'AI 예측 방향', color_discrete_map = {pipe_result_label: '#FF0000', not_pipe_result_label: '#0000FF'},
                    text = '피처 중요도(%)', orientation='h', hover_data = {'피처 중요도(%)': False, 'AI 예측 방향': False,
                                                                        'AI 탐지 키워드': False},
                    template = 'plotly_white',
                    )
    
    else:
        dl_summary_plot = px.bar(top10_dl_xai, x="피처 중요도(%)", y="AI 탐지 키워드", 
                    color = 'AI 예측 방향', color_discrete_map = {pipe_result_label: '#00FF00', not_pipe_result_label: '#0000FF'},
                    text = '피처 중요도(%)', orientation='h', hover_data = {'피처 중요도(%)': False, 'AI 예측 방향': False,
                                                                        'AI 탐지 키워드': False},
                    template = 'plotly_white',
                    )

    dl_summary_plot.update_traces(textposition="outside")
    
    dl_summary_plot.update_layout(xaxis_fixedrange=True, yaxis_fixedrange=True,
                            yaxis = dict(autorange="reversed"),
                            yaxis_categoryorder = 'total descending',
                            legend_itemclick = False, legend_itemdoubleclick = False,
                            title_text='AI 예측 상위 10개 딥러닝 피처 중요도', title_x=0.5,
                            yaxis_title = None,
                            hovermode = False
                            )
    
    # plotly to html and all config false
    dl_summary_html = dl_summary_plot.to_html(full_html=False, include_plotlyjs=True,
                                config = {'displaylogo': False,
                                'modeBarButtonsToRemove': ['zoom', 'pan', 'zoomin', 'zoomout', 'autoscale', 'select2d', 'lasso2d',
                                'resetScale2d', 'toImage']
                                }
                                )
    
    # 보안 시그니처 패턴 리스트 highlight
    dl_ai_field = top10_dl_xai['AI 탐지 키워드'].tolist()
    print(dl_ai_field)
    dl_sig_ai_pattern, dl_sig_df = dl_highlight_text(raw_data_str, signature_list, dl_ai_field)
    print(dl_sig_ai_pattern)


    # HTML 형태 payload 의 경우, 소괄호 치환 필요
    dl_sig_ai_pattern = re.sub(r'[\\<]', r'%26lt%3B', dl_sig_ai_pattern)
    dl_sig_ai_pattern = re.sub(r'[\\>]', r'%26gt%3B', dl_sig_ai_pattern)

    foreground_regex = r'\x1b\[91m(.*?)\x1b\[39m'
    background_regex = r'\x1b\[103m(.*?)\x1b\[49m'
    
    dl_sig_ai_pattern = re.sub(foreground_regex, r'<font color = "red">\1</font>', dl_sig_ai_pattern)
    dl_sig_ai_pattern = re.sub(background_regex, r'<span style = "background-color:yellow;">\1</span>', dl_sig_ai_pattern)

    dl_sig_pattern_html = f"<head>{dl_sig_ai_pattern}</head>"
    dl_sig_df_html = dl_sig_df.to_html(index=False, justify='center')
    ####################################################################################


    try:
        # IGLOO XAI 리포트 작성
        start = time.time()
        xai_report_html = xai_llmchain.run({'xai_result': top10_shap_values_html, 'question': default_ques2})
        end = time.time()
        print('IGLOO XAI 리포트 작성: %.2f (초)' %(end - start))

        # 질의 1단계
        # 공격 판단 근거, Tactics ID, 사이버 킬 체인 모델
        start = time.time()
        init_answer_string_1 = base_llmchain.run({'payload': raw_data_str, 'question': default_ques1})
        end = time.time()
        print('공격 판단 근거: %.2f (초)' % (end - start))
        
        start = time.time()
        init_answer_string_2 = base_llmchain.run({'payload': raw_data_str, 'question': ques2})
        end = time.time()
        print('Tactics 추천: %.2f (초)' % (end - start))
        
        start = time.time()
        init_answer_string_3 = base_llmchain.run({'payload': raw_data_str, 'question': ques3_0})
        end = time.time()
        print('사이버 킬 체인 모델: %.2f (초)' % (end - start))

        # 질의 2단계
        # Sigma Rule 추천, 사이버 킬 체인 대응 단계 추천        
        start = time.time()
        continue_answer_string_1 = snort_sigma_llmchain.run({'context': init_answer_string_1, 'sample_rule': sigmarule_file, 'payload': raw_data_str, 'question': ques5})
        end = time.time()
        print('Sigma Rule 추천: %.2f (초)' % (end - start))

        start = time.time()
        continue_answer_string_2 = continue_llmchain.run({'context': init_answer_string_3, 'payload': raw_data_str, 'question': ques3})
        end = time.time()
        print('사이버 킬 체인 대응 단계 추천: %.2f (초)' % (end - start))

        # Snort Rule 추천, CVE 추천
        start = time.time()
        continue_answer_string_3 = snort_sigma_llmchain.run({'context': init_answer_string_1, 'sample_rule': snortrule_file, 'payload': raw_data_str, 'question': ques4})
        end = time.time()
        print('Snort Rule 추천: %.2f (초)' % (end - start))
        
        start = time.time()
        continue_answer_string_4 = continue_llmchain.run({'context': init_answer_string_1, 'payload': raw_data_str, 'question': ques1})
        end = time.time()
        print('CVE 추천: %.2f (초)' % (end - start))
        
        # IGLOO XAI 딥러닝 리포트 작성
        start = time.time()
        dl_xai_report_html = xai_llmchain.run({'xai_result': top10_dl_xai_html, 'question': default_ques2})
        end = time.time()
        print('IGLOO XAI 딥러닝 리포트 작성: %.2f (초)' %(end - start))


        q_and_a_1_df = pd.DataFrame([
                ['공격 판단 근거', init_answer_string_1],
            ], columns=['Question', 'Answer'])

        # q_and_a_2_df = pd.DataFrame([
        #        ['WEB 구문 분석', init_answer_string_4],
        #    ], columns=['Question', 'Answer'])
        
        q_and_a_3_df = pd.DataFrame([
                ['Tactics 추천', init_answer_string_2],
                ['Sigma Rule 추천', continue_answer_string_1],
                ['Snort Rule 추천', continue_answer_string_3],
                ['CVE 추천', continue_answer_string_4],
                ['사이버 킬 체인 대응 단계 추천', continue_answer_string_2]
            ], columns=['Question', 'Answer'])

        
        q_and_a_1_html = q_and_a_1_df.to_html(index=False, justify='center')
        # q_and_a_2_html = q_and_a_2_df.to_html(index=False, justify='center')
        q_and_a_3_html = q_and_a_3_df.to_html(index=False, justify='center')
        q_and_a_3_html = q_and_a_3_html.replace('description:', '<br>description:').replace('logsource:', '<br>logsource:').replace('detection:', '<br>detection:').replace('falsepositives:', '<br>falsepositives:').replace('level:', '<br>level:')
            
    except:
        xai_report_html = '질의 응답 과정에서 오류가 발생했습니다.'
        q_and_a_1_html = '질의 응답 과정에서 오류가 발생했습니다.'
        # q_and_a_2_html = '질의 응답 과정에서 오류가 발생했습니다.'
        q_and_a_3_html = '질의 응답 과정에서 오류가 발생했습니다.'
        dl_xai_report_html = '질의에 대한 답변을 생성하는데 실패했습니다.'

    return render_template('WEB_XAI_output.html', payload_raw_data = request.form['raw_data_str'],  
                                payload_anonymize_highlight_html = payload_anonymize_highlight_html,
                                anony_raw_data = raw_data_str,
                                # train_mean_proba_html = train_mean_proba_html,
                                # force_html = force_html,
                                ml_pred_comment = ml_pred_comment,
                                summary_html = summary_html,
                                # pie_html = pie_html,
                                first_statement = first_statement,
                                second_statement = second_statement,
                                third_statement = third_statement,
                                summary_statement_1 = summary_statement_1,
                                sig_pattern_html = sig_pattern_html,
                                sig_df_html = sig_df_html,
                                xai_report_html = xai_report_html,
                                q_and_a_1_html = q_and_a_1_html,
                                # q_and_a_2_html = q_and_a_2_html,
                                q_and_a_3_html = q_and_a_3_html,                                
                                web_parsing_result_html = web_parsing_result_html,
                                weblog_type_comment = weblog_type_comment,
                                useragent_pred_explain = useragent_pred_explain,
                                start_ip_country_explain = start_ip_country_explain,
                                text_html = text_html,
                                dl_pred_comment = dl_pred_comment,
                                dl_summary_html = dl_summary_html,
                                dl_xai_report_html = dl_xai_report_html,
                                dl_sig_pattern_html = dl_sig_pattern_html,
                                dl_sig_df_html = dl_sig_df_html
                                )



if __name__ == '__main__':
   app.run(host = SERVER_IP, port = PORT, debug = True)
