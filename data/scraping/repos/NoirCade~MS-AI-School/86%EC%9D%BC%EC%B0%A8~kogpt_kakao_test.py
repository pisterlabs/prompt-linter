# coding=utf8
# REST API 호출에 필요한 라이브러리
import requests
import json
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

import datetime

from selenium import webdriver

import os
import openai
# coding=utf8
# REST API 호출에 필요한 라이브러리


# Start a webdriver (e.g. Chrome)
driver = webdriver.Chrome()

juso = "https://finance.daum.net/quotes/A005930#news/stock"

# Navigate to a website
driver.get(juso)

# Find an element using XPath
#abc = driver.find_element_by_xpath("/html/body/div/table[1]")


def doScrollDown(whileSeconds):
    start = datetime.datetime.now()
    end = start + datetime.timedelta(seconds=whileSeconds)
    while True:
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
        time.sleep(1)
        if datetime.datetime.now() > end:
            break
scroll = doScrollDown(2)
scroll


element = driver.find_element(By.XPATH,"//*[@id='boxContents']/div[5]/div[1]/div[2]/div/ul/li[1]/span/a[1]")




time.sleep(3)
print('기사제목= ',element.text)
element.click()
#element = driver.find_element(By.XPATH,"/html/body/div/table[1]/tbody/tr[1]/td[1]/a").click
#rows = element.find_elements(By.XPATH,".//tr")
# Interact with the element (e.g. get text, click, etc.)
time.sleep(3)
texts = driver.find_element(By.XPATH,"//*[@id='boxApp']/div[3]/div[1]")
reallytexts = texts.find_element(By.XPATH,"//*[@id='dmcfContents']/section")
print(reallytexts.text)
news=reallytexts.text
time.sleep(3)  # 5초 대기
driver.quit()  # 브라우저 종료

#
# import bs4
# import requests
# url = juso
# result = requests.get(url).text
#
#
# steamObj = bs4.BeautifulSoup(result, 'html.parser')
# abcd = steamObj.find_all('a')
#
#
# print(abcd)
#
# exit()




news=news+ ' \n 한줄요약:'







# [내 애플리케이션] > [앱 키] 에서 확인한 REST API 키 값 입력
REST_API_KEY = '2ad7a7523a0ea6f5132526a4910c7beb'

# KoGPT API 호출을 위한 메서드 선언
# 각 파라미터 기본값으로 설정
def kogpt_api(prompt, max_tokens = 1, temperature = 1.0, top_p = 1.0, n = 1):
    r = requests.post(
        'https://api.kakaobrain.com/v1/inference/kogpt/generation',
        json = {
            'prompt': prompt,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'n': n
        },
        headers = {
            'Authorization': 'KakaoAK ' + REST_API_KEY,
            'Content-Type': 'application/json'
        }
    )
    # 응답 JSON 형식으로 변환
    response = json.loads(r.content)
    return response

# # KoGPT에게 전달할 명령어 구성
# prompt = '''난 너를 사랑해'''
#
# # 파라미터를 전달해 kogpt_api()메서드 호출
# response = kogpt_api(
#     prompt = prompt,
#     max_tokens = 32,
#     temperature = 1.0,
#     top_p = 1.0,
#     n = 3
# )
#


prompt= '''news
한줄요약:
'''
response = kogpt_api(
    prompt = news,
    max_tokens = 128,
    temperature = 1.0,
    top_p = 0.7,
    n = 3
)

print(list(response.values())[1][1])