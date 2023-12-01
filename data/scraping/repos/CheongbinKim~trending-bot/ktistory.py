import os
import re
import asyncio
import openai
import requests
import json

from kselenium import KSelenium 
from selenium.webdriver.common.by import By

from requests_toolbelt import MultipartEncoder

from klogging import *

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.environ.get('OPENAI_KEY')
kakao_id = os.environ.get("KAKAO_ID")
kakao_pw = os.environ.get("KAKAO_PW")
api_id = os.environ.get("API_ID")
api_pw = os.environ.get("API_PW")
api_callback = os.environ.get("CALLBACK")
blog_name = os.environ.get("BLOG_NAME")

async def addThum(image_url,access_token):
    info(image_url)
    # 대상 서버 URL (이미지를 전송할 서버의 엔드포인트)
    target_url = f'https://tistory.com/apis/post/attach?access_token={access_token}&blogName={blog_name}&targetUr={blog_name}&output=json'

    image_extension = os.path.splitext(image_url)[-1].lower()

    # 이미지 확장자에 따라 Content-Type 설정
    if image_extension == '.jpg' or image_extension == '.jpeg':
        content_type = 'image/jpeg'
    elif image_extension == '.png':
        content_type = 'image/png'
    elif image_extension == '.gif':
        content_type = 'image/gif'
    elif image_extension == '.tiff':
        content_type = 'image/tiff'
    elif image_extension == '.webp':
        content_type = 'image/webp'
    else:
        content_type = 'application/octet-stream'  # 기본값으로 설정

    # MultipartEncoder를 사용하여 데이터 준비
    multipart_data = MultipartEncoder(
        fields={
            'uploadedfile': ('image' + image_extension, requests.get(image_url).content, content_type)
        }
    )

    info(multipart_data.content_type)

    # 요청 헤더 설정 (multipart/form-data로 전송)
    headers = {
        'Content-Type': multipart_data.content_type,
        'Authorization': f'Bearer {access_token}'
    }

    # 파일을 대상 서버로 전송
    response = requests.post(target_url, headers=headers, data=multipart_data)
    
    thum_url = json.loads(response.text)['tistory']['url']

    info(thum_url)

    # 임시 파일 삭제
    if response.status_code == 200:
        info('이미지 업로드 성공')
    else:
        info('이미지 업로드 실패. 응답 코드:', response.status_code)

    

    return thum_url

async def clickHeart(url):
    info(url)
    driver = KSelenium()
    driver.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: ()=> undefined});")
    driver.go(url)
    await asyncio.sleep(5)
    btn = driver.get('//*[@id="content"]/div/div[2]/div[4]/div[1]/div[1]/button')
    btn.click()
    driver.close()

async def sendCodeCallback():
    driver = KSelenium()
    driver.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: ()=> undefined});")

    driver.go(f'https://www.tistory.com/oauth/authorize?client_id={api_id}&redirect_uri={api_callback}&response_type=code')
    # kakao login

    el = driver.get('//*[@id="cMain"]/div/div/div/a[2]')
    el.click()

    await asyncio.sleep(3)

    el = driver.get('//*[@id="loginId--1"]')
    el.send_keys(kakao_id)

    el = driver.get('//*[@id="password--2"]')
    el.send_keys(kakao_pw)

    el = driver.get('//*[@id="mainContent"]/div/div/form/div[4]/button[1]')
    el.click()

    await asyncio.sleep(3)

    #driver.driver.save_screenshot('test.png')

    el = driver.get('//*[@id="contents"]/div[4]/button[1]')

    el.click()

    await asyncio.sleep(1)

    driver.close()

def text_to_html(text):
    text = re.sub(r'\n\s*\n', '\n', text)

    lines = text.split('\n')

    html_text = ''
    for line in lines:
        mg = re.match('\d+\.\s[\w\s\S]+:',line)
        if mg:
            html_text = html_text + f'<p style="font-size:20px;font-weight=bold;">{line}</p>\n'
        else:
            html_text = html_text + f'<p>{line}</p>\n'


    text = ''.join(f'<p>{line.strip()}</p>\n' for line in lines)
    return text.replace("\n","<br>")

async def getTrendsToWrite(access_token):
    driver = KSelenium()

    driver.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: ()=> undefined});")

    driver.go('https://trends.google.com/trends/trendingsearches/daily?geo=US&hl=en-US')

    await asyncio.sleep(5)

    
    search_date = driver.get('/html/body/div[3]/div[2]/div/div[2]/div/div[1]/ng-include/div/div/div/div[1]/div/div[1]')
    search_date = search_date.text
    info(search_date)
    for i in range(1):
        #keyword = driver.get('/html/body/div[3]/div[2]/div/div[2]/div/div[1]/ng-include/div/div/div/div/md-list[1]/feed-item/ng-include/div/div/div[1]/div[2]/div[2]/div[1]/div[1]/a')
    
        f = open('./ranking','r')
        ranking = f.readline()
        if '\n' in ranking:
            ranking = re.sub(r'\n','',ranking)
            f.close()
            f = open('./ranking','w')
            if ranking == '8':
                f.write('1')
                f.close()
            else:
                f.write(str(int(ranking)+1))
                f.close()

        keyword = driver.get(f'/html/body/div[3]/div[2]/div/div[2]/div/div[1]/ng-include/div/div/div/div/md-list[{ranking}]/feed-item/ng-include/div/div/div[1]/div[2]/div[2]/div[1]/div[1]/a')

        gpt_keyword = keyword.text

        origin_link = keyword.get_attribute('href')

        info(origin_link)

        origin = f'<p data-ke-size="size16"><a href="{origin_link}">{origin_link}</a>&nbsp;</p>'

        driver.go(origin_link)

        await asyncio.sleep(5)

        og_image_url = ''

        try:
            og_image_tag = driver.driver.find_element(By.CSS_SELECTOR,"meta[property='og:image']")
            og_image_url = og_image_tag.get_attribute("content")
            info(og_image_url)
            og_image_url = await addThum(og_image_url,access_token)
            info(og_image_url)
            await asyncio.sleep(1)
        except:
            info('og:image 가져오기 실패, 자동생성 썸네일 추가 필요')
            og_image_url = ''
                
        origin_img = f'<p><img src={og_image_url}></p>'

        content = f'{search_date} Please create a blog about {gpt_keyword} topics in html format, no footer, only body without body tag'

        info(content)

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user",'content':content}],
        )

        answer = response['choices'][0]['message']['content']

        write_params = {
            'access_token': access_token,
            'output': 'json',
            'blogName': blog_name,
            'title': gpt_keyword,
            'visibility':'3',
            'content': origin_img + origin + answer,
            'tag': 'GoogleTrends,ChatGPT,TistoryAPI,자동글쓰기',
            'acceptComment':'1',
            'category':'678557'
        }

        response = requests.post('https://www.tistory.com/apis/post/write',params=write_params)

        json_res = response.json()

        hit_url = json_res['tistory']['url']

        info(hit_url)

        driver.close()

        for i in range(5):
            await clickHeart(hit_url)
    

