from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time, os
import urllib.request
import json
import base64
import requests
import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import openai
import pyttsx3
from pydub import AudioSegment
from pydub.playback import play
from PIL import Image


## 여기부터
# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context
## 여기까지 MAC 용

#소리내 읽기 (TTS)
def playsound_with_wait(file_name):
    sound = AudioSegment.from_file(file_name, format='mp3')
    play(sound)

def speak_origin(text):
    print('[Eyescape]'+ text)
    # result.txt에 추가
    with open('result.txt', 'a', encoding='utf-8') as f:
        f.write('[Eyescape]'+ text + '\n')
    file_name = 'voice.mp3'
    tts = gTTS(text=text, lang='ko')
    tts.save(file_name)

    sound = AudioSegment.from_file(file_name, format='mp3')
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={"frame_rate": int(sound.frame_rate * 1.15)})
    # sound_with_altered_frame_rate = sound_with_altered_frame_rate[:-50]
    sound_with_altered_frame_rate.export(file_name, format='mp3')

    # 재생
    playsound_with_wait(file_name)
    if os.path.exists(file_name): # voice.mp3 파일 삭제 -> 권한 문제가 생겨서 제대로 안 될 수가 있어서
        os.remove(file_name)


def get_files_count(folder_path):
    files = os.listdir(folder_path)
    return len(files)


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        elif os.path.exists(directory):
            for file in os.scandir(directory):
                os.remove(file.path)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def product_lists():
    products_name=driver.find_elements(By.CSS_SELECTOR, "#productList > li > a > dl > dd > div > div.name")
    products_price=driver.find_elements(By.CSS_SELECTOR, "#productList > li > a > dl > dd > div > div.price-area > div.price-wrap > div.price > em")
    for i in range(len(products_name)):
        products_name[i] = products_name[i].text
        products_price[i] = products_price[i].text
    # for i in range(len(products_name)):
    #     print(f"{i+1}. {products_name[i]}, {products_price[i]}")

    elem = driver.find_element(By.TAG_NAME, "body")
    elem.send_keys(Keys.PAGE_DOWN)

    for i in range(len(products_name)-33):
        speak_origin(f"{i+1}. {products_name[i]}, {products_price[i]}")
        # 맥은 시간 조정 바람 (playsound가 플레이어마다 다르게 동작한다고 함.)
        #time.sleep(5)
    elem.send_keys(Keys.PAGE_DOWN)

def textCrawling():
    list = []
    try:
        driver.find_element(By.CSS_SELECTOR, '#itemBrief > div > p.essential-info-more > a').click()
    except:
        pass

    elem_detail = driver.find_element(By.TAG_NAME, "body")
    for i in range(60):
        elem_detail.send_keys(Keys.PAGE_DOWN)
    
    product = driver.find_element(By.CSS_SELECTOR, '.prod-buy-header__title').text
    list.append("상품명: " + product)

    price = driver.find_elements(By.CSS_SELECTOR, '.total-price > strong')
    list.append("가격: " + price[0].text)
    try:
        list.append("와우 할인가: " + price[1].text)
    except:
        pass

    try:
        unit_price = driver.find_element(By.CSS_SELECTOR, '#contents > div.prod-atf > div > div.prod-buy.new-oos-style.not-loyalty-member.eligible-address.dawn-only-product.without-subscribe-buy-type.DISPLAY_0.only-one-delivery > div.prod-price-container > div.prod-price > div > div.prod-coupon-price.price-align.major-price-coupon > span.unit-price.font-medium').text
        list.append(unit_price)
    except:
        pass

    star = driver.find_element(By.CSS_SELECTOR, '#btfTab > ul.tab-contents > li.product-review.tab-contents__content > div > div.sdp-review__average.js_reviewAverageContainer > section.sdp-review__average__total-star > div.sdp-review__average__total-star__info > div.sdp-review__average__total-star__info-gray > div')
    list.append("별점(평점): " + star.get_attribute('data-rating'))

    if(driver.find_element(By.CSS_SELECTOR, '.prod-shipping-fee-message > span').text != ""):
        shipping_fee = driver.find_element(By.CSS_SELECTOR, '.prod-shipping-fee-message > span').text
        list.append("배송비: " + shipping_fee)
    else:
        shipping_fee1 = driver.find_element(By.CSS_SELECTOR, '#contents > div.prod-atf > div > div.prod-buy.new-oos-style.not-loyalty-member.eligible-address.without-subscribe-buy-type.DISPLAY_0 > div.prod-shipping-fee-and-pdd-wrapper.apply-unknown-customer-handling > div.shipping-fee-list > div.radio-list-item.SHIPPING_FEE_NUDGE_DISPLAY_0.selected > span.shipping-fee-list-item.inline-flex-v-center').text
        shipping_fee2 = driver.find_element(By.CSS_SELECTOR, '#contents > div.prod-atf > div > div.prod-buy.new-oos-style.not-loyalty-member.eligible-address.without-subscribe-buy-type.DISPLAY_0 > div.prod-shipping-fee-and-pdd-wrapper.apply-unknown-customer-handling > div.shipping-fee-list > div.radio-list-item.SHIPPING_FEE_NUDGE_DISPLAY_1 > span.shipping-fee-list-item.inline-flex-v-center').text
        list.append("배송비: " + shipping_fee1 + " 또는 " + shipping_fee2)

    if driver.find_element(By.CSS_SELECTOR, '.prod-pdd-container > div').text != "":
        shipping_day = driver.find_element(By.CSS_SELECTOR, '.prod-pdd-container > div').text
        list.append("배송 예정일: " + shipping_day)
    else:
        ship = driver.find_elements(By.CSS_SELECTOR, '.prod-pdd-list > div')
        list.append("배송 예정일: " + ship[0].text + " 또는 " + ship[1].text)

    try:
        deliver_info = driver.find_element(By.CSS_SELECTOR, "#contents > div.prod-atf > div > div.prod-buy.new-oos-style.not-loyalty-member.eligible-address.without-subscribe-buy-type.DISPLAY_0.has-loyalty-exclusive-price.with-seller-store-rating > div.prod-vendor-container").text
        list.append(deliver_info)
    except:
        pass

    try:
        option=driver.find_element(By.CSS_SELECTOR, '#contents > div.prod-atf > div > div.prod-buy.new-oos-style.not-loyalty-member.eligible-address.without-subscribe-buy-type.DISPLAY_1 > div.prod-shipping-fee-and-pdd-wrapper.apply-unknown-customer-handling').text
        list.append(option)
    except:
        pass

    try:
        options=driver.find_element(By.CSS_SELECTOR, '#contents > div.prod-atf > div > div.prod-buy.new-oos-style.not-loyalty-member.eligible-address.without-subscribe-buy-type.DISPLAY_0 > div.prod-option > div > div > div > button > table > tbody > tr > td:nth-child(1)').text
        list.append(options)
    except:
        pass

    try: 
        production = driver.find_element(By.CSS_SELECTOR, '#contents > div.prod-atf > div > div.prod-buy.new-oos-style.not-loyalty-member.eligible-address.without-subscribe-buy-type.DISPLAY_0 > div.prod-description').text
        list.append(production)
    except:
        pass

    try:
        production = driver.find_element(By.CSS_SELECTOR, '#contents > div.prod-atf > div > div.prod-buy.new-oos-style.not-loyalty-member.eligible-address.dawn-only-product.without-subscribe-buy-type.DISPLAY_0.only-one-delivery > div.prod-description').text
        list.append(production)
    except:
        pass

    table1 = driver.find_element(By.CSS_SELECTOR, '#itemBrief > div')
    tbody1 = table1.find_element(By.TAG_NAME, "tbody")
    rows1 = tbody1.find_elements(By.TAG_NAME, "tr")
    for tr in rows1:
        ths = tr.find_elements(By.TAG_NAME, "th")
        tds = tr.find_elements(By.TAG_NAME, "td")
        for th, td in zip(ths, tds):
            list.append(th.text.strip() + ": " + td.text.strip())

    def get_page_data():
        users = driver.find_elements(By.CSS_SELECTOR, '.sdp-review__article__list__info__user__name.js_reviewUserProfileImage')
        ratings = driver.find_elements(By.CSS_SELECTOR, '.sdp-review__article__list__info__product-info__star-orange.js_reviewArticleRatingValue')
        headline = driver.find_elements(By.CSS_SELECTOR, '.sdp-review__article__list__headline')
        contents = driver.find_elements(By.CSS_SELECTOR, '.sdp-review__article__list__review__content.js_reviewArticleContent')
        survey = driver.find_elements(By.CSS_SELECTOR, '.sdp-review__article__list__survey')
            
        if len(users) == len(ratings):            
            for index in range(len(users)):
                data = {}
                data['사용자'] = users[index].text
                data['평점'] = int(ratings[index].get_attribute('data-rating'))
                try:
                    data['리뷰 제목'] = headline[index].text
                except:
                    data['리뷰 제목'] = ""
                try:
                    data['리뷰 내용'] = contents[index].text
                except:
                    data['리뷰 내용'] = ""
                try:
                    data['설문 조사'] = survey[index].text
                except:
                    data['설문 조사'] = ""
                list.append(data)
                
    # get_page_data() 
    return list
    

def crawling():
    createFolder('./images/')
    links = []
    detail_images = driver.find_elements(By.ID, "productDetail")
    for detail_image in detail_images:
        if detail_image.find_elements(By.TAG_NAME, "img") != None:
            detail_images = detail_image.find_elements(By.TAG_NAME, "img")
            for detail_image in detail_images:
                if detail_image.get_attribute('src') != None:
                    links.append(detail_image.get_attribute('src'))

    for k, i in enumerate(links):
        url = i
        urllib.request.urlretrieve(url, './images/'+str(k)+'.png')

    speak_origin('상품 정보를 불러오고 있습니다. 잠시만 기다려주세요...')

    list = []
    list = textCrawling()
    speak_origin('상품 텍스트 정보를 불러왔습니다. 이미지 정보를 불러오고 있으니 잠시만 기다려주세요...')
    return list


# crawling
options = webdriver.ChromeOptions()
options.add_experimental_option("excludeSwitches", ["enable-logging"])

driver = webdriver.Chrome(service=Service('chromedriver.exe'), options=options)
driver.implicitly_wait(3)
driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"})
driver.get('https://www.coupang.com')

count = 0
while True:
    r1 = sr.Recognizer()
    m1 = sr.Microphone()

    name = ''
    if count == 0:
        speak_origin('상품명을 말씀해주세요.')
    else:
        speak_origin('상품명을 다시 말씀해주세요.')

    #음성 인식 (STT)
    def listen_object(recognizer, audio):
        try:
            text = recognizer.recognize_google(audio, language='ko')
            print('[사용자]'+ text)
            with open('result.txt', 'a', encoding='utf-8') as f:
                f.write('[사용자]'+ text + '\n')
            global name 
            name = text

        except sr.UnknownValueError:
            print('인식 대기')
        except sr.RequestError as e:
            print('요청 실패 : {0}'.format(e))


    stop_listening1 = r1.listen_in_background(m1, listen_object)
    time.sleep(7)

    del r1
    del m1
    stop_listening1(wait_for_stop=False) #더 이상 듣지 않음

    r0 = sr.Recognizer()
    m0 = sr.Microphone()
    checkKeyword = ''

    # 맥은 time.sleep 지우고 한번에 처리
    speak_origin(f'검색을 원하는 키워드가 {name} 맞나요?')

    #음성 인식 (STT)
    def listen_object(recognizer, audio):
        try:
            text = recognizer.recognize_google(audio, language='ko')
            print('[사용자]'+ text)
            with open('result.txt', 'a', encoding='utf-8') as f:
                f.write('[사용자]'+ text + '\n')
            global checkKeyword 
            checkKeyword = text

        except sr.UnknownValueError:
            print('인식 대기')
        except sr.RequestError as e:
            print('요청 실패 : {0}'.format(e))
    
    stop_listening0 = r0.listen_in_background(m0, listen_object)
    time.sleep(7)

    del r0
    del m0
    stop_listening0(wait_for_stop=False) #더 이상 듣지 않음
    
    if(checkKeyword == '예' or checkKeyword == '네' or checkKeyword == '맞아요' or checkKeyword == '네 맞아요' or checkKeyword == '네 맞아' or checkKeyword == '네 맞습니다' or checkKeyword == '맞습니다' or checkKeyword == '맞아' or checkKeyword == '응 맞아' or checkKeyword == '응'):
        speak_origin('검색 중입니다. 잠시만 기다려주세요.')
        break
    else:
        count += 1


elem = driver.find_element(By.NAME, "q")
elem.send_keys(name)
elem.send_keys(Keys.RETURN)

if(name != ''):
    speak_origin(f'{name} 검색 결과입니다.')

product_lists()

# while True:
#     next_page=input("다음 페이지로 넘어가시겠습니까? 예, 아니오로 대답해주세요. ")
#     if next_page == '예' or next_page == '네':
#         try:
#             next_btn = driver.find_element(By.CSS_SELECTOR, "#searchOptionForm > div.search-wrapper > div.search-content.search-content-with-feedback > div.search-pagination > a.btn-next")
#             next_btn.click()
#             time.sleep(1)
#             driver.switch_to.window(driver.window_handles[-1])
#             elem = driver.find_element(By.TAG_NAME, "body")
#             product_lists()
#         except:
#             print("마지막 페이지입니다.")
#             break
#     elif next_page == '아니오' or next_page == '아니요':
#         break
#     else:
#         print("예, 아니오로 대답해주세요.")


def numlist(i):
    num1 = ['일', '이', '삼', '사', '오', '육', '칠', '팔', '구', '십', '십일', '십이', '십삼', '십사', '십오', '십육', '십칠', '십팔', '십구', '이십', '이십일', '이십이', '이십삼', '이십사', '이십오', '이십육', '이십칠', '이십팔', '이십구', '삼십', '삼십일', '삼십이', '삼십삼', '삼십사', '삼십오', '삼십육']
    num2 = ['일번', '이번', '삼번', '사번', '오번', '육번', '칠번', '팔번', '구번', '십번', '십일번', '십이번', '십삼번', '십사번', '십오번', '십육번', '십칠번', '십팔번', '십구번', '이십번', '이십일번', '이십이번', '이십삼번', '이십사번', '이십오번', '이십육번', '이십칠번', '이십팔번', '이십구번', '삼십번', '삼십일번', '삼십이번', '삼십삼번', '삼십사번', '삼십오번', '삼십육번']
    num3 = ['1번', '2번', '3번', '4번', '5번', '6번', '7번', '8번', '9번', '10번', '11번', '12번', '13번', '14번', '15번', '16번', '17번', '18번', '19번', '20번', '21번', '22번', '23번', '24번', '25번', '26번', '27번', '28번', '29번', '30번', '31번', '32번', '33번', '34번', '35번', '36번']
    num4 = ['첫번째', '두번째', '세번째', '네번째', '다섯번째', '여섯번째', '일곱번째', '여덟번째', '아홉번째', '열번째', '열한번째', '열두번째', '열세번째', '열네번째', '열다섯번째', '열여섯번째', '열일곱번째', '열여덟번째', '열아홉번째', '스무번째', '스물한번째', '스물두번째', '스물세번째', '스물네번째', '스물다섯번째', '스물여섯번째', '스물일곱번째', '스물여덟번째', '스물아홉번째', '서른번째', '서른한번째', '서른두번째', '서른세번째', '서른네번째', '서른다섯번째', '서른여섯번째']
    if i in num1:
        i = num1.index(i)+1
    elif i in num2:
        i = num2.index(i)+1
    elif i in num3:
        i = num3.index(i)+1
    elif i in num4:
        i = num4.index(i)+1
    return i


products_list=driver.find_elements(By.CSS_SELECTOR, "#productList > li")

r2= sr.Recognizer()
m2 = sr.Microphone()
newlen = len(products_list) - 33

chooseOption = ''
# 맥은 time.sleep 지우고 한번에 처리
speak_origin(f"방금 말씀드린 것과 같이 현재 페이지에 {newlen}개의 상품이 있습니다. 하나의 상품에 대한 자세한 정보를 원하면 '한 개'라고 대답하고,두 상품 비교를 원하면 '두 개'라고 대답해주세요.")

#음성 인식 (STT)
def listen_choose(recognizer, audio):
    
    global chooseOption  # chooseOption을 함수 외부에서 선언

    while not chooseOption:
        try:
            text = recognizer.recognize_google(audio, language='ko')
            print('[사용자]' + text)
            with open('result.txt', 'a', encoding='utf-8') as f:
                f.write('[사용자]'+ text + '\n')
            chooseOption = text
        except sr.UnknownValueError:
            print('chooseOption 인식 대기')
        except sr.RequestError as e:
            print('요청 실패: {0}'.format(e))


stop_listening2 = r2.listen_in_background(m2, listen_choose)

time.sleep(7)
print('chooseOption은 이렇게 인식됐음 :',chooseOption)
stop_listening2(wait_for_stop=False) #더 이상 듣지 않음

if(chooseOption == '한 개' or chooseOption == '한개'):
    r3= sr.Recognizer()
    m3 = sr.Microphone()
    i = None
    speak_origin("원하는 상품의 번호를 숫자로 말씀해주세요.")

    #음성 인식 (STT)
    def listen_num(recognizer, audio):
        try:
            text = recognizer.recognize_google(audio, language='ko')
            print('[사용자]'+ text)
            with open('result.txt', 'a', encoding='utf-8') as f:
                f.write('[사용자]'+ text + '\n')
            global i
            i = text
        except sr.UnknownValueError:
            print('인식 대기')
        except sr.RequestError as e:
            print('요청 실패 : {0}'.format(e))

    stop_listening3 = r3.listen_in_background(m3, listen_num)
    time.sleep(5)

    try:
        i = int(i)
    except:
        i = numlist(i)

    products_list[i-1].click()

    driver.switch_to.window(driver.window_handles[-1])
    time.sleep(0.1)

    stop_listening3(wait_for_stop=False) #더 이상 듣지 않음

    onelist=[]
    onelist = crawling()
    driver.quit()

elif(chooseOption == '두 개' or chooseOption == '두개'):
    if os.path.exists('./images/'):
            for file in os.scandir('./images/'):
                os.remove(file.path)
            os.rmdir('./images/')
    print('두개라고 했을때 여기로 들어왔다!!!')
    r4= sr.Recognizer()
    m4 = sr.Microphone()
    firItem = None
    speak_origin("비교하고 싶은 두 상품 중 첫번째 상품의 번호를 말씀해주세요. ")

    #음성 인식 (STT)
    def listen_fir(recognizer, audio):
        
        try:
            text = recognizer.recognize_google(audio, language='ko')
            print('[사용자]' + text)
            with open('result.txt', 'a', encoding='utf-8') as f:
                f.write('[사용자]'+ text + '\n')
            global firItem
            firItem = text
            
        except sr.UnknownValueError:
            print('인식 대기')
        except sr.RequestError as e:
            print('요청 실패: {0}'.format(e))

    stop_listening4 = r4.listen_in_background(m4, listen_fir)
    while firItem is None:
        time.sleep(0.1)   # 값이 할당될 때까지 대기

    print('firItem은 이렇게 인식됐음 :',firItem)
    try:
        firItem = int(firItem)
    except:
        firItem = numlist(firItem)
    products_list[firItem-1].click()
    speak_origin("상품 정보를 불러오고 있습니다. 잠시만 기다려주세요.")

    driver.switch_to.window(driver.window_handles[-1])
    time.sleep(0.1)

    stop_listening4(wait_for_stop=False) #더 이상 듣지 않음

    list1 = []
    list1 = textCrawling()

    driver.close()
    driver.switch_to.window(driver.window_handles[0])

    r5= sr.Recognizer()
    m5 = sr.Microphone()
    secItem = None
    speak_origin("첫번째 상품의 정보를 불러왔습니다.")
    speak_origin("두번째 상품의 번호를 말씀해주세요.")

    #음성 인식 (STT)
    def listen_sec(recognizer, audio):
        while True:
            try:
                text = recognizer.recognize_google(audio, language='ko')
                print('[사용자]'+ text)
                with open('result.txt', 'a', encoding='utf-8') as f:
                    f.write('[사용자]'+ text + '\n')
                global secItem
                secItem = text
                break
            except sr.UnknownValueError:
                print('인식 대기')
            except sr.RequestError as e:
                print('요청 실패 : {0}'.format(e))

    
    stop_listening5 = r5.listen_in_background(m5, listen_sec)
    time.sleep(5)

    try:
        secItem = int(secItem)
    except:
        secItem = numlist(secItem)
    products_list[secItem-1].click()
    speak_origin("상품 정보를 불러오고 있습니다. 잠시만 기다려주세요.")

    driver.switch_to.window(driver.window_handles[-1])
    time.sleep(0.1)

    stop_listening5(wait_for_stop=False) #더 이상 듣지 않음

    list2 = []
    list2 = textCrawling()

    speak_origin("두번째 상품의 정보를 불러왔습니다.")
    driver.quit()



# 사진 자르기
from PIL import Image

def image_crop( infilename , save_path):
    img = Image.open( infilename )
    (img_h, img_w) = img.size

    # crop 할 사이즈 : grid_w, grid_h
    grid_w = 1960 # crop width
    grid_h = img_h # crop height
    range_w = (int)(img_w/grid_w)
    range_h = (int)(img_h/img_h)
 
    i = 0
 
    folderSize = get_files_count(save_path)
    if img_w > 1960:
        for w in range(range_w + 1):
            for h in range(range_h):
                bbox = (h*grid_h, w*grid_w, (h+1)*(grid_h), (w+1)*(grid_w))
                # 가로 세로 시작, 가로 세로 끝
                crop_img = img.crop(bbox)

                fname = "{}.png".format("{0:01d}".format(folderSize + i))
                crop_img.save('images/' + fname)
                i += 1

 
if os.path.exists('images'):
    length = get_files_count('images')
    for i in range(length):
        image_crop('./images/' + str(i) + '.png', './images/')


# OCR
 # 본인의 APIGW Invoke URL로 치환
URL = ""
    
 # 본인의 Secret Key로 치환
KEY = ""
    
ocr_result = []
if not os.path.exists('images'):
    print("사진이 없습니다.")
else:
    length = get_files_count('images')
    for i in range(length):
        with open('images/'+ str(i) +'.png', "rb") as f:
            img = base64.b64encode(f.read())
            image = Image.open('images/'+ str(i) +'.png')
            (image_h, image_w) = image.size
            if(image_h > 860 or image_w > 1960):
                continue
        headers = {
            "Content-Type": "application/json",
            "X-OCR-SECRET": KEY
        }
    
        data = {
            "version": "V1",
            "requestId": "sample_id", # 요청을 구분하기 위한 ID, 사용자가 정의
            "timestamp": 0, # 현재 시간값
            "lang": "ko",
            "resultType": "string",
            "images": [
                {
                    "name": "sample_image",
                    "format": "jpg",
                    "data": img.decode('utf-8')
                }
            ]
        }

        data = json.dumps(data)
        response = requests.post(URL, data=data, headers=headers)
        res = json.loads(response.text)

        ocr_text = []

        for i in range(len(res['images'][0]['fields'])):
            plz = res['images'][0]['fields'][i]['inferText']
            ocr_text.append(plz)

        result = ' '.join(s for s in ocr_text)
        ocr_result.append(result)
print(ocr_result)
if(ocr_result != []):
    speak_origin("상품 정보를 모두 불러왔습니다.")




# GPTchatBot
openai.api_key = ""


messages = []

try:
    init = "이제 내가 물어볼건데 한 줄로 대답해주길 바래. 이 때 ml은 미리리터 라고 알려주고 x는 곱하기라고 말해줘. 또 p는 피스라고 말해줘. 배송 예정일 같은 경우는 몇월 며칠이라고 읽어줘. 그리고 모든 대답은 존댓말로 해줘. 이제 물어볼게"
    content = str(onelist) + str(ocr_result)
    print('1번 선택')
except:
    init = "위와 같이 [첫번째 상품]과 [두번째 상품]을 비교하는 질문을 할건데 한 줄로 대답해줘. 이 때 ml은 미리리터로 x는 곱하기로,p는 피스라고 바꿔서 알려줘. 배송일 같은경우는 /앞에있는 숫자에에 '월',/뒤에있는 숫자에 '일'을 붙여서 말해줘. 그리고 모든 대답은 존댓말로 해줘. 이제 물어볼게"
    content = '[첫번째 상품]: ' + str(list1) + '\n[두번째 상품]: ' + str(list2)
    print('2번 선택')

def answer(recognizer,audio):
    
    text = recognizer.recognize_google(audio, language='ko')

    if('종료' in text):
        stop_listening(wait_for_stop=False)
        exit()

    else:    
        try:
            text = recognizer.recognize_google(audio, language='ko')
            
            print('[사용자]'+ text)
            with open('result.txt', 'a', encoding='utf-8') as f:
                f.write('[사용자]'+ text + '\n')

            user_content = content + "              " + init + text
            
            
            messages.append({"role": "user", "content": f"{ user_content}"})

            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

            assistant_content = completion.choices[0].message["content"].strip()

            messages.append({"role": "assistant", "content": f"{assistant_content}"})

            speak(assistant_content)

            messages.clear()

        except sr.UnknownValueError:
            print('인식 실패')
        except sr.RequestError as e:
            print('요청 실패 : {0}'.format(e))   

# 소리내 읽기 (TTS)
def speak(text):
    print('[Eyescape]'+ text)
    with open('result.txt', 'a', encoding='utf-8') as f:
        f.write('[Eyescape]'+ text + '\n')
    file_name = 'voice.mp3'
    tts = gTTS(text=text, lang='ko')
    tts.save(file_name)

    # 속도 조절
    sound = AudioSegment.from_file(file_name, format='mp3')
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={"frame_rate": int(sound.frame_rate * 1.15)})
    # sound_with_altered_frame_rate = sound_with_altered_frame_rate[:-50]
    sound_with_altered_frame_rate.export(file_name, format='mp3')

    # 재생
    playsound_with_wait(file_name)

    # 파일 삭제
    if os.path.exists(file_name): # voice.mp3 
        os.remove(file_name)   
        
    

#마이크로부터 음성듣기
r = sr.Recognizer()
m = sr.Microphone()

speak('무엇을 도와드릴까요?')


stop_listening = r.listen_in_background(m, answer)
# stop_listening(wait_for_stop=False)

while True:
    time.sleep(0.1) 