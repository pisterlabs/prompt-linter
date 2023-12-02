from bs4 import BeautifulSoup
import requests
import json
# import logging
import openai
import os
import boto3
import base64

# from dotenv import load_dotenv
import time
from custom_error import HolidayError, NetworkError
from menu_example import student_lunch_1


ENCRYPTED = os.environ['GPT_API_KEY']
# Decrypt code should run once and variables stored outside of the function
# handler so that these are decrypted once per container
DECRYPTED = boto3.client('kms').decrypt(
    CiphertextBlob=base64.b64decode(ENCRYPTED),
    EncryptionContext={'LambdaFunctionName': os.environ['AWS_LAMBDA_FUNCTION_NAME']}
)['Plaintext'].decode('utf-8')


# OpenAI API 인증
# openai.api_key = os.environ.get("GPT_API_KEY")
openai.api_key = DECRYPTED


# # 로거 인스턴스 생성
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# # 파일 핸들러 생성 및 설정
# file_handler = logging.FileHandler('../data/my_log.log')
# file_handler.setLevel(logging.DEBUG)

# # 콘솔 핸들러 생성 및 설정 (표준 출력에 로그를 출력)
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)

# # 로그 포맷 설정
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)
# console_handler.setFormatter(formatter)

# # 핸들러를 로거에 추가
# logger.addHandler(file_handler)
# logger.addHandler(console_handler)

def practice_student_restarant(date: str):

    res = requests.get(
        f"http://m.soongguri.com/m_req/m_menu.php?rcd=1&sdt={date}")

    soup = BeautifulSoup(res.content, "html.parser")

    if soup.find(text="오늘은 쉽니다."):
        # logger.error(f"The date is holiday. {date}")
        raise HolidayError(date)
    elif "휴무" in res.text:
        # logger.error(f"The date is holiday. {date}")
        raise HolidayError(date)


    tr_list = soup.find_all('tr')
    menu_nm_dict = dict()

    for tr_tag in tr_list:  # tr_tag는 tr과 그 하위 태그인 Beautifulsoup 객체
        td_tag = tr_tag.find('td', {'class': 'menu_nm'})
        if td_tag:
            menu_nm_dict[td_tag.text] = tr_tag

    menu_nm_dict = strip_string_from_html(menu_nm_dict)

    res_from_ai = chat_with_gpt_student_restaurant(menu_nm_dict)

    # refine_res = {
    #     "restaurant": "학생식당",
    #     "date": date,
    #     "menu": res_from_ai
    # }

    # logger.debug(f"{__name__} final result is {refine_res}")

    return res_from_ai


def strip_string_from_html(menu_dict):

    for key, value in menu_dict.items():
        new_text = ""

        for text in value.stripped_strings:
            new_text += f"{text}/n"
        menu_dict[key] = new_text

    return menu_dict


def chat_with_gpt_student_restaurant(today_mnu_dict) -> dict:
    setup_messages = [
        {"role": "system", "content": "너는 메인메뉴와 사이드메뉴를 구분하는 함수의 역할을 맡았다. 너는 메인메뉴가 담긴 list만을 반환하는 함수의 역할을 맡았다. input값에서 메인 메뉴와 사이드 메뉴를 구분해야해. list에는 input값에서의 메인 메뉴만 골라낸 요소들의 이름이 들어가. 만약 동일한 메인메뉴가 있다면 한 개만 리스트에 넣어. 그외에 부가적인 설명은 하지 않고 오직 json을 반환해."},
        {"role": "user", "content": f"input은 바로 이거야. 여기서 메인 메뉴를 골라내어 배열을 만들고 반환해줘.:{student_lunch_1}"},
        {"role": "assistant", "content": '["뚝배기부대찌개*당면사리"]'},
    ]

    max_retries = 3
    delay_seconds = 5

    for key, value in today_mnu_dict.items():
        retries = 0
        while retries < max_retries:
            try:
                setup_messages.append(
                    {"role": "user", "content": f"input은 바로 이거야. 여기서 메인 메뉴를 골라내어 list을 반환해줘.:{value}"})

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=setup_messages
                )

                today_mnu_dict[key] = json.loads(
                    response['choices'][0]['message']['content'])

                setup_messages.pop()
                break  # 성공한 경우 반복문 종료
            except openai.error.RateLimitError:  # API 요청 제한에 도달한 경우
                # logging.error(f"API Request Cap Reached.")
                # 재시도 대기 시간만큼 대기
                time.sleep(delay_seconds)
                retries += 1
            except KeyError:  # chatgpt response가 안와서 dict에 key값이 없음
                time.sleep(delay_seconds)
                retries += 1
            except json.decoder.JSONDecodeError:
                time.sleep(delay_seconds)
                retries += 1

        if retries >= max_retries:
            # logging.error(f"All retrials are failed.")
            raise NetworkError

    return today_mnu_dict
