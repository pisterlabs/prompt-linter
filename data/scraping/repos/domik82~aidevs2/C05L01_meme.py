import os
import time
from datetime import datetime

import openai
from dotenv import load_dotenv, find_dotenv
from icecream import ic
from requests import post

from common.aidevs_taskhandler import TaskHandler
from common.constants import RENDERFORM_IO_TOKEN
from common.logger_setup import configure_logger

load_dotenv(find_dotenv())
log = configure_logger()

# Wykonaj zadanie API o nazwie “meme”. Celem zadania jest nauczenie Cię pracy z generatorami grafik i dokumentów.
# Zadanie polega na wygenerowaniu mema z podanego obrazka i podanego tekstu. Mem ma być obrazkiem JPG o wymiarach
# 1080x1080. Powinien posiadać czarne tło, dostarczoną grafikę na środku i podpis zawierający dostarczony tekst.
# Grafikę z memem możesz wygenerować za pomocą darmowych tokenów dostępnych w usłudze RenderForm (50 pierwszych
# grafik jest darmowych). URL do wygenerowanej grafiki spełniającej wymagania wyślij do endpointa /answer/. W razie
# jakichkolwiek problemów możesz sprawdzić hinty https://zadania.aidevs.pl/hint/meme

# {
#     'code': 0,
#     'msg': 'Create meme using RednerForm API and send me the URL to JPG via /answer/ endpoint',
#     'service': 'https://renderform.io/',
#     'image': 'https://zadania.aidevs.pl/data/monkey.png',
#     'text': 'Gdy koledzy z pracy mówią, że ta cała automatyzacja to tylko chwilowa moda, a Ty właśnie zastąpiłeś ich '
#             'jednym, prostym skryptem',
#     'hint': 'https://zadania.aidevs.pl/hint/meme'
# }

RENDERFORM_IO_API_URL = 'https://get.renderform.io/api/v2/render'
TEMPLATE_ID = 'puffy-dogs-dive-loosely-1799'

HEADERS = {
    'Content-Type': 'application/json',
    'X-API-KEY': RENDERFORM_IO_TOKEN
}


def meme_task():
    task_name = "meme"
    try:
        handler = TaskHandler(task_name=task_name, logger=log)
        task_description = handler.get_task()
        log.info(f'task_description: {task_description.response_json}')
        meme_img = handler.get_task().image
        meme_txt = handler.get_task().text
        payload = {
            "template": TEMPLATE_ID,
            "data": {
                "image.src": meme_img,
                "title.text": meme_txt
            }
        }
        url = post(RENDERFORM_IO_API_URL, headers=HEADERS, json=payload).json()['href']
        ic(url)

        response = url

        ic(response)
        task_answer = response

        answer_response = handler.post_answer(task_answer)
        log.info(f'Answer Response: {answer_response.note}')

        assert answer_response.code == 0, "We have proper response code"
        assert answer_response.msg == 'OK', "We have proper response msg"
        assert answer_response.note == 'CORRECT', "We have proper response note"

    except Exception as e:
        log.exception(f"Exception {e}")


if __name__ == "__main__":
    meme_task()
