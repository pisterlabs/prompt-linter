from dotenv import load_dotenv, find_dotenv
from icecream import ic

from common.aidevs_taskhandler import TaskHandler

from common.logger_setup import configure_logger
from tasks.c04l03.C04L03_gnome_openAI_part import give_me_answer_about_picture_based_on_question

# from openai import OpenAI

# Rozwiąż zadanie API o nazwie ‘gnome’. Backend będzie zwracał Ci linka do obrazków przedstawiających gnomy/skrzaty.
# Twoim zadaniem jest przygotowanie systemu, który będzie rozpoznawał, jakiego koloru czapkę ma wygenerowana postać.
# Uwaga! Adres URL zmienia się po każdym pobraniu zadania i nie wszystkie podawane obrazki zawierają zdjęcie postaci
# w czapce. Jeśli natkniesz się na coś, co nie jest skrzatem/gnomem, odpowiedz “error”.
# Do tego zadania musisz użyć GPT-4V (Vision).

# {
#     'code': 0,
#     'msg': 'I will give you a drawing of a gnome with a hat on his head. Tell me what is the color of the hat in '
#            'POLISH. If any errors occur, return \'ERROR\' as answer',
#     'hint': 'it won\'t always be a drawing of a gnome >:)',
#     'url': 'https://zadania.aidevs.pl/gnome/16fdfe293e71aa197c6229b6985a7012.png'
# }

load_dotenv(find_dotenv())
log = configure_logger("gnome")


def gnome_task():
    task_name = "gnome"
    try:
        handler = TaskHandler(task_name=task_name, logger=log)
        task_description = handler.get_task()
        log.info(f'task_description: {task_description.response_json}')

        url = handler.get_task().url

        question = "Jaki jest kolor czapki?"
        response = give_me_answer_about_picture_based_on_question(question, url)

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
    gnome_task()
