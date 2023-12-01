import os
import re

from common.aidevs_taskhandler import TaskHandler
from common.logger_setup import configure_logger
from tasks.c02l04.c02l04_whisper_openAI_part import openAI_transcribe, get_file_name, download


# 'please return transcription of this file: https://zadania.aidevs.pl/data/mateusz.mp3'
# 'hint': 'use WHISPER model - https://platform.openai.com/docs/guides/speech-to-text'

def extract_link(message):
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, message)
    return urls


def whisper_api_task():
    task_name = "whisper"
    log = configure_logger(task_name)
    try:
        handler = TaskHandler(task_name=task_name, logger=log)
        task_description = handler.get_task()
        log.info(f'task_description: {task_description.response_json}')

        url = extract_link(task_description.msg)[0]
        log.info(f"url: {url}")
        downloaded_file = get_file_name(url)

        if not os.path.exists(downloaded_file):
            download(url, downloaded_file)

        task_answer = openAI_transcribe(downloaded_file, prompt="As Polish would say:")
        log.info(f'Task_answer: {task_answer}')

        answer_response = handler.post_answer(task_answer)
        log.info(f'Answer Response: {answer_response.note}')

        assert answer_response.code == 0, "We have proper response code"
        assert answer_response.msg == 'OK', "We have proper response msg"
        assert answer_response.note == 'CORRECT', "We have proper response note"

    except Exception as e:
        log.exception(f"Exception {e}")


if __name__ == "__main__":
    whisper_api_task()
