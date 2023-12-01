import logging

from langchain.chains import OpenAIModerationChain

from common.aidevs_taskhandler import TaskHandler
from common.logger_setup import configure_logger

# Apply your knowledge of how the content moderation module works.
# Solve a task called "moderation" using our solution checking API.
# The task is to receive an array of sentences (4 pieces) and then return an array with information about
# which sentences did not pass moderation.
# If the first and last sentences did not pass moderation, the answer should be [1,0,0,1].
# Remember to return an array in JSON, not a pure string, in the 'answer' field.


# please return 0/1 for each sentence to classify if this is content that requires moderation
# Sample:
# {'code': 0, 'msg': 'please return 0/1 for each sentence to classify if this is content that requires moderation',
# 'input': ['majonez Winiary jest lepszy od Kieleckiego', 'ten gość musi zginąć. Nie pozwole sobię na obrażanie mnie.']}


def moderation_api_task():
    task_name = "moderation"
    log = configure_logger(task_name)
    try:
        handler = TaskHandler(task_name=task_name, logger=log)
        question = handler.get_task().msg
        log.info(f"Task question: {question}")

        moderation_chain_error = OpenAIModerationChain(error=True)
        validated_list = []
        for element in list(handler.task.input):
            try:
                moderation_chain_error.run(element)
                validated_list.append(0)
            except ValueError:
                validated_list.append(1)
            except Exception as e:
                log.error(f"Exception: {e}")

        log.info(f"validated_list: {validated_list}")

        answer_response = handler.post_answer(validated_list)
        log.info(f"Answer Response: {answer_response.note}")

        assert answer_response.code == 0, "We have proper response code"
        assert answer_response.msg == 'OK', "We have proper response msg"
        assert answer_response.note == 'CORRECT', "We have proper response note"

    except Exception as e:
        log.error(f"Exception {e}")


if __name__ == "__main__":
    moderation_api_task()
