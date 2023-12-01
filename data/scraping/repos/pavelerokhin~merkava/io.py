from langchain.schema import AIMessage, HumanMessage

from src.models import openai_chat
from src.utils import *


def exec_commands(file_text, log):
    i = 0
    while True:
        try:
            # where starts the first prompt
            prompt_start = file_text.find("#mrkv")
            # where ends the first prompt
            prompt_end = file_text.find("#end")

            if prompt_start == -1 or prompt_end == -1:
                break
            i += 1
            log.info(f"{GREY}parsing and executing command {i}{WHITE}")

            # extract prompt
            file_head = file_text[:prompt_start]
            file_tail = file_text[prompt_end + 4:]
            prompt = file_text[prompt_start + 5:prompt_end]

            type_ = "text" if prompt[:5] != " code" else "code"
            if type_ == "code":
                prompt += "\n Write only Python code and comments as response. Code:"

            response = openai_chat([AIMessage(content=file_head), HumanMessage(content=prompt)])
            log.info("OPENAI response: " + response.content)

            file_text = file_head
            file_text += '\n"""\n' if type_ == 'code' else ''
            file_text += response.content
            file_text += '\n"""\n' if type_ == 'code' else ''
            file_text += file_tail
        except Exception as e:
            log.error(e)
            break

    return file_text


def exec_commands_to_file(file_path, log):
    f = open(file_path, 'r+')
    file_text = f.read()
    f.close()

    new_file_text = exec_commands(file_text, log)

    f = open(file_path, 'w+')
    f.seek(0)
    f.truncate()
    f.write(new_file_text)
    f.close()

