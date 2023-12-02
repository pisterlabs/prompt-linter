import os
import openai


def get_api_key():
    with open("J:\\Xite development\\MedGPT\\medgpt\\core\\ai\\Data\\api.txt", "r") as fileopen:
        api_key = fileopen.read().strip()
    return api_key


openai.api_key = get_api_key()
completion = openai.Completion()


def ReplyBrain(user_name, question, chat_log=None):
    chat_log_path = f"J:\\Xite development\\MedGPT\\medgpt\\media\\chat\\histroy\\{user_name}.txt"
    chat_log_template_path = "J:\\Xite development\\MedGPT\\medgpt\\core\\ai\\Database\\template.txt"

    # Check if the file exists, and if not, create it
    if not os.path.exists(chat_log_path):
        with open(chat_log_path, "w") as filelog:
            filelog.write("")

    with open(chat_log_path, "r") as filelog:
        chat_log_template = filelog.read()

    with open(chat_log_template_path, "r") as filelog:
        chat_log_template2 = filelog.read()

    if chat_log is None:
        chat_log = chat_log_template2

    prompt = f'{chat_log}You: {question}\nDocuAssist: '
    response = completion.create(model="text-davinci-002",
                                 prompt=prompt,
                                 temperature=0.5,
                                 max_tokens=60,
                                 top_p=0.3,
                                 frequency_penalty=0.5,
                                 presence_penalty=0,
                                 )
    answer = response.choices[0].text.strip()
    chat_log_template_update = chat_log_template + f"\nYou: {question}\nDocuAssist: {answer}"

    with open(chat_log_path, "w") as filelog:
        filelog.write(chat_log_template_update)

    return answer
