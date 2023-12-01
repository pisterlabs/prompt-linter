#Api Key
fileopen = open("D:\\Bhardwaj\\J.A.R.V.I.S\\Data\\Api.txt")
API = fileopen.read()
fileopen.close()

#Modules
import openai

#Coding
openai.api_key = API
completion = openai.Completion()


def QuestionAnswer(question, chat_log=None):
    file_path = "D:\\Bhardwaj\\J.A.R.V.I.S\\Database\\chat_log.txt"
    with open(file_path, "r") as file:
        chat_log_template = file.read()

    if chat_log is None:
        chat_log = chat_log_template

    prompt = f'{chat_log} You : {question}\nJ.A.R.V.I.S. : '
    response = completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.0,
        max_tokens = 100,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    answer = response.choices[0].text.strip()

    return answer
