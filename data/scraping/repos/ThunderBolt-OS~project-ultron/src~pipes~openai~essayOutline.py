from dotenv import load_dotenv
import os
import openai
load_dotenv()


# Get the API key from the .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
completion = openai.Completion()

def EssayOutline(essayTopic, chat_log = None):
    FileLog = open("src\\database\\essayOutline.txt", "r")
    chat_log_template = FileLog.read()
    FileLog.close()

    if chat_log is None:
        chat_log = chat_log_template

    prompt = f"{chat_log}You: {essayTopic}\nUltron: "
        
    response = completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.3,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    answer = response["choices"][0].text.strip()
    chat_log_template_update = chat_log_template + f"You: {essayTopic} \nUltron: {answer}\n"
    FileLog = open("src\\database\\essayOutline.txt", "w")
    FileLog.write(chat_log_template_update)
    FileLog.close()

    return answer




