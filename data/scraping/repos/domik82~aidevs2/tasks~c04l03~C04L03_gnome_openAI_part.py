from dotenv import load_dotenv, find_dotenv
from icecream import ic
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage

from common.logger_setup import configure_logger

load_dotenv(find_dotenv())
log = configure_logger()

# openai.api_key = os.getenv("OPENAI_API_KEY")

system_prompt = """ 
    'Twoim zadaniem jest powiedzenie w jakim kolorze jest czapka skrzata (inaczej gnoma) na obrazku. \n'
    'Zasady: \n'
    ' - odpowiadasz jednym słowem którym jest kolor czapki gnoma \n'
    ' - jeśli na obrazku jest co innego niż gnom odpowiadasz "error"\n'
"""


def give_me_answer_about_picture_based_on_question(usr_question=None, url=None, ):
    log.info(f"usr_question:{usr_question}")
    try:

        chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=256)
        ai_response = chat.invoke(
            [SystemMessage(content=system_prompt),
             HumanMessage(
                 content=[
                     {"type": "text", "text": usr_question},
                     {
                         "type": "image_url",
                         "image_url": {
                             "url": url,
                             "detail": "auto",
                         },
                     },
                 ]
             )
             ]
        )
        log.info(f"content: {ai_response}")
        response_content = ai_response.content
        log.info(f"response_content: {response_content}")

        return response_content
    except Exception as e:
        log.error(f"Exception: {e}")


if __name__ == "__main__":
    load_dotenv(find_dotenv())

    red_url = 'https://zadania.aidevs.pl/gnome/16fdfe293e71aa197c6229b6985a7012.png'
    green_url = 'https://zadania.aidevs.pl/gnome/414c7e2fcd1e1c9c923e939084819e36.png'
    troll_url = 'https://zadania.aidevs.pl/gnome/08d8c505c496f224458ce6032760f440.png'

    try:
        question = "Jaki jest kolor czapki?"
        answer = give_me_answer_about_picture_based_on_question(question, red_url)


    except Exception as e:
        log.exception(f'Exception: {e}')
