import openai
import time
from get_news import get_top_news


def get_res(news, question):
    openai.api_key = "***"  # Your openai key
    content = "I will give you some news about an event. Please read the news I give and answer my question based on these news. News are these:{}. My question is {}. If the news I give do not contain these information, you just say 'Unknown'. Please do not output any other content or explanations, just answer the question is OK.".format(news, question)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "user",
                "content": content
             }
        ]
    )
    return completion.choices[0].message['content']


def run(keyword, question):
    news_string = ""
    news_list = get_top_news(keyword)
    if len(news_list) == 0:
        return "Sorry, but I didn't get such news."
    for news in news_list:
        news_string += news
        news_string += "|"

    if len(news_string) > 3000:
        news_string = news_string[:3000]

    retry_count = 0
    max_retries = 5

    while retry_count < max_retries:
        try:
            res = get_res(news_string, question)
            return res

        except Exception as e:
            print("An exception occurred: {}".format(str(e)))
            print("Retrying...")
            retry_count += 1
            time.sleep(10)

    if retry_count == max_retries:
        print("Reach the max retries... Seems something wrong.")
