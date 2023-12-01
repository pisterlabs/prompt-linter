
import os
import requests
import json
import dotenv
import openai
import datetime

dotenv.load_dotenv()

WIKI_TOKEN = os.getenv('Wiki_token')
WIKI_MAIL = os.getenv('My_mail')
NASA_API_KEY = os.getenv('NASA_API_KEY')
openai.api_key = os.getenv('OPEN_AI_API_KEY')



# Line 發送到測試群組(測試用狗勾)
CHANNEL_ACCESS_TOKEN = os.getenv('Test_Access_Token')


    

def execute_line_test():
    data = {
        "messages": [
            {
                "type": "text",
                "text": "測試"
            }
        ]
    }


    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}"
    }

    response = requests.post("https://api.line.me/v2/bot/message/broadcast", data=json.dumps(data), headers=headers)

    if response.status_code != 200:
        print("Error broadcasting message: ", response.status_code, response.text)
    else:
        print("Message broadcasted successfully!")


# Wiki API section
# def test_wiki():
#     date = datetime.datetime.now()
#     date = date.strftime('%m/%d')
#     print("!!! Active !!! date: ", date)
#     text = ""
#     url = f'https://api.wikimedia.org/feed/v1/wikipedia/zh/onthisday/selected/{date}'
#     wiki_headers = {
#         'Authorization': f'Bearer {WIKI_TOKEN}',
#         'User-Agent': WIKI_MAIL
#     }
#     wiki_response = requests.get(url, headers=wiki_headers)
#     response = wiki_response.json().get('selected')
#     data_set = {"messages": []}
#     for i in response:
#         year = i['pages'][0]['title']
#         message = i['text']
#         more_info = i['pages'][1]['content_urls']['mobile']['page']
#         text = f"{year}{message}\n\n看更多:{more_info}"
#         data_set["messages"].append({
#             "type": "text",
#             "text": text
#         })
#     print(data_set)
# test_wiki()

if __name__ == '__main__':
    execute_line_test()