import os
import openai
import requests
import json
def chat_complection(order,name,address,email):    
    openai.api_key = "opanai_api_key"
    TARGET_URL = "https://api.openai.com/v1/chat/completions"
    headers = {'Authorization': 'Bearer ' + openai.api_key}
    

    data = {
        "model": "gpt-3.5-turbo",
        "temperature": 0.5,
        
        "messages":[{"role":"assistant",
                    "content":f"依以下資訊寫一封退貨信。訂單編號:{order},客戶姓名:{name},地址:{address},e-mail:{email}"
                    }]
        }

    result = requests.post( TARGET_URL,headers=headers,json=data)
    response_data = result.json()
    print(response_data)
    for res in response_data['choices']:
        print(res['message']['content'])

        with open("output.json", "w", encoding="utf-8") as fp:
            json.dump(res['message']['content'], fp, ensure_ascii=False, indent=4)


########################################################################
print(print("您好，歡迎使用客服機器人，請輸入您的退貨資訊... (請按 Enter 繼續...："))
order=input("訂單編號:")
name=input("名字:")
address=input("地址:")
email=input("email:")

chat_complection(order,name,address,email)

