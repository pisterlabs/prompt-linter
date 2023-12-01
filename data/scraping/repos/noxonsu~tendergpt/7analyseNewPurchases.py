from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import requests
import json

chat = ChatOpenAI(temperature=0)

#load prompt.txt
with open("prompt.txt", "r") as f:
    prompt = f.read()
    f.close()

#load purchases/1056946888.json as json
with open("purchases/1057124558.json", "r") as f:
    purchase = f.read()
    f.close()
purchase = json.loads(purchase)
purchasetext = json.dumps(purchase, ensure_ascii=False)

if (len(purchasetext) > 30000): 
    print(" Too long ")
    #remove lotsList field from purchase
    purchase.pop('lotsList', None)
    purchasetext = json.dumps(purchase, ensure_ascii=False)
    print(len(purchasetext))


messages = [
    SystemMessage(
        content=prompt
    ),
    HumanMessage(
        content=purchasetext
    )
]
gpttitle = chat(messages)
print(gpttitle.content)

if gpttitle.content is not None and len(gpttitle.content) > 20: # and "Не подходит" not in gpttitle.content
    #prepare to pass in url (escape & and ?)
    gpttitle.content = gpttitle.content.replace("&", "%26")
    webhook_url = f"https://noxon.wpmix.net/counter.php?totenders=1&msg={gpttitle.content}"
                            
    response = requests.post(webhook_url)
    if response.status_code != 200:
        print(f"Failed to send webhook for tender at index {idx}. Status code: {response.status_code}")
