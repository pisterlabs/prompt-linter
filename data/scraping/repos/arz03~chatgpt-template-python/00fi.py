import os
import openai

openai.api_key = os.environ['keyy']

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

continuee=True

def readchat():
  pr= open("chats/chat.txt", "r")
  l=pr.readlines()
  str1=""
  for ele in l:
    str1 += ele
  pr.close()
  return str1

print("chat starts here, write \"exit\" to end conversation")
while continuee:
  inputt=input("you:")
  if inputt=="exit":
    continuee=False
  else:
    chatlog=readchat()
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=(chatlog+"\nHuman:"+inputt+"\nAI:"),
    temperature=0.9,
    max_tokens=150,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"])
    print("bot:", response['choices'][0]['text'])
    pr= open("chats/chat.txt", "a+")
    pr.writelines("\nHuman:"+inputt+"\nAI:"+response['choices'][0]['text'])
    pr.close()
    continuee=True
else:
  print("end")

