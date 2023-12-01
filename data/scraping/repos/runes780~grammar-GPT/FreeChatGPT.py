import openai
import datetime
openai.api_key = ""
# 输入问题
context= ''
for i in range(0, 10):
    question = input("请输入问题：")
    
    chat = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": ""},
        {"role": "user", "content": question},
        {"role": "assistant", "content": context}

      ]
    ) 
    question = 'user: ' + question 
    output_message = chat.choices[0].message
    context = context + question + "\n" +("ChatGPT:" + output_message.content) 
    print(chat.usage)
    print('ChatGPT:'+ output_message.content)
    with open("dialogue.txt", "a", encoding="utf-8") as f:
      now = datetime.datetime.now()
      date = now.strftime("%Y-%m-%d")
      time = now.strftime("%H:%M:%S")
      f.write('\n' + '\n' + f"当前的日期是{date}，当前的时间是{time}\n"+context)

    