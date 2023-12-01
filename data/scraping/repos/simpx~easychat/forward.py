from easychat import Bot, Session
import openai
import yaml
import json
import logging

bot = Bot("forward")

prompt = "你是一个专业、精准、简洁的助手"

with open('../../config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    openai.api_key = config['api_key']

def ask_gpt(txt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "你是一个专业、精准、简洁的助手"},
                  {"role": "user", "content": f"{txt}\n"}],
        max_tokens=1024,
        temperature=0.7,
    )
    return response['choices'][0]['message']['content']

def ask_gpt_with_histroy(history):
    messages = [{"role": item["role"], "content": item["content"]} for item in history]
    ppp = [{"role": "system", "content": f"{prompt}"}] + messages
    logging.info(f"messages: {messages}")
    logging.info(f"prompt: {ppp}")

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=ppp,
        max_tokens=1024,
        temperature=0.7,
    )
    return response['choices'][0]['message']['content']


@bot.on_chat(["wkhGrzVQAAZ3gXdLV_HJM1V00Y_4QjiA"])
def handle_chat(request, session: Session):
    r_str = json.dumps(request, indent=4)
    logging.info(f"request in : {r_str}")
    session.send_message("...", True)
    try:
        result = ask_gpt_with_histroy(session.messages)
    except Exception as e:
        result = "出错了，请重试"
        logging.exception(f"Failed in ask_gpt")
    session.send_message(result)
    session.send_menu([
         {
             "type": "click",
             "click": {
                 "id": "101",
                 "content": "结束并开启新一轮交谈"
             }
         },
    ]) 

@bot.on_command(["wkhGrzVQAAZ3gXdLV_HJM1V00Y_4QjiA"])
def handle_command(request, session: Session):
    r_str = json.dumps(request, indent=4)
    logging.info(f"command request in : {r_str}")
    session.send_message("...", True)
    if request['command'] != '101':
        session.send_message("wrong command %s!" % request['command'], True)
        return
    session.send_message("再见")
    session.destroy()
