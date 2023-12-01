from crypt import methods
import openai
import myopenaiapikey
import json
from flask import Flask, request

openai.api_base = myopenaiapikey.base
openai.api_key = myopenaiapikey.api

app = Flask(__name__)

from SparkAPI import get_Spark_response

prompt_template = "你要成为一台Linux终端，我来输入命令，你来显示运行结果.第一条指令是："

def ai_run_command(query):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "assistant", 
             "content": """
                        I want you to act as a Linux terminal. I will provide commands and history, then you will reply with what the terminal should show. I want you to only reply with the terminal output inside one unique code block, and nothing else. Do no write explanations. Do not type commands unless I instruct you to do so.\n\n### Command:\n{command}\n\n### History:\n{history}\n### Response:\n
                        """},
            {"role": "user", "content": query}],
    )
    message = response["choices"][0]["message"]["content"]
    return message



@app.route('/admin',methods=['GET','POST'])
def hello_admin():
    cmd = request.json['cmd']
    try:
        message = ai_run_command(cmd)
    except:
        message = "Permission denied"
    return {"message":message}

if __name__ == '__main__':

    app.run('0.0.0.0',9000,debug=True)
