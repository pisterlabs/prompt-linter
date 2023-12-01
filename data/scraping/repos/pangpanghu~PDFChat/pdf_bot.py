from flask import Flask, render_template, request, current_app, jsonify, session
from flask_session import Session  # 为了使用 Flask session，你需要先安装 Flask-Session
import openai
import logging
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
import time
import random
import os
import re

#从配置文件中取出openai的key
load_dotenv()
openai.api_key = os.getenv('OPENAI_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')

delimiter ="```"
pdf_path="./data/零信任.pdf"
db_path = "./vector_store"

#指定本地的VectorStore 
embeddings = OpenAIEmbeddings()
docsearch = Chroma(persist_directory=db_path, embedding_function=embeddings)

pdf_bot = Flask(__name__)
pdf_bot.config['SECRET_KEY'] = '9a1c2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a'  # 为了使用 session，需要设置一个密钥
pdf_bot.config['SESSION_TYPE'] = 'filesystem'  # 设置session 存储类型

# Initialize the Session
Session(pdf_bot)


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    for i in range(3):  # 加入重试机制最多试三次
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature, 
                max_tokens=max_tokens, 
            )
            return response.choices[0].message["content"]
        except openai.error.RateLimitError:
            sleep_time = (1.3 ** i) + random.random()
            time.sleep(sleep_time)

#聊天记录打印
def print_chat_history(messages):
    
    for item in messages:
        print(f"Row: {item}\n") 


@pdf_bot.route('/')
def home():
    return render_template('index.html')

@pdf_bot.route('/bot', methods=['POST'])
def bot_reply():
    question = request.json['message']
    
    system_prompt = """You are a bilingual QA bot. You are tasked with providing responses in Chinese, and your responses should be directly based on provided reference content. If the reference does not contain the information necessary to respond to a query, do not invent a response. Instead, reply with "Sorry, I can't find related information from the WiKi. Please contact the oncall team for this question."
The user will indicate their questions or comments by enclosing them in triple backticks.
Your reference material is as follows:

<context>

Example: ZIT is a zero trust architecture that is based on the principle of “never trust, always verify.”

</context>
"""
    
    # Initialize chat history if it does not exist
    if 'chat_history' not in session:
        session['chat_history'] = [{"role": "system", "content": system_prompt}]
    
    session['chat_history'].append({"role": "user", "content": question})
    
    #正常输入的话，则开始组合Prompt
    docs = docsearch.similarity_search(query=question,k=3)
    context = f"{docs[0].page_content}\n\n{docs[1].page_content}\n\n{docs[2].page_content}"
    # 提取系统消息
    system_message = session['chat_history'][0]["content"]
    pattern = r'<context>(.*?)</context>'
    match = re.search(pattern, system_message, re.DOTALL)  # 加入re.DOTALL标志
    if match:
        old_content = match.group(1)  # 提取出的内容
    # 将旧内容替换为新内容
    new_content = f"\n{context}\n"
    new_system_message = system_message.replace(old_content, new_content)
    session['chat_history'][0]["content"] = new_system_message
    session['chat_history'].append({"role": "user", "content": f"{delimiter}{question}{delimiter}"})
    
    
    answer = get_completion_from_messages(session['chat_history'])
    
    session['chat_history'].append({"role": "assistant", "content": answer})
    #for s in session['chat_history']:
    #    print(s['role'] + ": " + s['content'])                    
    return jsonify({'reply': answer})

if __name__ == '__main__':
        
    pdf_bot.run(host='0.0.0.0',port=8001)
