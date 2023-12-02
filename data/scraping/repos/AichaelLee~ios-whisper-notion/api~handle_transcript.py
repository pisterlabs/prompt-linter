import openai
from api import txt2notion
import logging
import os
from dotenv import load_dotenv
# Load the .env file
load_dotenv()

api_key = os.environ.get('OPENAI_KEY')

# init OpenAI API
openai.api_key = api_key

# Get custom_base_url from environment variables
custom_base_url = os.getenv('OPENAI_PROXY')

# Set openai.api_base if custom_base_url is not empty
if custom_base_url:
    openai.api_base = custom_base_url

text_gpt_prompt = os.getenv('TEXT_GPT_PROMPT')
if text_gpt_prompt is None:
    text_gpt_prompt = '你是一个写作大师，基于以下逻辑和观点，拓展出一篇文章，详略有序，有故事有论证'


def handle_transcript(file_path):
    # 从音频文件中获取文本
    transcript = transcribe_audio(file_path)
    # 使用 GPT 处理文本
    # handle_response(transcript)


# 转录音频
def transcribe_audio(file_path):
    response = openai.Audio.transcribe("whisper-1", file_path)
    transcript = response['text']
    logging.info('文稿转录成功‘')
    txt2notion.send_to_notion("转录文稿", transcript)
    return transcript


# 获取GPT处理后信息
def handle_response(message):
    response_text = chatgpt_get_response(message)
    logging.info('gpt 处理成功')
    txt2notion.send_to_notion("处理文稿", response_text)


# 使用ChatGPT获取结果
def chatgpt_get_response(message):
    logging.info('begin to get response by message')
    prompt = f"""
       {text_gpt_prompt}
       ```
       {message}
       ```
        """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "user",
            "content": prompt
        }],
        max_tokens=2000,
        temperature=0.5,
    )

    # 提取生成的文本
    response_json = response['choices'][0]['message']['content']
    # print('the response json is', response_json)
    return response_json