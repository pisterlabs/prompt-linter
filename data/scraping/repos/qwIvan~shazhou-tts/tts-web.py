import os
import requests
import tempfile
import xmltodict
import openai
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
from pywebio import config
from pywebio.platform.tornado_http import start_server
from pywebio.input import input, textarea, input_group, radio, actions
from pywebio.output import put_text, put_file, put_processbar, set_processbar, put_markdown, put_collapse, put_success

headers = {
    'content-type': 'application/ssml+xml; charset=utf-8',
    'ocp-apim-subscription-key': '09dd983c815c4ef7aca1b01146b41b37',
    'x-microsoft-outputformat': 'audio-16khz-128kbitrate-mono-mp3',
}

config(title='沙洲之歌 - 沙洲书社的论文音频合成工具')


def split_text(content, size=1500):
    chunks = []
    start = 0
    while start < len(content):
        end = start + size if start + size < len(content) else len(content)
        last_period = content.rfind('。', start, end)
        if last_period != -1:
            end = last_period + 1
        chunks.append(content[start:end])
        start = end
    return chunks


def generate_audio(content, voice_name):
    data_dict = {
        'speak': {
            '@version': '1.0',
            '@xml:lang': 'zh-CN',
            'voice': {
                '@name': voice_name,
                '#text': content
            }
        }
    }
    data = xmltodict.unparse(data_dict).encode()

    response = requests.post(
        'https://southeastasia.tts.speech.microsoft.com/cognitiveservices/v1', headers=headers, data=data,
    )

    temp_file_path = tempfile.mktemp(suffix=".mp3")
    with open(temp_file_path, 'wb') as temp_file:
        temp_file.write(response.content)

    return AudioSegment.from_mp3(temp_file_path)


def generate_article_audio(contents):
    combined = AudioSegment.empty()

    for i, content in enumerate(contents):
        data_dict = {
            'speak': {
                '@version': '1.0',
                '@xml:lang': 'zh-CN',
                'voice': {
                    '@name': 'zh-CN-YunzeNeural',
                    '#text': format(content)
                }
            }
        }
        data = xmltodict.unparse(data_dict).encode()

        response = requests.post(
            'https://southeastasia.tts.speech.microsoft.com/cognitiveservices/v1', headers=headers, data=data,
        )

        temp_file_path = tempfile.mktemp(suffix=".mp3")
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(response.content)

        audio_segment = AudioSegment.from_mp3(temp_file_path)
        combined += audio_segment

        # set_processbar(process_id, (i + 1) / len(contents))  # 更新进度条

    return combined


def main():
    # set_env(auto_scroll_bottom=True)

    put_markdown('# 沙洲之歌')
    put_markdown('> 沙洲书社的论文音频合成工具')
    voice_options = [('云泽', 'zh-CN-YunzeNeural'), ('晓秋', 'zh-CN-XiaoqiuNeural')]
    data = input_group("请输入论文信息", [
        input("论文的标题：", name='title', required=True),
        input("作者的名称：", name='author', required=True),
        input("作者的简介：", name='author_intro', required=True),
        input("论文的来源：", name='source', required=True),
        #radio("开头的语音：", options=voice_options, name='head-voice', inline=True, value='zh-CN-XiaoqiuNeural'),
        #radio("正文的语音：", options=voice_options, name='body-voice', inline=True, value='zh-CN-YunzeNeural'),
        #radio("结尾的语音：", options=voice_options, name='foot-voice', inline=True, value='zh-CN-XiaoqiuNeural'),
        textarea("论文的内容：", name='article', rows=10, placeholder="请将论文内容粘贴到这里", required=True),
        textarea("论文的结尾：", name='outro_text', rows=3, value='以上内容，由沙洲书社淡定洲同志制作，仅供学术研究使用。'),
        radio("格式化：", options=[('自动格式化', True), ('不格式化', False)], inline=True, name='formatting', value=True),
    ])

    title = data['title']
    author = data['author']
    author_intro = data['author_intro']
    source = data['source']
    formatting = data['formatting']
    #head_voice = data['head-voice']
    #body_voice = data['body-voice']
    #foot_voice = data['foot-voice']
    article = data['article']
    article = article.strip()

    head_voice = 'zh-CN-XiaoqiuNeural'
    body_voice = 'zh-CN-YunzeNeural'
    foot_voice = 'zh-CN-XiaoqiuNeural'

    ## 再次让用户编辑
    # article = textarea(value=article, rows=20)

    intro_text = f"标题：{title}\n作者：{author}，{author_intro}\n来源：{source}"
    outro_text = data['outro_text'].strip()

    article_contents = split_text(article)

    put_text("正在进行语音合成...")
    process_id = 'generate_audio'
    put_processbar(process_id)  # 初始化进度条

    # article_audio = generate_article_audio(article_contents, process_id)
    while True:
        try:
            combined_text, article_audio = process_contents_parallel(article_contents, body_voice, process_id, formatting)
        except:
            import traceback
            traceback.print_exc()
            # 询问用户是否想要重试
            retry = actions(label="发生错误，是否重试？", buttons=[{'label': '重试', 'value': 'retry'}, {'label': '取消', 'value': 'cancel'}])
            if retry == 'cancel':
                return
        else:
            break
    put_text("正在生成音频文件...")
    intro_audio = generate_audio(intro_text, head_voice)

    combined_audio = intro_audio + article_audio
    if outro_text:
        combined_audio += generate_audio(outro_text, foot_voice)

    temp_combined_audio_path = tempfile.mktemp(suffix=".mp3")
    combined_audio.export(temp_combined_audio_path, format='mp3')

    with open(temp_combined_audio_path, "rb") as f:
        content = f.read()

    put_success("音频文件已生成")
    put_file(f'{title}_{author}.mp3', content)  # 提供下载链接
    if formatting:
        put_collapse('整理好的论文', combined_text)


def format(raw_text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {
                "role": "system",
                "content": "我给你一段文字，你帮我去掉文中的脚注编号，其他的保留原文不变，不改变原文的文字"
            },
            {
                "role": "user",
                "content": "权力总是存在\n于权力主体和权力客体的相互作用之中。“理解 ‘权力’概念的最好的方法是将其视为冲突的意志\n之间的关系”。⑤\n"
            },
            {
                "role": "assistant",
                "content": "权力总是存在于权力主体和权力客体的相互作用之中。“理解 ‘权力’概念的最好的方法是将其视为冲突的意志之间的关系”。"
            },
            {
                "role": "user",
                "content": raw_text,
            }
        ],
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['message']['content']


pool_size = 10  # 线程池的大小，你可以根据你的机器情况调整这个值
pool = ThreadPoolExecutor(pool_size)  # 创建线程池


def process_content(raw_text, voice_name, formatting=True):
    formatted_text = format(raw_text) if formatting else raw_text
    audio_segment = generate_audio(formatted_text, voice_name)

    return formatted_text, audio_segment


def process_contents_parallel(contents, voice_name, process_id, formatting=True):
    futures = [pool.submit(process_content, content, voice_name, formatting) for content in contents]
    audio_results = [None] * len(contents)  # 创建一个空列表来存储音频结果
    text_results = [''] * len(contents)  # 创建一个空列表来存储文本结果

    for future in as_completed(futures):  # 在每个 future 完成时更新进度条
        done_count = len([f for f in futures if f.done()])
        set_processbar(process_id, done_count / len(futures))

    for i, future in enumerate(futures):  # 按提交顺序获取任务的结果并存储
        text_results[i], audio_results[i] = future.result()

    combined_audio = AudioSegment.empty()
    for result in audio_results:  # 按原始文本的顺序合并音频
        combined_audio += result

    combined_text = "\n".join(text_results)  # 按原始文本的顺序合并文本

    return combined_text, combined_audio


if __name__ == "__main__":
    start_server(main)
