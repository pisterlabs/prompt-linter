#!/usr/bin/env python3
# -*- coding utf-8 -*-

import openai
import yaml
import os
from pydub import AudioSegment

'''
对博客文件先进行转换为mp3
然后分割音频文件
对每个文件语音转文字
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


if __name__ == '__main__':
    get_api_key()

    # 使用ffmpeg将博客转换为mp3格式
    # ffmpeg -i ./data/podcast_long.mp4 -vn -c:a libmp3lame -q:a 4 ./data/podcast_long.mp3

    # 加载MP3文件
    podcast = AudioSegment.from_mp3("./data/podcast_long.mp3")

    # 使用PyDub将音频文件拆分为15分钟一个
    total_length = len(podcast)
    fifteen_minutes = 15 * 60 * 1000

    start = 0
    index = 0
    while start < total_length:
        end = start + fifteen_minutes
        if end < total_length:
            chunk = podcast[start:end]
        else:
            chunk = podcast[start:]
        with open(f"./data/podcast_clip_{index}.mp3", "wb") as f:
            chunk.export(f, format="mp3")
        start = end
        index += 1
    
    # 对每个音频文件，进行语音转文字
    prompt = "这是一段Onboard播客，里面会聊到ChatGPT以及PALM这个大语言模型。这个模型也叫做Pathways Language Model。"
    for i in range(index):
        clip = f"./data/podcast_clip_{i}.mp3"
        audio_file= open(clip, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file, prompt=prompt)
        
        # 如有需要创建结果输出文件夹
        if not os.path.exists("./data/transcripts"):
            os.makedirs("./data/transcripts")

        # 输入翻译后结果
        with open(f"./data/transcripts/podcast_clip_{i}.txt", "w") as f:
            f.write(transcript['text'])

        # 获取本次翻译的最后一句话，作为下一次的promot
        sentences = transcript['text'].split("。")
        prompt = sentences[-1]
    
