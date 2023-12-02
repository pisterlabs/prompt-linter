# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import os
os.environ["OPENAI_API_KEY"] = "sk-"
openai.api_key = os.getenv("OPENAI_API_KEY")
audio_file= open("1.mp3", "rb")
response = openai.Audio.transcribe("whisper-1", audio_file)
# 提取出转录文本
print(response)
transcript = response.text
# 将转录结果写入到文本文件
with open('transcription.txt', 'w') as f:
    f.write(transcript)