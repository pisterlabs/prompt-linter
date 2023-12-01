import openai
import re

openai.api_key = "YOUR_API_KEY"  # 将YOUR_API_KEY替换为你的OpenAI API密钥

def generate_lyrics(prompt, style):
    # 根据曲风和风格来设置生成歌词的参数
    if style == "pop":
        max_tokens = 1024
        temperature = 0.5
        stop = ["\n\n"]
    elif style == "rock":
        max_tokens = 2048
        temperature = 0.7
        stop = ["\n\n", "Verse:"]
    elif style == "country":
        max_tokens = 512
        temperature = 0.3
        stop = ["\n\n", "Chorus:"]
    else:
        raise ValueError("Invalid style. Please choose from 'pop', 'rock', or 'country'.")

    # 发送请求生成歌词
    response = openai.Completion.create(
      engine="davinci",
      prompt=prompt,
      max_tokens=max_tokens,
      n=1,
      stop=stop,
      temperature=temperature,
    )

    # 获取生成的歌词并清洗
    lyrics = response.choices[0].text
    lyrics = re.sub('[^0-9a-zA-Z\n.,?! ]+', '', lyrics)
    lyrics = lyrics.strip()

    return lyrics

# 获取用户输入的歌曲标题和曲风
song_title = input("请输入歌曲标题：")
song_style = input("请输入曲风（可选项：'pop', 'rock', 'country'）：")

# 生成歌曲
song_prompt = f"生成一首名为'{song_title}'的{song_style}歌曲，歌词如下：\n\n"
song_lyrics = generate_lyrics(song_prompt, song_style)

# 输出生成的歌曲歌词
print(f"\n\n生成的'{song_title}'的{song_style}歌曲的歌词：\n\n{song_lyrics}")
