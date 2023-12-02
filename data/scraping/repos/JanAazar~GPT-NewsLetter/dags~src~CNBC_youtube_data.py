from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
import os
from src.auth import openai_api_key as api_key
import openai

os.environ["OPENAI_API_KEY"] = api_key


loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=KYjrD4HHnJM", add_video_info=True)

result = loader.load()

# print (type(result))
video_title = result[0].metadata["title"]
video_description = result[0].page_content
prompt ="Summarize the following CNBC video: " + video_title + "\n" + video_description 
summary = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])

print(summary.choices[0].message.content)
