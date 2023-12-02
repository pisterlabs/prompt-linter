# pip install youtube-transcript-api
# pip3 install pytube
from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
import os
from dotenv import load_dotenv
load_dotenv('.env')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

loader = YoutubeLoader.from_youtube_url('https://www.youtube.com/watch?v=pNcQ5XXMgH4',add_video_info=True)
result = loader.load()
print(type(result))
print(f"result: {result[0].metadata['author']}, {result[0].metadata['title']},总时长为{result[0].metadata['length']}秒")
llm = OpenAI(temperature=0.7,openai_api_key=OPENAI_API_KEY)
chain = load_summarize_chain(llm, chain_type="stuff",verbose=False)
result = chain.run(result)
print(result)