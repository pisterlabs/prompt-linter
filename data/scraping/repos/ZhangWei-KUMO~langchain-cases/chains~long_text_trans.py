
from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv('.env')
text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=1600)
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

loader = YoutubeLoader.from_youtube_url('https://www.youtube.com/watch?v=i-wpzS9ZsCs',add_video_info=True)
data = loader.load()
if(len(data) == 0):
    print("没有找到英文字幕")
    exit()
texts = text_splitter.split_documents(data)
print(len(texts))
t = "What is the meaning of life?"
openai.api_key ='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'

response = openai.Completion.create(
  model="text-davinci-003",
  prompt="Translate this into Chinese:\n\n{t}\n\n1.",
  temperature=0.3,
  max_tokens=1800,
  top_p=1.0,
  frequency_penalty=0.0,
  presence_penalty=0.0
)
print(response)
# llmchain = LLMChain(llm=llm, prompt=prompt)
# chinese = llmchain.run(language="简体中文", text=texts[1])
# print(texts[1],chinese)
