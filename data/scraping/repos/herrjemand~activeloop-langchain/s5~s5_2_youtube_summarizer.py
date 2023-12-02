from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

import yt_dlp

def download_mp4_from_youtube(url):
    filename = "lectuninterview.mp4"

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': filename,
        'quiet': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)

url = "https://www.youtube.com/watch?v=mBjPyte2ZZo"
# download_mp4_from_youtube(url)

# Whisper

# import whisper

# model = whisper.load_model("base")
# result = model.transcribe("lectuninterview.mp4")


# with open("transcript.txt", "w") as f:
#     f.write(result["text"])

# Summarizing

from langchain import OpenAI, LLMChain
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

llm = OpenAI(model_name="text-davinci-003", temperature=0.0)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"])


from langchain.docstore.document import Document

with open("transcript.txt", "r") as f:
    text = f.read()

texts = text_splitter.split_text(text)
docs = [Document(page_content=text) for text in texts[:4]]

import textwrap

chain = load_summarize_chain(llm, chain_type="map_reduce")
output_summary = chain.run(docs)

wrapped_text = textwrap.fill(output_summary, width=100)
print(wrapped_text)

print(chain.llm_chain.prompt.template)

bullet_point_prompt = PromptTemplate(
    template="""Write a concise bullet point summary of the following:

{text}

CONSCISE SUMMARY IN BULLET POINTS:""",
    input_variables=["text"],
)

chain = load_summarize_chain(
    llm,
    chain_type="stuff",
    prompt=bullet_point_prompt
)

output_summary = chain.run(docs)

wrapped_text = textwrap.fill(
    output_summary,
    width=1000,
    break_long_words=False,
    replace_whitespace=False,
)

print(wrapped_text)


# Refine

chain = load_summarize_chain(llm, chain_type="refine")
output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
print(wrapped_text)
