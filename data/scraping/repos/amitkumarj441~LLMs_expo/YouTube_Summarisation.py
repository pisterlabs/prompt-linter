# Fire this command "OPENAI_API_KEY=xxxx python summarize.py <youtube_id>"

import sys

from youtube_transcript_api import YouTubeTranscriptApi

from langchain import OpenAI, PromptTemplate
from langchain.text_splitter import TokenTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain


srt = YouTubeTranscriptApi.get_transcript(sys.argv[1])
captions = [c['text'] for c in srt]
text = ' '.join(captions)

llm = OpenAI(temperature=0, max_tokens=1000)

text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=100)
docs = text_splitter.create_documents([text])

prompt_template = """Summarize in 10 bullet points the following presentation:

{text}"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
chain = load_summarize_chain(llm, chain_type="map_reduce", combine_prompt=PROMPT)

# run langchain chain and print results
result = chain.run(docs)
print(result)
