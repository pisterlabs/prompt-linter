from langchain import OpenAI, ConversationChain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from chapter_helpers import *
import requests
import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain import OpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate    
)

llm = OpenAI(temperature=0.9)
chat = ChatOpenAI(temperature=0.9)
embeddings = OpenAIEmbeddings()

assembly_url = "https://api.assemblyai.com/v2"
assembly_key = os.environ.get("ASSEMBLYAI_KEY")

headers = {
    "authorization": assembly_key,
}

# Set an optional parameter to control caption length
params = {
  "chars_per_caption": 300
}
#one of my transcriptions that had auto chapters set to true
transcript_id = '6x92vx3l5n-62a5-4128-8101-5877b0a1dbb7'

vtt_transcription_endpoint = assembly_url + "/transcript/" + transcript_id + '/vtt'
transcription_endpoint = assembly_url + "/transcript/" + transcript_id

vtt_response = requests.get(vtt_transcription_endpoint, params=params, headers=headers)

response = requests.get(transcription_endpoint, params=params, headers=headers).json()

chapters = response['chapters']

chapter_text = text_from_chapters(chapters, vtt_response.text)

db = Chroma.from_texts(chapter_text, embeddings)

query = "how did alluo integrate superfluid into their auto invest product?"
docs = db.similarity_search(query)

context1 = docs[0].page_content
context2 = docs[1].page_content

template = """
  You are an expert on answering questions about the Devs Do Something podcast, a show for software engineers in crypto. You are about to be asked a question about a recent episiode. If you don't know the answer, please truthfully say that you don't know.
  Here a two pieces of context that will help you answer your question:
  First piece of context: 
  {context1}
  Second piece of context:
  {context2}
"""

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_template = "{text}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chain = LLMChain(llm=chat, prompt=chat_prompt)

print(chain.run(context1=context1, context2=context2, text=query))