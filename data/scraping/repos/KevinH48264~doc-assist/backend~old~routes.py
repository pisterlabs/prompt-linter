'''old file for documents'''

from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader
from gpt_index import GPTListIndex, GoogleDocsReader
import os
from dotenv import load_dotenv
from langchain import HuggingFaceHub
# from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain import OpenAI
# from gpt_index.embeddings.huggingface import HuggingFaceEmbeddings
from gpt_index import LangchainEmbedding
from gpt_index import LLMPredictor, GPTSimpleVectorIndex, PromptHelper
from bs4 import BeautifulSoup
from bs4.element import Comment
import openai
import requests
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
import sseclient
import json

load_dotenv()
hfhub_api_key = os.getenv('HUGGINGFACEHUB_API_KEY')
openai_api_key = os.getenv('OPENAI_API_KEY')
hf = HuggingFaceHub(repo_id="google/flan-t5-xl", huggingfacehub_api_token=hfhub_api_key, model_kwargs={'temperature':1e-10})
llm_predictor = LLMPredictor(llm=hf)

# define prompt helper
# set maximum input size
max_input_size = 4096

# set max chat input size
MAX_CHAT_SIZE = 1000

# set number of output tokens
num_output = 256
# set maximum chunk overlap
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

headers = requests.utils.default_headers()
headers.update({
    'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:52.0) Gecko/20100101 Firefox/52.0',
})

# WEB SCRAPE
# Helper functions to extract all text
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    # if isinstance(element, Comment):
    #     return False
    return True

def text_from_URL(URL):
    r = requests.get(URL, headers=headers)
    soup = BeautifulSoup(r.content, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return " ".join(t.strip() for t in visible_texts)

def save_text_to_dir(text, model_index):
    print('saving')
    folder_path = 'data/text_{}'.format(model_index)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open('data/text_{}/text.txt'.format(model_index), 'w') as f:
        if 'www.' in text[:min(15, len(text))]:
            text = text_from_URL(text)
        print("TEXT: ", text)
        f.write(text)

def embed(model_index, openai_api_key):
    print("embed")
    if not os.path.exists('indices'):
        os.makedirs('indices')

    # there's an openai key
    global llm_predictor
    if openai_api_key:
        openai = OpenAI(temperature=0, model_name="text-davinci-003", openai_api_key=openai_api_key)
        llm_predictor = LLMPredictor(llm=openai)
    else:
        global hf
        llm_predictor = LLMPredictor(llm=hf)


    # 1.5. Customizing LLM to use google/flan-t5-x1 (free version) instead of OpenAI's text-davinci-003 (pro version?) model
    # define LLM
    documents = SimpleDirectoryReader('data/text_{}'.format(model_index)).load_data()
    index = GPTSimpleVectorIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.save_to_disk('indices/index_{}.json'.format(model_index))

# queries 
def query(question, model_index, type):
    print("query", question, model_index, type)
    # 1.5. Customizing LLM to use google/flan-t5-x1 (free version) instead of OpenAI's text-davinci-003 (pro version?) model
    if type == "text":
        index = GPTSimpleVectorIndex.load_from_disk('indices/index_{}.json'.format(model_index), llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    elif type =="url":
        index = GPTListIndex.load_from_disk('indices/index_{}.json'.format(model_index), llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    response = index.query(question, verbose=True)
    print("response: ", response)
    return response

def embed_google_docs(document_ids, model_index):
    # make sure credentials.json file exists
    '''document_ids = ["<document_id>"]'''
    document_ids_extracted = []
    for id in document_ids:
        start_idx = id.find('/document/d/')
        end_idx = id.find('/edit')
        if start_idx and end_idx:
            document_ids_extracted += [id[start_idx + 12:end_idx]]
    print('document_ids_extracted', document_ids_extracted)

    # 1. Load in Documents
    documents = GoogleDocsReader().load_data(document_ids=document_ids_extracted)
    


    # attempt at using all-mpnet-base-v2 embedding model
    # index = GPTListIndex([])

    # model_name = "sentence-transformers/all-mpnet-base-v2"
    # hf = HuggingFaceEmbeddings(model_name=model_name)
    # embed_model = LangchainEmbedding(hf)

    # doc_chunks = []
    # for i, text in enumerate(text_chunks):
    #     embedding = embed_model.get_text_embedding(text)
    #     doc = Document(text, embedding = embedding)
    #     doc_chunks.append(doc)

    # for doc_chunk in doc_chunks:
    #     index.insert(doc_chunk)
    
    # 1.5. Customizing LLM to use google/flan-t5-x1 (free version) instead of OpenAI's text-davinci-003 (pro version?) model
    # 2. Index Construction
    # Seems like there are 0 LLM calls?
    index = GPTListIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index.save_to_disk('indices/index_{}.json'.format(model_index))
    print("finished embedding!")

# chatting for chrome extension with LangChain
def chat(question, website="", chat_history=""):
    print("chat", question, website, type)

    # default prompt
    prompt = question
    
    # prompt using website
    if website:
        prompt = "Given this information: " + website[:MAX_CHAT_SIZE - len(question)] + ", respond conversationally to this prompt: " + question

    template = """Given the text of question, respond conversationally to that question.
    {chat_history}
    Human: {prompt}
    AI:
    """

    prompt_template = PromptTemplate(input_variables=["chat_history", "prompt"], template=template)
    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(
        # llm=OpenAI(model_name="gpt-3.5-turbo", temperature=0.9),
        llm=OpenAI(temperature=0.9),
        prompt=prompt_template,
        verbose=True,
        memory=memory,
    )

    response = llm_chain.predict(prompt=prompt)

    # return the response
    print("response: ", response)
    return response

# test_document_id = ["https://docs.google.com/document/d/1Dz_5pLLpQ4LZN_zblxiIHRn_BIINClHYJsEpRkWnJnc/edit?usp=sharing"]
# # test_document_id = ["https://docs.google.com/document/d/1rgHsWyuHWIF9Z8M2I-ceCwnOgiIofvwa2rr1iDhPg94/edit"]
# embed_google_docs(test_document_id, 4)
# print(query("summarize it", 4, "url"))

def chat_openai(question="Tell me to ask you a prompt", website="", chat_history=[]):
    # define prompt
    prompt = question
    if website:
        prompt = "Given this information: " + website[:MAX_CHAT_SIZE - len(question)] + ", respond conversationally to this prompt: " + question

    # define message conversation for model
    if chat_history:
        messages = chat_history
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant, a large language model trained by OpenAI. Answer as concisely as possible. If you're unsure of the answer, say 'Sorry, I don't know'"},
        ]
    messages.append({"role": "user", "content": prompt})

    # create the chat completion
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    text_answer = completion["choices"][0]["message"]["content"]

    # updated conversation history
    messages.append({"role": "assistant", "content": text_answer})

    return text_answer, messages

def chat_openai_stream(question="Tell me to ask you a prompt", website="", chat_history=[]):
    print("chatting openai!")

    # define prompt
    prompt = question
    if website:
        prompt = "Given this information: " + website[:MAX_CHAT_SIZE - len(question)] + ", respond conversationally to this prompt: " + question

    # define message conversation for model
    if chat_history:
        messages = chat_history
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant, a large language model trained by OpenAI. Answer as concisely as possible. If you're unsure of the answer, say 'Sorry, I don't know'"},
        ]
    messages.append({"role": "user", "content": prompt})


    # create the chat completion
    reqURL = 'https://api.openai.com/v1/chat/completions'
    reqHeaders = {
        'Accept': 'text/event-stream',
        'Authorization': 'Bearer ' + openai_api_key
    }
    reqBody = {
        'model': 'gpt-3.5-turbo',
        'messages': messages,
        'stream': True,
        'temperature': 0.2
    }

    # updated conversation history
    request = requests.post(reqURL, stream=True, headers=reqHeaders, json=reqBody)
    client = sseclient.SSEClient(request) # creates a generator

    # for event in client.events():
    #     if event.data != '[DONE]' and 'content' in json.loads(event.data)['choices'][0]["delta"]:
    #         # print(event.data)
    #         print(json.loads(event.data)['choices'][0]["delta"]["content"], end="", flush=True)

    # TODO: update on frontend side when DONE token is hit to append placeholder to messages
    # messages.append({"role": "assistant", "content": text_answer})

    return client, messages

# question = "Where are the most famous egyptian pyramids found?"
# website = "The most famous Egyptian pyramids are those found at Giza, on the outskirts of Cairo. Several of the Giza pyramids are counted among the largest structures ever built.[9] The Pyramid of Khufu is the largest Egyptian pyramid. It is the only one of the Seven Wonders of the Ancient World still in existence, despite its being the oldest wonder by about 2,000 years.[10]"
# chat(question, website)

client, messages = chat_openai_stream(question = "Tell me a joke", website="The most famous Egyptian pyramids are those found at Giza, on the outskirts of Cairo. Several of the Giza pyramids are counted among the largest structures ever built.[9] The Pyramid of Khufu is the largest Egyptian pyramid. It is the only one of the Seven Wonders of the Ancient World still in existence, despite its being the oldest wonder by about 2,000 years.[10]")
# print("res", res)

for event in client.events():
    if event.data != '[DONE]' and 'content' in json.loads(event.data)['choices'][0]["delta"]:
        print(json.loads(event.data)['choices'][0]["delta"]["content"], end="", flush=True)
