# imports
import pinecone
from pinecone.core.client.exceptions import NotFoundException

from langchain import LLMChain
from langchain import PromptTemplate
#from langchain.llms import OpenAI (old depreciated)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import Pinecone

import streamlit as st

from streamlit_chitchat import message
from StreamlitCallbackHandlerOEHV import StreamlitCallbackHandlerOEHV
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain

# constants
model_name = "gpt-4"  # 'gpt-3'
#model_name = "gpt-3.5-turbo"
#model_name = "gpt-4-0613"

openai_api_key = st.secrets['OPENAI_API_KEY']
pinecone_api_key = 'b8aadd4c-6fe0-4de9-8f1e-28794846b692'  # find at app.pinecone.io

#pinecone_api_key = 'f0b3d3aa-2924-4af0-a2c4-934c6daed97b'
#pinecone_environment = 'gcp-starter'  # next to api key in console
pinecone_environment = 'northamerica-northeast1-gcp'

## store messages
if 'bot' not in st.session_state:
    st.session_state.bot = []

if 'temp' not in st.session_state:
    st.session_state.temp = ''

if 'user' not in st.session_state:
    st.session_state.user = []

## store model
if 'model' not in st.session_state:
    st.session_state.model = None

if 'handler' not in st.session_state:
    st.session_state.handler = None

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Functions
# @st.cache_resource
def get_model() -> ConversationalRetrievalChain:
    if st.session_state.model is not None:
        return st.session_state.model
    
    # Construct a ConversationalRetrievalChain with a streaming llm for combine docs
    # and a separate, non-streaming llm for question generation
    llm = ChatOpenAI(temperature=0)
    #llm = ChatOpenAI()
    handler = StreamlitCallbackHandlerOEHV()
    st.session_state.handler = handler

    streaming_llm = ChatOpenAI(streaming=True, callbacks=[handler], temperature=0, model_name=model_name, max_tokens = 300)
    #streaming_llm = ChatOpenAI(streaming=True, callbacks=[handler], model_name=model_name)

    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

    prompt_template = """You are an AI that only answers questions related to Sportresort Alpenblick. You first check the Question and answer in the same language of the Question.
    Do not answer if you are not sure. Do not write anything as answer that is not related to the Sportressort Alpenblick. Give a structured and short answer. Only if a question is about booking a room, answer that this information is not available as you are a AI system and they should either call the hotel (+43 6542 5433) or use the Inquiry form on the website. For requests in English provide this link: https://www.alpenblick.at/en/inquire, for German requests show this link: https://www.alpenblick.at/de/anfragen. Provide this links only for booking inquieries. The hotels address is: Alte Landesstr. 6, 5700 Zell am See. 
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Answer in the language of the user prompt.

{context}

Question: {question}
Helpful Answer:"""

    qa_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=qa_prompt)

    qa = ConversationalRetrievalChain(
        retriever=ds.as_retriever(), combine_docs_chain=doc_chain, question_generator=question_generator)
    
    st.session_state.model = qa

    return st.session_state.model


def clear_input():
    st.session_state.user.append(st.session_state.text)
    st.session_state.text = ''


# initialize llm
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# initialize pinecone
#st.write(pinecone.list_indexes())
#st.write(pinecone.describe_index('alpen2'))

#index_name = "alpen2"
#index_name = "chatbot"

pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_environment
)


#pinecone.create_index("sara-index", dimension=256, metric="cosine", replicas = 1, shards = 1, pods=1, pod_type="p1.x1")

index_name = "alpen2"

ds = Pinecone.from_existing_index(index_name, embeddings)
conversation_chain = get_model()
handler = st.session_state.handler

# display website
## display header

#st.image("https://storage.googleapis.com/my_publications/segl.jpg")

## display chat history
history = st.container()

with history:
    if st.session_state.user:
        for um, bm in zip(st.session_state.user, st.session_state.bot):
            message(um, is_user=True, background='lightgreen')
            message(bm, background='lightblue')

    current_user_question = message(is_user=True, background='lightgreen')
    handler.tokens_area = message(background='lightblue')

footerSection = st.container()

user_question = footerSection.text_input(
    "Wie können wir Ihnen weiterhelfen? Sie können ganze Sätze schreiben... Ask us in any language!", key='text',
    on_change=clear_input)

if st.session_state.user:
    user_question = st.session_state.user[-1]

if user_question:
    st.session_state.user.append(user_question)
    current_user_question.write(user_question) 
    response = conversation_chain.run( {'question':user_question, 'chat_history':st.session_state.chat_history} )

    st.session_state.chat_history.append((user_question, response))
    st.session_state.bot.append(handler.tokens_stream)
    
    handler.tokens_stream = ''

    st.markdown(
        "To speak with one of our employees, please call us at 0043 6542 5433 or write us on WhatsApp [click here](https://api.whatsapp.com/send/?phone=4367764828204&text=Los+gehts&type=phone_number&app_absent=0) \n"
        "Rate our Chatbot and help us to improve it: "
        "[click here](https://forms.gle/WKJqdjTicZkAiJre8)"
    )
