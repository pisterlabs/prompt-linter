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

import langdetect
import openai


# constants
#model_name = "gpt-4"  # 'gpt-3'
#model_name = "gpt-3.5-turbo"
#model_name = "gpt-4-0613"
model_name = "gpt-4-1106-preview"

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

    prompt_template = """You are an AI that only answers questions related to Sportresort Alpenblick. 
    For requests in English provide this link: https://www.alpenblick.at/en/inquire, for German requests show this link: https://www.alpenblick.at/de/anfragen. 
    Provide this links only for booking inquieries. The hotels address is: Alte Landesstr. 6, 5700 Zell am See.

    For weather data, get weather from Google.

    Answer in the language of Language in {lang}.

    {context}

    Question: {question}
    Language:  {lang}
    Helpful Answer:"""


    qa_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question", "lang"])

    doc_chain = load_qa_chain(streaming_llm, chain_type="stuff", prompt=qa_prompt)

    qa = ConversationalRetrievalChain(
        retriever=ds.as_retriever(), combine_docs_chain=doc_chain, question_generator=question_generator)
    
    st.session_state.model = qa

    return st.session_state.model


def clear_input():
    st.session_state.user.append(st.session_state.text)
    st.session_state.text = ''



	# Initializing the Langchain object

 
def detect_lang(txt):
    """
    Detect the language of a given text using Langchain.
 
    Parameters:
    - txt: str
        The text for which the language needs to be detected.
 
    Returns:
    - str:
        The detected language of the text.
 
    Raises:
    - ValueError:
        Raises an error if the input text is empty or None.
    """

    # Checking if the input text is empty or None
    if not txt:
        raise ValueError("Input text cannot be empty or None.")

 
    # Using Langchain to detect the language of the text
    detected_lang = langdetect.detect(txt)
    return detected_lang


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

st.image("https://storage.googleapis.com/my_publications/segl.jpg")

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


#st.text_input(r"$\textsf{\Large Enter text here}$", key = 'text1')

    
#user_question = footerSection.text_input(
#    "\n :raising_hand: Hallo, Wie können wir Ihnen weiterhelfen? Sie können ganze Sätze schreiben... Ask us in any language!\n", key='text',
#    on_change=clear_input)

user_question = footerSection.text_input(
    '\n'+ ':raising_hand:' + r"$\textsf{\ Hallo, Wie können wir Ihnen weiterhelfen? Sie können ganze Sätze schreiben... Ask us in any language!}$", key='text',
    on_change=clear_input)


# SS

#st.write('')
#st.write(st.session_state.user)

#SS


# SS
#from openai import OpenAI
#client = OpenAI()

#my_assistant = client.beta.assistants.create(
#    instructions="You are a personal math tutor. When asked a question, write and run Python code to answer the question.",
#    name="Math Tutor",
#    tools=[{"type": "code_interpreter"}],
#    model="gpt-4",
#)
#st.write(my_assistant)


## SS

if st.session_state.user:
    user_question = st.session_state.user[-1]

if user_question:
# SS
#    st.session_state.user.append(user_question)
#    response = conversation_chain.run( {'question':user_question, 'chat_history':st.session_state.chat_history} )
# SS
    current_user_question.write(user_question)



    lang = detect_lang(user_question)

    #st.write('Language' , lang)

    
    response = conversation_chain.run( {'question':user_question, 'chat_history':st.session_state.chat_history, 'lang': lang} )

    st.session_state.chat_history.append((user_question, response))
    st.session_state.bot.append(handler.tokens_stream)
    
    handler.tokens_stream = ''

    st.markdown(
        "\n\nTo speak with one of our employees, please call us at 0043 6542 5433 or write us on WhatsApp [click here](https://api.whatsapp.com/send/?phone=4367764828204&text=Los+gehts&type=phone_number&app_absent=0) \n"
        "Rate our Chatbot and help us to improve it: "
        "[click here](https://forms.gle/WKJqdjTicZkAiJre8)"
    )
