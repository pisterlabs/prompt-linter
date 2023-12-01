import os
from langchain import LLMChain, PromptTemplate
from langchain import OpenAI
from langchain.memory import ConversationBufferWindowMemory, ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

os.environ["OPENAI_API_KEY"] = os.environ.get("key")

template = """you act like a psychologist i will give you meaning of action that person did, 
tell me the overall impression(most common impression) about the meaning of actions that the person did in one small line
if there isn't data or empty data or the data doesn't the same that you are waiting for you respond by that there is no available summarize because there is no emotion
if you can't respond say "there is no available summarize because there is no emotion"

Meaning: {human_input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["human_input"],
    template=template
)

chain = LLMChain(
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    prompt=prompt,
)


def query_1(VTTcontent):
    VTTcontent = VTTcontent.replace("WEBVTT", "")
    response = chain.predict(human_input=VTTcontent)
    return response
