"""
A chatbot that will answer using australian slang
"""
import os
import time
import gradio as gr
import openai
from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')


def get_template() -> str:
    """
    Returns the template for the chatbot
    """
    template = """Brissy is an Australian Slang Chatbot based on large language model.

    Brissy is a fair dinkum Aussie model and knows all about Australian slang. It's a top-notch mate and can answer questions about Australia, Aussie culture, and a whole bunch of other topics. It always uses friendly slang and can chat like a true blue Aussie. Brissy start answering every question differently. Brissy will always answer every question within 4000 characters.

    Reckon you can rewrite your response using Australian slang?

    {history}
    Human: {human_input}
    Brissy:"""

    return template


def get_chain() -> LLMChain:
    """
    Returns the chatbot chain
    """
    template = get_template()

    prompt = PromptTemplate(
        input_variables=['history', 'human_input'],
        template=template
    )

    chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1.0)

    chatgpt_chain = LLMChain(
        llm=chat,
        prompt=prompt,
        verbose=True,
        memory=ConversationBufferWindowMemory(k=5),
    )
    return chatgpt_chain


chatgpt_chain = get_chain()

def response(message, history):
    response = chatgpt_chain.predict(human_input=message)
    for i in range(len(response)):
         time.sleep(0.01)
         yield response[:i+1]
    

if __name__ == '__main__':
    gr.ChatInterface(response).queue().launch()