import os
import logging
from slack_bolt import App
from dotenv import load_dotenv
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

load_dotenv()

app = App(token=os.getenv('slack_bot_token'))

logging.basicConfig(filename=os.getenv('logfile_path'), filemode='a', datefmt='%H:%M:%S',
                    format='%(asctime)s\t[%(name)s]\t%(levelname)s\t%(message)s', level=logging.WARNING)

template = open(os.getenv('template_path')).read()
prompt = PromptTemplate(
    input_variables=['history', 'human_input'],
    template=template
)

gpt_chain = LLMChain(
    llm=OpenAI(
        temperature=0,
        model=os.getenv('ai_model'),
    ),
    prompt=prompt,
    memory=ConversationBufferWindowMemory(k=int(os.getenv('buffer_message_count'))),
    verbose=False,  # Logging to console
)
