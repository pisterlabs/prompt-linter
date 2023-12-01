from langchain.chat_models import ChatOpenAI
import os
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv(find_dotenv())

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
os.environ["openai_api_key"] = OPENAI_API_KEY

def _respond(user_input):
    chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    # CHANGE THIS:
    template = """
    
    You're (name), a personal assistant.
    Your goal is to provide clear and concise advice on diverse topics (etc). 
    Use simple language and avoid technical jargon, unless explicitly asked for by the user.
    Be responsive and helpful to users.
    Make sure to sign off with {signature}.
    
    """

    signature = "Best, your personal bot" # CHANGE THIS
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)

    human_template = "Here's what the user is asking you: {user_input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(user_input=user_input, signature=signature)

    return response
