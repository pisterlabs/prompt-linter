from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI



# LLM is core reasoning engine
# Prompt template : Instruction to OpenAI
# Ouput: Translate LLM output to human



import os
from dotenv import dotenv_values

def load_from_dot_env() -> str:
    config = dotenv_values('.env')
    return config['OPENAI_API_KEY']


if __name__ == '__main__':
    os.environ['OPENAI_API_KEY'] = load_from_dot_env()
    llm = OpenAI()
    text = 'Suggest channel names based on company that makes youtube videos on AI'
    print(llm.predict(text))

    chat_model = ChatOpenAI()
    print(chat_model.predict(text))

