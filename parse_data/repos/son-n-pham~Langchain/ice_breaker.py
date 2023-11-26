from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

import os

openai_api_key = os.environ.get("OPENAI_API_KEY")

information = """
"""

if __name__ == "__main__":
    print(openai_api_key)
