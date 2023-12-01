from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain import LLMChain

from langchain.prompts import load_prompt
loaded_prompt = load_prompt("./prompts/Yasuke.json")

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
chain = LLMChain(llm=chat, prompt=loaded_prompt)
print(chain.run("What's the secret to happiness?"))

