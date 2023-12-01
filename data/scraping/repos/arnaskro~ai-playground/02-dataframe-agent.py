"""
Pandas agent and ChatGPT for Data Analysis
"""
import dotenv
dotenv.load_dotenv()

from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd

df = pd.read_csv('../sample-data/01-income-survey.csv')

chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.0)
agent = create_pandas_dataframe_agent(chat, df, verbose=True, agent_name='Pandas Agent')


from colorama import Fore

def loop():
    text = input("\n\n" + Fore.WHITE + ">>> ")

    if text == "exit":
        print (Fore.RED +"Bye!")
    else:
        answer = agent.run(text)
        print(Fore.YELLOW + 'AI:', answer)
        loop()

loop()