from langchain.agents import create_csv_agents
from langchain.llms import OpenAI
from dotenv import load_dotenv


def create_csv_agent(csv_file,user_Question):
   
   load_dotenv()
   if csv_file is not None:
      llm = OpenAI(temperature=0)
      agent = create_csv_agent(llm, csv_file, verbose=True)

    


if __name__ == '__main__':
   create_csv_agent()