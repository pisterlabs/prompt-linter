from langchain import OpenAI
from langchain.agents import create_csv_agent

file_path = "/Users/senthis/Desktop/CoolProject/train 2.csv"

openai_api_key = "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"

llm = OpenAI(temperature=1, openai_api_key=openai_api_key)
agent = create_csv_agent(llm, file_path, verbose=True)

agent.run("how can i improve my revenue per customer using this data. Be very analytical and use your business knowledge. Do not exceed 1 sentence however.")