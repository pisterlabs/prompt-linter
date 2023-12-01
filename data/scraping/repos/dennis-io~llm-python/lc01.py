from dotenv import load_dotenv
from langchain.llms import OpenAI

load_dotenv()

llm = OpenAI(temperature=0.6)
predict = llm.predict("What will my longrun be like today?")
print(predict)
