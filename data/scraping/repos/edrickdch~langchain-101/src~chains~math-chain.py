from langchain import OpenAI, LLMMathChain
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(temperature=0)
llm_math = LLMMathChain.from_llm(llm, verbose=True)

llm_math.run("What is 13 raised to the .3432 power?")
