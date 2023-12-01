from langchain import OpenAI
from langchain.chains import LLMMathChain
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(
    temperature=0
    )

llm_math = LLMMathChain.from_llm(llm=llm, verbose=True)

# print(llm_math.prompt.template)
print(llm_math.run("What is 13 raised to the .3432 power?"))

# Advanced: inspecting the source code of a method
# import inspect
# print(inspect.getsource(llm_math._call))