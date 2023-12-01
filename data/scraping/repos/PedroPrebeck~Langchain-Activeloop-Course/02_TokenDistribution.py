from dotenv import load_dotenv

load_dotenv()

from langchain.llms import OpenAI

llm = OpenAI(model_name="text-davinci-003", temperature=0)

text = "What would be a good company name for a company that makes colorful socks?"

print(llm(text))

from langchain.callbacks import get_openai_callback

llm = OpenAI(model_name="text-davinci-003", n=2, best_of=2)

with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    print(cb)
