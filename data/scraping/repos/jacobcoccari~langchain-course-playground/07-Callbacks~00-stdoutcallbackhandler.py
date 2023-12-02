# StdOutCallBackHandler
from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

handler = StdOutCallbackHandler()
model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template("1 + {number} = ")

# Constructor callback: First, let's explicitly set the StdOutCallbackHandler when initializing our chain
chain = LLMChain(llm=model, prompt=prompt, callbacks=[handler])
chain.run(number=2)

# Use verbose flag: Then, let's use the `verbose` flag to achieve the same result
chain = LLMChain(llm=model, prompt=prompt, verbose=True)
chain.run(number=2)

# Request callbacks: Finally, let's use the request `callbacks` to achieve the same result
chain = LLMChain(llm=model, prompt=prompt)
chain.run(number=2, callbacks=[handler])
