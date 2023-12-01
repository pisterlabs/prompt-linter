from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = Ollama(model='llama2',
             callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
             temperature=0.1, )

prompt = PromptTemplate(
    input_variables=['topic'],
    template='Can you tell me 5 things about {topic} ?')

chain = LLMChain(llm=llm, prompt=prompt)

response = chain.run('Psyche asteroid')
print(response)
