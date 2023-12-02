from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler    

llm = Ollama(model="yi:34b-q4_K_M", 
            #  callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0.9,
             )

from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["topic"],
    template="给我5个关于{topic}的事实？",
)

from langchain.chains import LLMChain
chain = LLMChain(llm=llm, 
                 prompt=prompt,
                 verbose=False)

# Run the chain only specifying the input variable.
print(chain.run("the moon"))

