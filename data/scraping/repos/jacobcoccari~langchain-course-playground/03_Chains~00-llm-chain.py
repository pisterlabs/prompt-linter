# I recommend using LLMChain when you need to get the highest level of customization
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

prompt_template = "On what day did {world_event} start?"

model = ChatOpenAI(temperature=0)

prompt = ChatPromptTemplate.from_template(prompt_template)

llm_chain = LLMChain(llm=model, prompt=prompt)
# you can call it directly.
# Notice how it is retuns a dictionary now, instead of an object.
# Also, it includes any of the input variables in the response.

# response = llm_chain("the cold war")
# print(response.text)
# print(type(response))

# If we want to run it on a list of inputs, we can use the .input method.
# Notice how it returns a list of dictionaries.
# Also, it does not include the input variables.
input_list = [
    {"world_event": "the cold war"},
    {"world_event": "the great depression"},
    {"world_event": "the american civil war"},
]
# result = llm_chain.apply(input_list)
# print(result)
# print(type(result))

# Just like before, we can also use the .generate method to get more information about our calls.
# Can be called on one to many inputs. However, even a single input must be in a list of dictionaries.

# result = llm_chain.generate([{"world_event": "the cold war"}])
# print(result)

# Side note: this must just inherit a call the chatopenai....
result_list = llm_chain.generate(input_list)
print(result_list)
