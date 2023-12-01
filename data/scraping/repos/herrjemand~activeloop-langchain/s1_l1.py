from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')


# The LLM
from langchain.llms import OpenAI

llm = OpenAI(model="text-davinci-003", temperature=0.9)
# text = "Suggest a personalized workout routine for someone looking to improve cardiovascular endurance and prefers outdoor activities."
# print(llm(text))


# The chains
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

print(chain.run("eco-friendly water bottles"))

# The Memory

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
    verbose=True,
)

conversation.predict(input="Tell me about yourself.")

conversation.predict(input="What can you do?")
conversation.predict(input="How can you help me with the data analysis?")

print(conversation)