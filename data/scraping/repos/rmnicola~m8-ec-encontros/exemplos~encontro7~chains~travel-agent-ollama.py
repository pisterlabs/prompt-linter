from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate

model1 = Ollama(model="mistral")
model2 = Ollama(model="dolphin2.2-mistral")

prompt1 = ChatPromptTemplate.from_template("""
Apenas traduza o texto a seguir:
{text}
"""
)

prompt2 = ChatPromptTemplate.from_template( """
You are now my personal travel agent. Act as someone who has immense travel
experience and knows the best places in the world to do certain activities. I
want to know where I should go to {activity}. Give the answers as a list of
items, no bigger than 5 items. Only respond with the list of 5 items, no
summarizing statement, forewords or warnings or even explanations are required.
"""
)

chain1 = prompt1 | model2

chain2 = {"activity": chain1} | prompt2 | model1

chain3 = {"text": chain2} | prompt1 | model2

for s in chain3.stream({"text": "caminhar na praia"}):
    print(s, end="", flush=True)
print()
