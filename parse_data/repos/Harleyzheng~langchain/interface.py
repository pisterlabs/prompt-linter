from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

model = ChatOpenAI()

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")

chain = prompt | model

# stream
# for s in chain.stream({"topic": "bears"}):
#     print(s)

# invoke
# print(chain.invoke({"topic": "bears"}))

# batch
print(chain.batch([{"topic": "bears"}, {"topic": "cats"}]))

