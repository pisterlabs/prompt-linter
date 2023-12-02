from langchain import BasePromptTemplate
from langchain.chat_models.openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceTextGenInference

_PROMPT = (
    "I want to learn about {topic}. I would like to represent this as OWL classes in a hierarchical structure so that I can do complex OWL DL queries on the ontology. The ontology should contain not just technology classes, but also things that might be related to technology, such as a Person class. What are all of the relevant OWL classes I need to consider?"
    " It needs to be very detailed and comprehensive to include many different types of classes of potential interest. I would think about 50 classes would be good. Please output in Turtle format."
)
prompt = PromptTemplate(
    input_variables=["topic"],
    template=_PROMPT,
)

llm = ChatOpenAI(temperature=0, model="gpt-4")
chain = LLMChain(llm=llm, prompt=prompt)
res = chain.predict(topic="the latest technology news and innovation")
print(res)
