from langchain.chat_models import ChatOpenAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import FAISS 
from langchain.schema import HumanMessage
from langchain.embeddings import OpenAIEmbeddings

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)

template ="Example input: {input} \n Example output: {output}"
example_prompt = PromptTemplate(input_variables=["input", "output"], template=template)

examples = [
    {"input": "pirate", "output": "ship"},
    {"input": "pilot", "output": "plane"},
    {"input": "driver", "output": "car"},
    {"input": "tree", "output": "ground"},
    {"input": "bird", "output": "nest"},
]

example_selector = SemanticSimilarityExampleSelector.from_examples(examples, OpenAIEmbeddings(),FAISS, k=2)

similar_prompts = FewShotPromptTemplate(example_selector=example_selector, example_prompt=example_prompt,
                                        prefix="Give the location an item is usually found in",
                                        suffix="Input: {noun}\nOutput:",
                                        input_variables=["noun"])
my_noun = "student"

response = model([HumanMessage(content=similar_prompts.format(noun=my_noun))])
print(similar_prompts.format(noun=my_noun))
print(response)