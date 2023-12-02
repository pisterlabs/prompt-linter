from util.keys import initial

# 初始化秘钥配置
initial('../../.env')

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

example_prompt = PromptTemplate(
    input_variables=["question"],
    template="Question: {question}",
)

# These are a lot of examples of a pretend task of creating antonyms.
examples = [
    {"question": "How many actors are there in Second Fix?"},
    {"question": "Who played in Second Fix?"},
    {"question": "Who played in Second-2 Fix?"}
]

example_selector = SemanticSimilarityExampleSelector.from_examples(
    # This is the list of examples available to select from.
    examples,
    # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
    OpenAIEmbeddings(),
    # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
    Chroma,
    # This is the number of examples to produce.
    k=1
)

similar_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Find the most relevant question",
    suffix="Input: {question}",
    input_variables=["question"],
)

# Input is a feeling, so should select the happy/sad example
print(similar_prompt.format(question="How many actors are there in Top Gun?"))

# Input is a measurement, so should select the tall/short example
# print(similar_prompt.format(question="fat"))

# You can add new examples to the SemanticSimilarityExampleSelector as well
# similar_prompt.example_selector.add_example({"input": "enthusiastic", "output": "apathetic"})
# print(similar_prompt.format(question="joyful"))

