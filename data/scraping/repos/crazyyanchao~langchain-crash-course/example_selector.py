from util.keys import initial

# 初始化秘钥配置
initial()

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

examples = [
    {"question":
         """
        Who played in Second Fix?
        """,
     "cypher":
         """
         MATCH (a:Actor)-[:ACTED_IN]->(m:Movie {name: 'Second Fix'})
         RETURN a.name AS result;
    """},
    {"question":
         """
        How many actors are there in Second Fix?
        """,
     "cypher":
         """
         MATCH (a:Actor)-[:ACTED_IN]->(m:Movie {name: 'Second Fix'})
         RETURN COUNT(a) AS result;
    """},
]


def extract(data):
    questions = []
    for item in data:
        if "question" in item:
            questions.append({'question': item["question"]})
    return questions


def split_and_get_second_element(string, delimiter):
    elements = string.split(delimiter)
    if len(elements) >= 2:
        return elements[1]
    else:
        return None


def get_cypher(qa):
    for item in examples:
        if "question" in item and qa in item["question"].strip():
            return item['cypher']
    return None


class ExampleSelector():
    similar_prompt = None

    def __init__(self):
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            # This is the list of examples available to select from.
            extract(examples),
            # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
            OpenAIEmbeddings(),
            # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
            Chroma,
            # This is the number of examples to produce.
            k=1
        )

        self.similar_prompt = FewShotPromptTemplate(
            # We provide an ExampleSelector instead of examples.
            example_selector=example_selector,
            example_prompt=PromptTemplate(
                input_variables=["question"], template="Question: {question}",
            ),
            prefix="Find the most relevant questions",
            suffix="# {question}",
            input_variables=["question"],
        )

    def get_examples(self, question):
        qa = split_and_get_second_element(
            self.similar_prompt.format(question=question),
            "\n        ")
        cypher = get_cypher(qa)
        return f"# {qa}\n{cypher}"


# similar_prompt = ExampleSelector()
#
# print(similar_prompt.get_examples(question="Who played in Top Gun?"))

# print(get_cypher("Who played in Second Fix?"))

