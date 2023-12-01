import pandas as pd
from langchain import LLMChain, FewShotPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.schema import SystemMessage
from langchain.vectorstores import Chroma

from axxesHackathon2023 import settings

df = pd.read_csv(settings.DATASET_URL)
df_qa = df[['question', 'answers']]
loader = DataFrameLoader(df, page_content_column="question")
docs = loader.load()
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings, collection_name="axxess")

role_prompt = """You are a tutor for students relating to the medical field. Your job is to respond with answers to health related questions
and explain the answers in a way that is easy to understand. You are also expected to provide sources for your answers. Be sure to be honest
if you don't know the answer to a question. Do not make up sources! {chat_history}"""


# %%


def get_suggestions(search_query):
    suggestions = db.max_marginal_relevance_search(search_query,
                                                   fetch_k=10)  # gets a variety of other questions related to the query that the user can branch off to
    suggestion_questions = []
    for sug in suggestions:
        content = sug.page_content
        suggestion_questions.append(content.capitalize())
    return suggestion_questions


def get_summary(search_query):
    output = db.similarity_search(search_query)
    qa = []
    for res in output:
        qa.append({"Question": f"{res.page_content}", "Answer": f"{res.metadata['answers']}"})
    system_message_prompt = SystemMessage(content=role_prompt)
    example_prompt = PromptTemplate(input_variables=["Question", "Answer"],
                                    template="Question: {Question}\n{Answer}")
    fewShotPrompt = FewShotPromptTemplate(
        examples=qa,
        example_prompt=example_prompt,
        suffix="Question: {input}",
        input_variables=["input"]
    )
    human_message_prompt = HumanMessagePromptTemplate(prompt=fewShotPrompt)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chat = ChatOpenAI(temperature=0)
    qa_chain = LLMChain(
        llm=chat,
        prompt=chat_prompt
    )
    return qa_chain.run(search_query)


class SummaryQuery:
    def __init__(self, search_query):
        self.message = get_summary(search_query)
        self.suggestions = get_suggestions(search_query)
