import logging
import os, uvicorn
from operator import itemgetter

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.globals import set_llm_cache
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import format_document
from langchain.schema.runnable import RunnableLambda, RunnableMap, RunnablePassthrough
from langchain.vectorstores import DeepLake

set_llm_cache(SQLiteCache(database_path=".langchain.db"))
app = FastAPI()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_deep_lake(path):
    embeddings = OpenAIEmbeddings(show_progress_bar=True, model_kwargs={'batch_size': 50})
    dl = DeepLake(
        dataset_path=path, embedding=embeddings, read_only=True,
    )
    return dl


local_path = "./my_deeplake/"

_template = """Answer to the best of your knowledge from the given context: {question}"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

template = """Answer the question  to the best of your knowledge based on the following context:
{context}
Also state the FAR or sections from FAA documents that discusses the topic of the answer if known and relevant.
Use bullet points if suited.
Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


try:
    db = get_deep_lake(local_path)
    retriever = db.as_retriever()

    _inputs = RunnableMap(
        standalone_question=RunnablePassthrough.assign() | RunnableLambda(lambda x: x["question"].strip()),
    )

    retrieved_documents = {
        "docs": itemgetter("standalone_question") | retriever,
        "question": lambda x: x["standalone_question"]
    }

    final_inputs = {
        "context": lambda x: _combine_documents(x["docs"]),
        "question": lambda x: x["question"],
        "docs": itemgetter("docs"),
    }

    # And finally, we do the part that returns the answers
    answer = {
        "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(model='gpt-3.5-turbo-16k-0613', temperature=0.8),
        "docs": itemgetter("docs"),
    }

    conversational_qa_chain = _inputs | retrieved_documents | final_inputs | answer

except Exception as e:
    logger.exception("Error occurred while initializing the conversational_qa_chain")
    raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")


class Question(BaseModel):
    question: str


class Doc(BaseModel):
    page_content: str
    metadata: dict


class Answer(BaseModel):
    answer: str
    docs: list[Doc]


@app.post("/ask")
def ask(question: Question):
    try:
        result = conversational_qa_chain.invoke({"question": question.question})
        return Answer(answer=result["answer"].content, docs=[Doc(page_content=doc.page_content, metadata=doc.metadata) for doc in result["docs"]])
    except Exception as e:
        logger.exception("Error occurred while processing the question")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    uvicorn.run(app, port=int(os.environ.get("PORT", 8000)), host="0.0.0.0")