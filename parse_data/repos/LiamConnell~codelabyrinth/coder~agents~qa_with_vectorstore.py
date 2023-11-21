from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document

from coder.utils import get_git_hash, ConversationLogger, summarize_title
from coder.vectorstore import VectorStore

prompt_template = "Context:\n{context}\n\n Question: {question}"
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = ChatOpenAI(model_name="gpt-4", temperature=0)
chain = LLMChain(llm=llm, prompt=PROMPT)


def agent(question, vectorstore_collections: list[str]):
    v = VectorStore()
    vectorstore_collection = vectorstore_collections[0]
    docs = v.similarity_search(vectorstore_collection, question, k=5)
    context = "\n".join([_format_doc(doc) for doc in docs])
    result = chain.apply([{"question": question, "context": context}])[0]

    clogger = ConversationLogger(summarize_title(question))
    clogger.log_prompt(prompt_template.format(context=context, question=question))
    clogger.log_response(result['text'])
    clogger.log_metadata(result)
    return result['text']


def _format_doc(doc: Document):
    language = ""
    if doc.metadata['source'].endswith(".py"):
        language = "python"
    elif doc.metadata['source'].endswith(".ts"):
        language = "typescript"
    elif doc.metadata['source'].endswith(".tsx"):
        language = "tsx"
    elif doc.metadata['source'].endswith(".js"):
        language = "javascript"
    elif doc.metadata['source'].endswith(".jsx"):
        language = "jsx"
    return f"""```{language}\n#{doc.metadata['source']}\n{doc.page_content}\n```"""
