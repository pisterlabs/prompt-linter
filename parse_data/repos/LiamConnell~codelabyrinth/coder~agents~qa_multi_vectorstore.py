from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI

from coder.agents.qa_with_vectorstore import _format_doc
from coder.utils import ConversationLogger, summarize_title
from coder.vectorstore import VectorStore


prompt_template = "Context:\n{context}\n\n Question: {question}"
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

llm = ChatOpenAI(model_name="gpt-4", temperature=0)
chain = LLMChain(llm=llm, prompt=PROMPT)


def agent(question: str, vectorstore_collections: list[str]):
    v = VectorStore()
    vectorstore_collections = vectorstore_collections
    docs = []
    for collection in vectorstore_collections:
        docs.extend(v.similarity_search(collection, question, k=4))
    context = "\n".join([_format_doc(doc) for doc in docs])
    result = chain.apply([{"question": question, "context": context}])[0]

    clogger = ConversationLogger(summarize_title(question))
    clogger.log_prompt(prompt_template.format(context=context, question=question))
    clogger.log_response(result['text'])
    clogger.log_metadata(result)
    return result['text']