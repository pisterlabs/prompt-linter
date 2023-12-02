import click
import langchain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from ml_playground.langchain.utils.db import from_documents


@click.command()
@click.option("--model", default="gpt-3.5-turbo", help="OpenAI model.")
@click.option("-ct", "--chain-type", default="stuff", help="Langchain chain type")
@click.option("-k", "--k", default=8, help="Retriever parameter k")
@click.option(
    "--memory", default=False, is_flag=True, help="Enable memory in conversation."
)
@click.option("--verbose", default=False, is_flag=True, help="Verbose for debug")
@click.option("--debug", default=False, is_flag=True, help="Debug mode")
def main(model: str, chain_type: str, k: int, memory: bool, verbose: bool, debug: bool):
    langchain.debug = debug
    embeddings = OpenAIEmbeddings()

    vectordb = from_documents(
        [],
        embeddings,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    llm = ChatOpenAI(model_name=model, temperature=0)

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    if memory:
        _memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            chain_type=chain_type,
            memory=_memory,
            verbose=verbose,
        )
    else:
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            chain_type=chain_type,
            verbose=verbose,
        )

    while True:
        query = input("Q: ")
        if query == "exit":
            print("good bye")
            break
        if memory:
            result = qa_chain({"question": query})
            print(result["answer"])
        else:
            result = qa_chain({"query": query})
            print(result["result"])
            print(result["source_documents"][0])
        print()


if __name__ == "__main__":
    main()
