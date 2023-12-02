from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.chat_models import ChatOpenAI
from langchain.evaluation.qa import QAGenerateChain, QAEvalChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from ml_playground.langchain.utils.db import from_documents


def get_qa_chain(model="gpt-3.5-turbo", verbose=False):
    embeddings = OpenAIEmbeddings()

    vectordb = from_documents(
        [],
        embeddings,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 8})

    llm = ChatOpenAI(model_name=model, temperature=0)

    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        chain_type="stuff",
        verbose=verbose,
    )
    return qa_chain


def get_examples():
    loader = DirectoryLoader(
        "./langchain/data/kolena/docs",
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True,
    )
    documents = loader.load()

    example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI())

    examples = [
        {
            "query": "Is Kolena client compatible with Pydantic v2?",
            "answer": "No",
        },
        {
            "query": "What's the license of Kolena client codebase?",
            "answer": "Apache License 2.0",
        },
        {
            "query": "What are the three core data types that need to be defined when building a workflow?",
            "answer": "The three core data types that need to be defined when building a workflow are Test Sample Type, Ground Truth Type, and Inference Type.",
        },
    ]
    new_examples = example_gen_chain.apply_and_parse(
        [{"doc": t} for t in documents[:3]]
    )
    examples += new_examples
    return examples


def get_eval_chain():
    llm = ChatOpenAI(temperature=0)
    eval_chain = QAEvalChain.from_llm(llm)
    return eval_chain


def main():
    examples = get_examples()
    qa_chain = get_qa_chain()
    eval_chain = get_eval_chain()

    predictions = qa_chain.apply(examples)
    graded_outputs = eval_chain.evaluate(examples, predictions)
    for i, eg in enumerate(examples):
        print(f"Example {i}:")
        print("Question: " + predictions[i]["query"])
        print("Real Answer: " + predictions[i]["answer"])
        print("Predicted Answer: " + predictions[i]["result"])
        print("Predicted Grade: " + graded_outputs[i]["text"])
        print()


if __name__ == "__main__":
    main()
