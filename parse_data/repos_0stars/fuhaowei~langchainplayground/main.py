from tqdm import tqdm
import time


from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate


def process_with_progress(chain, data, return_only_outputs=True):
    result = []
    for doc in tqdm(data["input_documents"]):  # add progress bar here
        single_doc_result = chain({"input_documents": [doc], "question": data["question"]}, return_only_outputs=return_only_outputs)
        result.append(single_doc_result)
    return result

def run(query):
    llm = OpenAI(batch_size=5,temperature=0)

    # loading documents
    loader = PyPDFLoader("hwsweresume2023.pdf")
    pages = loader.load_and_split()

    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
    docs = faiss_index.similarity_search(query, k=2)

    question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
                                {context}
                                Question: {question}
                                Relevant answers, if any:"""
    QUESTION_PROMPT = PromptTemplate(
        template=question_prompt_template, input_variables=["context", "question"]
    )

    combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer. 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.

    QUESTION: {question}
    =========
    {summaries}
    =========
    Answer:"""

    COMBINE_PROMPT = PromptTemplate(
        template=combine_prompt_template, input_variables=["summaries", "question"]
    )

    qa_document_chain = load_qa_chain(llm, chain_type="map_reduce", question_prompt=QUESTION_PROMPT, combine_prompt=COMBINE_PROMPT)
    result = process_with_progress(qa_document_chain, {"input_documents": docs, "question": query}, return_only_outputs=True)
    print(result)


if __name__ == "__main__":
    while True:
        query = input('Please provide your query (type "exit" to stop): ')
        if query.lower() == 'exit':
            break
        run(query)