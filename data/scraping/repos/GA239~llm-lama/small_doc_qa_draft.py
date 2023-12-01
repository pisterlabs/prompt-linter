import os

from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import NLTKTextSplitter
from langchain.vectorstores import Chroma

from runner import get_cpp_lama, get_local_model_path

URL = "https://luminousmen.com/post/github-pull-request-templates"
m_name = "Llama-2-13B-GGML"
model_file = "llama-2-13b.ggmlv3.q4_0.bin"
DOC_NUM = 3


def get_data(url=None):
    data_url = url or URL

    # TODO: add a loader for local files (exported from KB)
    loader = WebBaseLoader(data_url)

    return loader.load()


def split_text(data):

    # RecursiveCharacterTextSplitter does not create overlaps. I don't know why.
    text_splitter = NLTKTextSplitter(chunk_size=1500, chunk_overlap=1400)
    all_splits = text_splitter.split_documents(data)
    return all_splits


def vectorize_text(splits):
    # TODO: test other vector stores
    vectorstore = Chroma.from_documents(
        documents=splits,
        # TODO: test other embedding models
        embedding=GPT4AllEmbeddings())
    return vectorstore


def test_vector_store(vectorstore: Chroma):
    question = "What are the Steps to Create an Effective Template?"
    docs = vectorstore.similarity_search_with_score(question, k=DOC_NUM)
    for doc, score in docs:
        print(score, " " * 10, doc)


def get_model():
    model_path = os.path.join(get_local_model_path(m_name), model_file)
    return get_cpp_lama(m_name, model_path, lang_chain=True)


def create_chain(model):
    """
    Run an LLMChain with either model by passing in the retrieved docs and a simple prompt.

    It formats the prompt template using the input key values provided and passes the formatted string
    to LLama-V2, or another specified LLM.

    We can use a QA chain to handle our question above.
    chain_type="stuff" means that all the docs will be added (stuffed) into a prompt.
    """
    # Prompt
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Answer in one sentence as short as possible.
    
    Context delimited by triple backticks:
    ```{context}```

    Question delimited by triple backticks: ```{question}```
    Answer: """

    qa_chain_prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=template,
    )
    # Chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=qa_chain_prompt, verbose=True)
    # todo: debug. "stuff" just adds all documents in the prompt. Test more smart approaches.
    # chain = load_qa_chain(model, chain_type="map_reduce", question_prompt=qa_chain_prompt)
    return chain


def run_chain(chain, vectorstore, question):
    print("get relevant docs")
    docs = vectorstore.similarity_search(question, k=DOC_NUM)
    print("::DEBUG::")
    print(len(docs))
    for doc in docs:
        print(doc)
    print("::DEBUG::")

    print("ask question")
    return chain(
        {
            "input_documents": docs,
            "question": question
        },
        return_only_outputs=True
    )


def main():
    print("get data..")
    dt = get_data()
    sp_dt = split_text(dt)

    print("vectorisation")
    vs = vectorize_text(sp_dt)
    # test_vector_store(vs)

    print("create chain")
    ch = create_chain(get_model())

    print("run chain")
    # repl = run_chain(ch, vs, "Name the Steps to Create an Effective Template.")
    repl = run_chain(ch, vs, "What is a Pull Request Template?")
    print(repl)


if __name__ == "__main__":
    main()
