from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms.llamacpp import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma


def query_test(sources, query):
    documents = []
    for source in sources:
        documents.extend(TextLoader(source, encoding="utf-8").load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    model_path = "../models/WizardLM-7B-uncensored.gguf.q2_K.bin"
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=1,
        n_batch=512,
        n_ctx=2048,
        f16_kv=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=True,
    )

    # TODO: serialize the embeddings and/or docsearch to disk, consider pickle
    embeddings = LlamaCppEmbeddings(model_path=model_path)
    docsearch = Chroma.from_documents(texts, embeddings)

    prompt_template = """Answer in one sentence. If you do not know the answer, give a disclaimer to the user and then provide a possible answer
    
    {context}
    
    Question: {question}
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(),
                                     chain_type_kwargs=chain_type_kwargs, return_source_documents=True)

    result = qa({"query": query})

    print(f"Q: {query}")
    print(result['result'])
    print(list(map(lambda doc: doc.metadata, result['source_documents'])))

    return result


if __name__ == "__main__":
    query = "Where does Olga live?"
    sources = ["../data/file1.txt", "../data/file2.txt"]

    query_test(sources, query)
