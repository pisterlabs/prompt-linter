from dotenv import load_dotenv
from langchain.chains import RetrievalQA, RetrievalQAWithSourcesChain
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma

from experiments.api_models.models import InformationSource, CoreModelAnswer

# Settings for locally hosted OpenAI-compatible models
# os.environ["OPENAI_API_KEY"] = "KEY_DOESNT_MATTER_FOR_LOCALHOST"
# os.environ["OPENAI_API_BASE"] = "http://localhost:8080"

load_dotenv()


def mock_url_source_raw_content(url) -> str:
    with open(url, 'r') as file:
        return file.read()


def get_mapped_sources(sources: list[InformationSource]) -> list[InformationSource]:
    mapped_sources = []
    for source in sources:
        mocked_source = InformationSource(raw_content=mock_url_source_raw_content(source.url), url=source.url,
                                          title=source.title)
        mapped_sources.append(mocked_source)
    return mapped_sources


def get_text_chunks_langchain(text: str) -> list[Document]:
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs


def query_test(sources, query):
    documents = []
    for source in sources:
        documents.extend(TextLoader(source, encoding="utf-8").load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    llm = OpenAI(temperature=0, model_name='text-davinci-003')

    embeddings = OpenAIEmbeddings()
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


def query_test_advanced(sources: list[InformationSource], query):
    documents = []
    mapped_sources = get_mapped_sources(sources)
    for source in mapped_sources:
        docs = get_text_chunks_langchain(source.raw_content)
        docs = list(map(lambda d: Document(page_content=d.page_content, metadata={"source": source.title}), docs))
        documents.extend(docs)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    llm = OpenAI(temperature=1.5, model_name='text-davinci-003')

    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)

    prompt_template = """Answer in one sentence. If you do not know the answer, give a disclaimer to the user and then provide a possible answer

    {context}

    Question: {question}
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=docsearch.as_retriever(),
                                                     return_source_documents=True)

    result = qa({"question": query})

    return CoreModelAnswer(question=query, answer=result['answer'], sources=result['sources'])


if __name__ == "__main__":
    query = "Who is Steve?"

    # Old basic version that received only text files
    # sources = ["data/file1.txt", "data/file2.txt", "data/state_of_the_union_full.txt"]
    # query_test(sources, query)

    sources = [
        InformationSource(url="data/file1.txt", title="File about Steve's location"),
        InformationSource(url="data/file2.txt", title="File about Steve's wife"),
        InformationSource(url="data/state_of_the_union_short.txt", title="State of the union"),
    ]

    result = query_test_advanced(sources, query)
    print(f"Q: {result.question}")
    print(f"A: {result.answer}")
    print(f"Sources: {result.sources}")
