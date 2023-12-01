import dotenv

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

dotenv.load_dotenv()

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    verbose=True,
)


# logging.basicConfig()
# logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

# retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(), llm=llm)


template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Use Chinese to answer the question.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


while True:
    # Prompt the user for the PDF URL
    url = input("Enter the URL (or 'q' to quit): ")
    if url.lower() == "q":
        break

    loader = WebBaseLoader(url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(
        documents=all_splits, embedding=OpenAIEmbeddings()
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    )

    while True:
        # Prompt the user for a question
        question = input("Ask a question (or 'q' to change web source): ")
        if question.lower() == "q":
            break

        # Perform question-answering
        answer = qa_chain({"query": question})

        # Display the answer
        print("Answer:", answer)
