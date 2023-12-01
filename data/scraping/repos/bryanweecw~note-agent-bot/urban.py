# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
# from langchain.vectorstores import DeepLake
import os

# import getpass
# import openai

# os.environ['OPENAI_API_KEY'] = os.environ.get("OPENAI_API_KEY")

# embeddings = OpenAIEmbeddings()
# from langchain.document_loaders import TextLoader

# text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
# embeddings = OpenAIEmbeddings()

# db = DeepLake(dataset_path='./urban_deeplake', embedding_function=embeddings)

# # open pdf

# pdf = TextLoader('urbanreading.pdf')
# documents = pdf.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)
# db.add_documents(docs)


from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAIChat
from dotenv import load_dotenv

load_dotenv()
loader = PyPDFLoader("urbanreading.pdf")
pages = loader.load_and_split()


# db = FAISS.from_documents(pages, OpenAIEmbeddings())
# db.save_local("faiss_index")

# db = FAISS.load_local("faiss_index", OpenAIEmbeddings())
# docs = db.similarity_search("What are the features of sri lankan architecture?", k=8)
# for doc in docs:
#     print(str(doc.metadata["page"]) + ":", doc.page_content[:300])


def questionAnswer(input: str):
    db = FAISS.load_local("faiss_index", OpenAIEmbeddings())
    docs = db.similarity_search(input, k=4)
    prompt = "You are an urban studies professor who is teaching a class on Architecture and Society in Yale-NUS. You will be asked questions by a college undergraduate. You will be given the resources from the reading in question. Provide answers and guidance as needed.\n\n Resources:\n"
    resources = ""
    for doc in docs:
        resources += (
            "From page " + str(doc.metadata["page"]) + ":" + doc.page_content + "\n"
        )

    question = (
        "\n Please provide a comprehensive answer to the following question based on the resources above: "
        + input
    )

    prompt += resources + question

    llm = OpenAIChat()
    answer = llm.predict(prompt)
    print(answer)
    return answer
