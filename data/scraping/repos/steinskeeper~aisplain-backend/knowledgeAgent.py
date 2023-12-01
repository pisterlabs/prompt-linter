from pymongo import MongoClient
from langchain.document_loaders import SeleniumURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.indexes import VectorstoreIndexCreator
client = MongoClient("mongodb://localhost:27017/")
db = client["aisplain"]


def knowledgeTalk(ask, docsUrl):
    print("Knowledge Agent invoked")
    loader = SeleniumURLLoader(urls=docsUrl)
    data = loader.load()
    print(data)
    llm = OpenAI()
    embedding = OpenAIEmbeddings(disallowed_special=())
    index = VectorstoreIndexCreator(embedding=embedding).from_loaders([loader])
    prompt = """
        Question: {question}
        Rules for answering: 
            2. Answer the question as best you can. If you are unable to answer, say 'I am unsure, I need human assistance' .
            3. You must answer in one paragraph. Do not use formatting.
            4. Your paragraph must not have more than 70 words.
            5. You must analyze all the files in the project.
        """
    response = index.query(ask, llm)
    print(response)
    return response

    
