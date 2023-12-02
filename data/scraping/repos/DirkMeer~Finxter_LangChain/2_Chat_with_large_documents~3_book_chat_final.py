import pinecone
from decouple import config
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone


openai_key = config("OPENAI_API_KEY")
davinci_api = OpenAI(
    temperature=0, openai_api_key=openai_key, model_name="text-davinci-003"
)
embeddings_api = OpenAIEmbeddings(openai_api_key=openai_key)


pinecone.init(api_key=config("PINECONE_API_KEY"), environment="gcp-starter")
pinecone_index = pinecone.Index("langchain-vector-store")
vector_store = Pinecone(pinecone_index, embeddings_api, "text")


qa_chain = load_qa_chain(davinci_api, chain_type="stuff")


def ask_question_to_book(question: str, verbose=False) -> str:
    matching_pages = vector_store.similarity_search(question, k=5)
    if verbose:
        print(f"Matching Documents:\n{matching_pages}\n")
    result = qa_chain.run(input_documents=matching_pages, question=question)
    print(result, "\n")
    return result


ask_question_to_book("What is the fastest way to get rich?", verbose=True)
ask_question_to_book("What is the problem with most people?")
ask_question_to_book("What is the best way to peel bananas?")
