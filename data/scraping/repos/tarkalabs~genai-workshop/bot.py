from langchain import PromptTemplate
from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from env import QDRANT_URL, OPENAI_API_KEY

template = """
You are a chatbot designed to find answers to the given question using the provided context. 
The context contains parts of a long document along with question asked by the human.
Respond to the questions politely and in the same language of the question.

Context: {context}

Question: {question}
"""

class Bot:
  def ask(question):
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    client = QdrantClient(url=QDRANT_URL)
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    # Fetch the related content from vector store
    vector_store = Qdrant(client=client, collection_name="insurance", embeddings=embeddings)
    docs = vector_store.similarity_search(query=question,k=5)

    # Create the langchain
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt, verbose=True)
    result = chain({"input_documents": docs, "question":question}, return_only_outputs=True)
    return result['output_text']