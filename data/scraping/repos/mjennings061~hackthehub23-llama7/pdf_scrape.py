import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

from langchain.document_loaders import PyPDFLoader

def query_pinecone_chain(query):

    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_ENV  # next to api key in console
    )
    index_name = "highway-code" # put in the name of your pinecone index here

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # if index does not exist, create it
    docsearch = Pinecone.from_existing_index(index_name, embeddings)

    from langchain.llms import OpenAI
    from langchain.chains.question_answering import load_qa_chain
    llm = OpenAI(temperature=1, openai_api_key=OPENAI_API_KEY)
    from langchain.prompts import PromptTemplate

    template = """
        You are assessing an automotive car accident claim. Please provide guidance on who, if anyone, is at fault. The context will include relevant parts of the highway code.
        Context: {context}
        Question: {question}
        If the answer is not in the context, DO NOT MAKE UP AN ANSWER.
        However, in this case, if there are any relevant answers you can find, please state these. 

    """
    prompt = PromptTemplate.from_template(
            template
        )

    docs = docsearch.similarity_search(query)
    all_context = ('').join([f.page_content for f in docs])
    final_prompt = prompt.format(context=all_context ,question=query)
    answer = llm(final_prompt)
    # print("answer: ", answer)
    # print("docs: ",docs)
    return answer, docs

if __name__ == '__main__':

    query = '''
    Based on the information provided, here is the order of events in the car accident:

    The user was driving on the main road an hour ago.
    Another vehicle pulled out from the right side of the user near a set of traffic lights.
    The user was driving at 30mph and was worried about missing their appointment.
    There were no injuries reported.
    The user's car sustained damage on the right side and the wheel is making a strange noise.
    The user's car is not drivable.

    '''
    query_pinecone_chain(query)