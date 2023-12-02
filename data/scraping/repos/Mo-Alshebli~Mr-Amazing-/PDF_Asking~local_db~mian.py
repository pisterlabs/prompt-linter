import os
# import gpt_2_simple as gpt2

# os.environ["OPENAI_API_KEY"] = "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"
from dotenv import load_dotenv
load_dotenv()
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

persist_directory = 'local_db'
embedding = OpenAIEmbeddings()

vectordb2 = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding,
                   )

retriever = vectordb2.as_retriever(search_kwargs={"k": 2})

# Set up the turbo LLM
turbo_llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo'
)


# create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm,
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)

## Cite sources
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])



while True:
    # full example
    query = input(" Enter : ")
    llm_response = qa_chain(query)
    print(process_llm_response(llm_response))
