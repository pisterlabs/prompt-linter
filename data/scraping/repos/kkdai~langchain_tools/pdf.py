from langchain.document_loaders import PyPDFLoader  # for loading the pdf
from langchain.embeddings import OpenAIEmbeddings  # for creating embeddings
from langchain.vectorstores import Chroma  # for the vectorization part
from langchain.chains import ChatVectorDBChain  # for chatting with the pdf
from langchain.llms import OpenAI  # the LLM model we'll use (CHatGPT)


pdf_path = "./paper.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
# print(pages[0].page_content)

# 2. Creating embeddings and Vectorization
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings,
                                 persist_directory=".")
vectordb.persist()


# 3. Querying
llm = OpenAI(temperature=0.9, model_name="gpt-3.5-turbo")
pdf_qa = ChatVectorDBChain.from_llm(
    llm, vectordb, return_source_documents=True)

query = "What is the bitcoin?"
result = pdf_qa({"question": query, "chat_history": ""})
print("Answer:")
print(result["answer"])

chat_history = [(query, result["answer"])]
query2 = "When it be found?"
result = pdf_qa({"question": query2, "chat_history": chat_history})
print("Answer2:")
print(result["answer"])
