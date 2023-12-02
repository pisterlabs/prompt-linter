from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from pretty_print_callback_handler import PrettyPrintCallbackHandler


from dotenv import load_dotenv

load_dotenv()

raw_documents = TextLoader("./transcripts/day0/Boca_AI_-_Josh_EVT_LLM_talk.txt").load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
print(len(documents))

embeddings_model = OpenAIEmbeddings()
# embeddings_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_documents(documents, embeddings_model)

query = "Who were the people involved in this conversations ? Return the result as json and use the field firstname and lastname"
# docs = db.similarity_search(query)
# print(len(docs))
# print(docs[0].page_content)


from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

my_callback = PrettyPrintCallbackHandler()

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
llm.callbacks = [my_callback]

qa_chain = RetrievalQA.from_chain_type(
    llm, retriever=db.as_retriever(), return_source_documents=True
)
answer = qa_chain({"query": query})

print(answer)
