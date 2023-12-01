from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma

from dotenv import load_dotenv

load_dotenv(".env.local")

loader = TextLoader("./resources/about_me.txt")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(separators=[".", "\n", "\n\n"])
texts = text_splitter.split_documents(documents=documents)

embeddings = OpenAIEmbeddings()
store = Chroma.from_documents(documents=texts, embedding=embeddings)
retriever = store.as_retriever(search_kwargs={"k": 1}) # will use top related item from doc

llm = OpenAI(temperature=0) # temperature 0 for select the highest probability option
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, verbose=False)


input_text = ""
while True:
  input_text = input("[YOU]:  ")
  if input_text.lower() == "quit":
    break
  output_text = qa_chain.run(input_text)
  print("[BOT]:", output_text)



# Sample I/O
# [YOU]: Tell my name
# [BOT]:  Your name is Rafiul Islam.
# [YOU]: Where I am currently working?
# [BOT]: LogiQbits Limited.
# [YOU]: Tell me about my first job
# [BOT]: You did your first job as an intern at Cefalo Bangladesh Limited.
# [YOU]: What about SolseTech?
# [BOT]: I worked at SolseTech Limited for one and a half years before starting my own business.
# [YOU]: Who worked at SolseTech?
# [BOT]: Rafiul Islam worked at SolseTech.
# [YOU]: quit