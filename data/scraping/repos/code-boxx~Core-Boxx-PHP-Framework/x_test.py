# (A) LOAD SETTINGS & MODULES
import a_settings as set
import c_oto_rodo as oto
from langchain import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains import RetrievalQA

# (B) CHAIN
chain = RetrievalQA.from_chain_type(
  llm = oto.llm,
  retriever = Chroma(
    persist_directory = set.path_db,
    embedding_function = HuggingFaceInstructEmbeddings(**set.embed_args)
  ).as_retriever(),
  chain_type_kwargs = {
    "prompt": PromptTemplate (
      template = set.prompt_template,
      input_variables = ["question", "context"]
    )
  },
  ** set.chain_args
)

# (C) COMMAND LINE Q&A
while True:
  query = input("\nEnter a query: ")
  if query == "exit":
    break
  if query.strip() == "":
    continue

  # How many cakes can a person eat in a day?
  # What is John's favorite color?
  # What is John's dream?
  # What is John wearing?
  # How old is John?
  print(chain(query))