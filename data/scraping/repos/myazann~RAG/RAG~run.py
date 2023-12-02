import time
import os

from langchain.embeddings import HuggingFaceBgeEmbeddings, CacheBackedEmbeddings
from langchain.chains import ConversationalRetrievalChain, ConversationChain
from langchain.chains.question_answering import load_qa_chain
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_experimental.sql import SQLDatabaseChain
from langchain.memory import ConversationSummaryMemory, ConversationBufferWindowMemory
import huggingface_hub

from RAG.chatbots import choose_bot
from RAG.utils import get_args, get_device, get_NoOpChain
from RAG.loader import FileLoader
from RAG.retriever import Retriever
from RAG.prompter import Prompter
from RAG.output_formatter import csv_output_formatter

huggingface_hub.login(new_session=False)
args = get_args()
file_name = args.document
device = get_device()
chatbot = choose_bot()
file_loader = FileLoader()
file = file_loader.load(file_name)
file_type = file_loader.get_file_type(file_name)
prompter = Prompter()
if chatbot.q_bit is None:
  test_name = f"QA_{chatbot.name}_{time.time()}"
else:
  test_name = f"QA_{chatbot.name}_{chatbot.q_bit}-bit_{time.time()}"
os.environ["LANGCHAIN_PROJECT"] = test_name
if file_type == "db":
  db_chain = SQLDatabaseChain.from_llm(chatbot.pipe, file, verbose=True)
elif file_type == "csv":
  df = file
  csv_prompt = chatbot.prompt_chatbot(prompter.csv_prompt())
  CSV_PROMPT = PromptTemplate(input_variables=["chat_history", "user_input"], template=csv_prompt)
  csv_chain = ConversationChain(llm=chatbot.pipe, input_key="user_input", 
                                memory=ConversationBufferWindowMemory(k=3, memory_key="chat_history"), prompt=CSV_PROMPT)
else:
  doc = file_loader.LESSEN_preprocess(file)
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
  texts = text_splitter.split_documents(doc)
  model_name = "BAAI/bge-base-en"
  model_kwargs = {"device": device}
  encode_kwargs = {"normalize_embeddings": True}
  embeddings = HuggingFaceBgeEmbeddings(
      model_name=model_name,
      model_kwargs=model_kwargs,
      encode_kwargs=encode_kwargs
  )
  fs = LocalFileStore("./cache/")
  cached_embedder = CacheBackedEmbeddings.from_bytes_store(
      embeddings, fs, namespace=embeddings.model_name
  )
  # embeddings = HuggingFaceEmbeddings()
  db = Chroma.from_documents(texts, cached_embedder)
  retriever = Retriever(db)
  k = retriever.find_max_k(chatbot, [page.page_content for page in texts])
  retriever.init_base_retriever(k=k)
  retriever.add_embed_filter(embeddings, similarity_threshold=0.2)
  retriever.init_comp_retriever()
  qa_prompt = chatbot.prompt_chatbot(prompter.qa_prompt())
  memory_prompt = chatbot.prompt_chatbot(prompter.memory_summary())
  QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "chat_history", "question"], template=qa_prompt)
  MEMORY_PROMPT = PromptTemplate(input_variables=["summary", "new_lines"], template=memory_prompt)
  memory = ConversationSummaryMemory(llm=chatbot.pipe, memory_key="chat_history", return_messages=False, prompt=MEMORY_PROMPT,
                                     input_key="question", output_key="answer")
  doc_chain = load_qa_chain(
      chatbot.pipe,
      chain_type="stuff",
      **{"prompt": QA_CHAIN_PROMPT},
  )
  qa = ConversationalRetrievalChain(retriever=retriever.comp_retriever, combine_docs_chain=doc_chain, 
                                    question_generator=get_NoOpChain(chatbot.pipe), memory=memory, get_chat_history=lambda h: h,
                                    return_source_documents=True)
pretty_doc_name = " ".join(file_name.split(".")[:-1]).replace("_"," ")
print(f"""\nHello, I am here to inform you about the {pretty_doc_name}. What do you want to learn? (Press 0 if you want to quit!) \n""")
while True:
  print("User: ")
  query = input().strip()
  if query != "0":
    start_time = time.time()
    if file_type == "db":
      answer = db_chain.run(query)
    elif file_type == "csv":
      col_info = df.dtypes.to_string()
      first_rows = df.head(5).to_string()
      query = f"Column names and datatypes:\n{col_info}\nFirst 5 rows of the dataframe:\n{first_rows}\nUser Input: {query}"
      answer = csv_chain.predict(user_input=query).strip()
      code = csv_output_formatter(answer)
      try:
        exec(code)
        answer = ""
      except Exception as e:
        print(f"Got an error for the chatbot generated code:\n {code}")
        answer = e
    else:
      result = qa({"question": query})
      source_docs = result["source_documents"]
      answer = result["answer"].strip()
    print("\nChatbot:")
    print(f"{answer}\n")
    end_time = time.time()
    print(f"Took {end_time - start_time} secs!\n")
  else:
    print("Bye!")
    break