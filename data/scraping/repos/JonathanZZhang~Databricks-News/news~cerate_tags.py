from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings,SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers.document_compressors import EmbeddingsFilter
import time
from langchain import HuggingFacePipeline
import torch
import accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM
from langchain.memory import ConversationBufferMemory,ConversationSummaryBufferMemory
from langchain.document_loaders import TextLoader
from .chat_preload import *
from langchain.chains import ConversationChain

class chatbot:
  def __init__(self,document_path) -> None:
    tag_create = news_tag()
    self.tags = tag_create.create_tag("test.txt")
    self.document_path = document_path
    self.load_document()
    self.get_model()
    self.chat_history = []


  def load_document(self):
    print("embedding document, may take a while...")
    loader = TextLoader(self.document_path)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2500,
        chunk_overlap  = 100,
        length_function = len,
    )
    split_document = text_splitter.split_documents(document)
    embeddings_1 = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    combined_vector_store = FAISS.from_documents(split_document, embeddings_1)
    self.retriever = combined_vector_store.as_retriever(search_kwargs=dict(k=3))

  def get_model(self):
    print("loading model, may take a while...")
    repo_id = "google/flan-t5-large" # this one is ok for news
    self.llm_chat = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0})
    self.memory = ConversationSummaryBufferMemory(llm =self.llm_chat, max_token_limit=500, memory_key="chat_history", return_messages=True)
  
  def news_chat(self, question):
    instruction = """
    You are a chatbot having a conversation with a human. Your are asked to chat with the user for any other follow up questions with the news.
    Given the following extracted parts of a long document and a question, answer the user question.
    If you don't know, say that you do not know.
    """
    Query_template = instruction + """
        =========
        context: {context}
        =========
        Chat History:{chat_history}
        =========
        Question: {question}
        =========
        """
    QA = PromptTemplate(template=Query_template, input_variables=["context", "chat_history", "question"])
    print("loading chain, this can take some time...")
    news_conversation = ConversationalRetrievalChain.from_llm(
      llm= self.llm_chat,
      retriever=self.retriever,
      memory = self.memory,
      # verbose=True,
      # return_source_documents=True,
      combine_docs_chain_kwargs={'prompt': QA})
    
    result = news_conversation({"question": question})
        # print(result["answer"])
    res_dict = {
    "answer": result["answer"],
    }
    
    if question=="quit" or question=="q":
      res_dict = {"answer": "Bye",}
    
    return res_dict["answer"]
  
  def topic_chat(self, question_topic):
    tag_instruction = """
    You are a chatbot having a conversation with a human. Your are asked to chat with the user for any other follow up questions with the given topics.
    Given the related tags and a question, answer the user question.
    If you don't know, say that you do not know.
    """
    tag_template = tag_instruction + """tags:""" + self.tags + """
            =========
            Chat History:{history}
            =========
            Question: {input}
            =========
            """
    tag_prompt = PromptTemplate(template=tag_template, input_variables=["history", "input"])
    print("loading chain, this can take some time...")
    # memory2 = ConversationSummaryBufferMemory(llm =llm_chat, max_token_limit=500, memory_key="history", return_messages=True)
    # readonlymemory2 = ReadOnlySharedMemory(memory=memory2)
    tags_conversation = ConversationChain(
          llm= self.llm_chat,
          prompt=tag_prompt,
          # retriever=retriever,
          memory = ConversationBufferMemory())
    
    result = tags_conversation({"input": question_topic, "history": self.chat_history})
        # print(result["answer"])
    res_dict = {
    "answer": result["response"],
    }
    self.chat_history.append((question_topic, result["response"]))
    if question_topic=="quit" or question_topic=="q":
      res_dict = {"answer": "Bye",}
    
    return res_dict["answer"]

if __name__=="__main__":
  chatbot = chatbot("test.txt")
  print(chatbot.news_chat("what is it targeting to"))
  print(chatbot.topic_chat("what is digital marketing"))





# # news
# chat_history = []
# while True:
#   question = input()
#   if question == "q":
#     break
#   start_time = time.time()
#   result = news_conversation({"question": question, "chat_history": chat_history})
#   end_time = time.time()
#   # chat_history.append((question, result["answer"]))
#   print(result["answer"])
#   print(f"Time taken to generate response: {end_time - start_time} seconds")



















# embeddings_filter = EmbeddingsFilter(embeddings= embeddings_1, similarity_threshold=0.76)

# chat_history = []






# repo_id = "databricks/dolly-v2-3b" # this one is ok for news
# llm_tag = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0})
# model_name = "databricks/dolly-v2-3b" # can use dolly-v2-3b, dolly-v2-7b or dolly-v2-12b for smaller model and faster inferences.
# instruct_pipeline = pipeline(model=model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto",
                              #  max_new_tokens=256, top_p=0.95, top_k=50)
# llm_tag = HuggingFacePipeline(pipeline=instruct_pipeline)



# embeddings_filter = EmbeddingsFilter(embeddings= embeddings_1, similarity_threshold=0.76)
# PromptTemplate.from_template(prompt_template)
# chat_history = []

      # verbose=True,
      # return_source_documents=True,
      # combine_docs_chain_kwargs={'prompt': QA})




#   # tag
# chat_history = []
# while True:
#   question = input()
#   if question == "q":
#     break
#   start_time = time.time()
  
#   end_time = time.time()
#   # chat_history.append((question, result["response"]))
#   # print(result["answer"])
#   print(result["response"])
#   # print(result)
#   print(f"Time taken to generate response: {end_time - start_time} seconds")