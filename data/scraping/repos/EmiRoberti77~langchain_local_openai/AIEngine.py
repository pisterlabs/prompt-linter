import constants as constants
from constants import console as out
from constants import ColorWrapper as CR
from HTTReturn import HTTPReturn
from PromptRequest import PrompRequest
import os

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import openai
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

class AIEngine():
  chatHistory = []
  chain = None
  vectorstore = None
  loader = None

  def __init__(self) -> None:
    os.environ['OPENAI_API_KEY'] = constants.apiKey()
    out(msg=constants.LINE_BREAK, color=CR.blue, reset=False)
    print('DATA_ROOT', constants.DOCUMENT_ROOT)
    print('PERSIST_MODE', constants.PERSIST)
    print('PERSIST_ROOT', constants.PERSIST_ROOT)
    out(msg=constants.LINE_BREAK, color=CR.blue, reset=True)


    if constants.PERSIST and os.path.exists(constants.PERSIST_ROOT):
      out(msg=constants.RESUME_INDEX, color=CR.cyan, reset=True)
      self.vectorstore = Chroma(persist_directory=constants.PERSIST_ROOT, embedding_function=OpenAIEmbeddings())
      index = VectorStoreIndexWrapper(vectorstore=self.vectorstore)
    else:
      self.loader = DirectoryLoader(constants.DOCUMENT_ROOT)
      if constants.PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":constants.PERSIST_ROOT}).from_loaders([self.loader])
      else:
        index = VectorstoreIndexCreator().from_loaders([self.loader])

    self.chain = ConversationalRetrievalChain.from_llm(
      llm=ChatOpenAI(model="gpt-3.5-turbo"),
      retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

  def prompt(self, req:PrompRequest) -> str:
    query = req.content
    http_response = HTTPReturn()
    try:
      result = self.chain({"question":query, "chat_history":self.chatHistory})
      answer = result['answer']
      out(msg=answer, color=CR.green, reset=True)
      return http_response.httpReturn(200, {"answer":answer})
    except Exception as e:
      out(msg=str(e), color=CR.red, reset=True)
      return http_response.httpReturn(200, {"error":str(e)})