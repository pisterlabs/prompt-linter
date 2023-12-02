# load the necessary modules
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# ~ add new functionality!
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


from dotenv import load_dotenv #hide your API keys! add .env to .gitignore
load_dotenv()

# load the documents from a google drive folder
loader = GoogleDriveLoader(
    token_path = './token.json', #leave this alone
    folder_id = '1-sAD-TSBstS1IjbGPBXBqw2rCJpJcsnN', #copy from gdrive url
    recursive = False, # recursive=True will load all subfolders
    file_types = ["sheet", "document", "pdf"] # "document" means "google doc", not .docx or .doc
)
docs = loader.load()

# The beauty of Langchain.. all of the heavy lifting is done!
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 4000, chunk_overlap = 0, separators = [" ", ",", "\n"]) #chunk the texts
texts = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings) # vectorize the chunks

# ~ new functionality!
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) # Give it a memory
llm = ChatOpenAI(temperature=0.25, model_name="gpt-4-0613") # the "brain"
qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever(), memory=memory) # the behavior

# use custom prompt: https://python.langchain.com/docs/modules/chains/additional/question_answering.html#the-stuff-chain

# run the QA model
while True:
    query = input("Ask a question >>> ") # this is what the user sees in the terminal, on the line where they ask their question
    if query.lower() == "exit":
        exit() # allow user to exit the script by typing "exit"
    answer = qa.run(query)
    print(answer)
    