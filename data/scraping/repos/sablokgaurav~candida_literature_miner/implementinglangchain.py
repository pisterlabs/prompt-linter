from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import LlamaCpp
from langchain.vectorstores import Qdrant
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
split_my_text = CharacterTextSplitter(
separator="\n", chunk_size=500, chunk_overlap=100, length_function=len)
splitted_chunks = text_splitter.split_text(split_my_text)
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
model_base_explain = Qdrant.from_texts(splitted_chunks,embeddings, location=":memory:",collection_name="explain_chunks")


# you can take the literatureMiner.py and clean the tags and prepare the literature \
                   #  if you want to put this on the other text but it will need modifications. 
