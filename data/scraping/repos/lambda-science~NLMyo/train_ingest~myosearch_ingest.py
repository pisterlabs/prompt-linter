# %%
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import glob
from langchain.embeddings import HuggingFaceInstructEmbeddings

# Todo: metadata diagnosis, filter
persist_directory = "../db_myocon"
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=50,
    length_function=len,
)
myo_report = []
texts = []
metadatas = []
sentences_to_embed = []

list_files = glob.glob("../data/processed/*.txt")
for file in list_files:
    with open(file) as f:
        document_text = f.read()
        myo_report.append(document_text)
        texts.append(text_splitter.split_text(document_text))

for index, text in enumerate(texts):
    for sentence in text:
        source = list_files[index]
        sentences_to_embed.append(sentence)
        metadatas.append({"source": source})
# %%
# embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceInstructEmbeddings(
    query_instruction="Represent the medicine document for retrieval: "
)
docsearch = Chroma.from_texts(
    sentences_to_embed,
    embeddings,
    persist_directory=persist_directory,
    metadatas=metadatas,
)
docsearch.persist()
docsearch = None

# %%
