import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader


class EmbedRepo:
  def __init__(self, root_dir="./", deeplake_username="", deeplake_db=""):
    self.root_dir = root_dir
    self.deeplake_username = deeplake_username
    self.deeplake_db = deeplake_db
    self.embeddings = OpenAIEmbeddings(disallowed_special=())
    self.db = DeepLake(
      dataset_path=f"hub://{self.deeplake_username}/{self.deeplake_db}",
      embedding_function=self.embeddings,
    )
  
  # Split the files into chunks
  def split(self):
    docs = []
    for dirpath, dirnames, filenames in os.walk(self.root_dir):
        for file in filenames:
            try:
                loader = TextLoader(os.path.join(dirpath, file), encoding="utf-8")
                docs.extend(loader.load_and_split())
            except Exception as e:
                pass
    from langchain.text_splitter import CharacterTextSplitter

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    return texts
  
  # Save the embeddings
  def save(self, texts):
    self.db.add_documents(texts)

# embed = EmbedRepo(...)
# texts = embed.split()
# embed.save(texts)
