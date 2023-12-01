# for all images in the ./images folder, create a vector and store it in the Cassandra database
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import ImageCaptionLoader
from langchain.vectorstores.cassandra import Cassandra
from langchain.embeddings import OpenAIEmbeddings

from glob import glob
import sys
sys.path.append('./')
from utils import getCQLSession, getCQLKeyspace
from config import table_name

session = getCQLSession()
keyspace = getCQLKeyspace()

# extract all image files from the ./images folder
list_image_urls = glob('./images/*.jpg')
print(f'{len(list_image_urls)} images found.')

emb = OpenAIEmbeddings()
loader = ImageCaptionLoader(path_images=list_image_urls)

index_creator = VectorstoreIndexCreator(
    vectorstore_cls=Cassandra,
    embedding=emb,
    vectorstore_kwargs={
        'session': session,
        'keyspace': keyspace,
        'table_name': table_name,
    },
)

# load the image caption vectors into the database
index = index_creator.from_loaders([loader])
