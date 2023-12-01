from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
import constants

"""
We're basically going convert the pdf to a text and then convert it to smaller chunks (a question - answer pair as a query)
and in the end, we're going to put these chunks into a json file.

- In order to do this, we're going to follow the below mentioned steps:
- Take the input of the pdf file and extract all the text from it
- Split the text into smaller chunks of queries as mentioned above
- Embed those queries into a vector space
"""


input_filename = r"C:\Users\itsgu\Downloads\apple.pdf"

elements = partition(filename=input_filename, strategy="auto")
chunks = chunk_by_title(elements, multipage_sections=True)
text = ""

for chunk in chunks:
    text += str(chunk)

result = text.split("---")

print("setting up embedding query...")
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
query_result = embed_model.embed_query("Problem With iPhone")
print(len(query_result))

print("Pinecone setup")
pinecone.init(
    api_key= constants.PINECONE_API_KEY,
    environment= constants.PINECONE_ENVIRONMENT
)

print("Creating index")
# to create a new index
index = pinecone.Index(constants.PINECONE_INDEX_NAME)
vectorstore = Pinecone(index, embed_model.embed_query, "text")

vectorstore.add_texts(result)




