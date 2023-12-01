import os
import cohere
from dotenv import load_dotenv
import text_extractors
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
import pinecone


def intake_pdf(filename):
    ##### Import environment variables
    load_dotenv()
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    ##### Extract text from document
    text = text_extractors.read_pdf(filename)
    # print(text)

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=250,
        length_function=len,
        add_start_index=True,
    )

    texts = text_splitter.create_documents([text])
    print(texts[0].page_content)
    raw_texts = []

    for text in texts:
        raw_texts.append(text.page_content)

    ##### Get embeddings from Cohere
    co = cohere.Client(COHERE_API_KEY)
    embeds = co.embed(raw_texts, model="small").embeddings

    shape = np.array(embeds).shape
    print(shape)

    ##### Upload embeddings to Pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")
    index_name = "lamsu2"

    # if the index does not exist, we create it
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=shape[1], metric="cosine")

    # connect to index
    index = pinecone.Index(index_name)

    batch_size = 128

    ids = [str(i) for i in range(shape[0])]
    # create list of metadata dictionaries
    meta = [{"text": text} for text in raw_texts]

    # create list of (id, vector, metadata) tuples to be upserted
    to_upsert = list(zip(ids, embeds, meta))

    for i in range(0, shape[0], batch_size):
        i_end = min(i + batch_size, shape[0])
        index.upsert(vectors=to_upsert[i:i_end])

    # let's view the index statistics
    print(index.describe_index_stats())


def clear_index():
    ##### Import environment variables
    load_dotenv()
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    ##### Upload embeddings to Pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")
    index_name = "lamsu2"

    pinecone.delete_index("lamsu2")


# clear_index()
