from langchain.embeddings import OpenAIEmbeddings   
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter

import pinecone
import time
import uuid

from config import OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENVIRONMENT, PINECONE_INDEX_NAME, EMBEDDING_MODEL, SPLITTER_CHUNK_SIZE, SPLITTER_CHUNK_OVERLAP, UPLOAD_BATCH_SIZE

# Paper list
PAPER_LIST = ["data/paper1.pdf", "data/paper2.pdf", "data/paper3.pdf"]


# Helper functions
def print_match(result):
    for match in result['matches']:
        print("="*60)
        print(f"Score: {match['score']:.2f} \t Source: {match['metadata']['source']} \t Page: {int(match['metadata']['page'])}")
        print("="*60)
        print(f"{match['metadata']['text']}")
        print("="*60)
        print()

# Initialize OpenAI
embedding_model = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, 
    model=EMBEDDING_MODEL
)

print("="*30)
print("OpenAI initialization: OK")
print("="*30)
print()



# Initialize Pinecone vector storage
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT
)

if PINECONE_INDEX_NAME not in pinecone.list_indexes():
    # we create a new index if it doesn't exist
    pinecone.create_index(
        name=PINECONE_INDEX_NAME,
        metric='cosine',
        dimension=1536  # 1536 dim of text-embedding-ada-002
    )
    # wait for index to be initialized
    time.sleep(1)

pinecone_index = pinecone.Index(PINECONE_INDEX_NAME)
pinecone_stats = pinecone_index.describe_index_stats()
print("="*30)
print("Pinecone initialization: OK")
print(pinecone_stats)
print("="*30)
print()


for file_path in PAPER_LIST:
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    print(f"Processing [{file_path}]")
    print(f"Pages shape: {len(pages)}")

    text_splitter = TokenTextSplitter(
        chunk_size=SPLITTER_CHUNK_SIZE, 
        chunk_overlap=SPLITTER_CHUNK_OVERLAP
    )

    source = pages[0].metadata["source"]

    total_sentences = []
    page_number_list = []
    for idx, page in enumerate(pages):
        page_num = page.metadata["page"] + 1
        sentences = text_splitter.split_text(page.page_content)
        total_sentences += sentences
        page_number_list += [page_num] * len(sentences)

    # Due to OpenAPI rate limitation, I have to embed multiple chunks at the same time
    paper_embedding = embedding_model.embed_documents(total_sentences)

    # Reformat the vectors
    to_upsert = []
    for i, sentence_vector in enumerate(paper_embedding):
        to_upsert.append({
            "id": str(uuid.uuid4()),
            "values": sentence_vector,
            "metadata": {
                            "text": total_sentences[i],
                            "source": source,
                            "page": page_number_list[i]
                        }
        })

    # Upload the vectors in baches
    batch_size = UPLOAD_BATCH_SIZE
    n = len(to_upsert)
    print(f"Total number: {n}")

    for i in range(0, n, batch_size):
        if i + batch_size <= n:
            batch = to_upsert[i: i+batch_size]     
        else:
            batch = to_upsert[i:]

        pinecone_index.upsert(vectors=batch)
        print(f"Uploaded batch [{i} : {min(n, i+batch_size)}]")



# Auto testing
query_list = [
    "How to treat patient with ACHD?",
    "How to diagnose Resistant Hypertension?",
    "How to reduce the cardiorenal risk?"
]

for i, query in enumerate(query_list):
    print("="*30)
    print(f"Test {i+1}: {query}")
    print("="*30)
    query_embedding = embedding_model.embed_documents([query])
    res = pinecone_index.query(query_embedding, top_k=3, include_metadata=True)
    print_match(res)

print("Upload papers - Done!")
