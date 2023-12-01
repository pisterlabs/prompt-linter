# obe stvari treba ugradit i u postojece appove


# MMR similarity clustering for Pinecone
import nltk
from langchain.embeddings import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

embeddings = Embeddings()
index = Pinecone.from_index("my_index")
query_vector = embeddings.encode("find relevant documents")
results = index.maximal_marginal_relevance_search(
    query_vector, top_k=10, fetch_k=50)


# context aware splitting addition to recursive characteh text splitter


embeddings = Embeddings()


def similarity(text1, text2):
    embed1 = embeddings.embed_text(text1)
    embed2 = embeddings.embed_text(text2)
    return embed1.similarity(embed2)


def split_into_sentences(text):
    return nltk.sent_tokenize(text)


def combine_sentences_into_chunk(sentences, chunk_size):
    combined = []
    curr_size = 0
    for sentence in sentences:
        if curr_size + len(sentence) > chunk_size:
            yield " ".join(combined)
            combined = []
            curr_size = 0
            combined.append(sentence)
            curr_size += len(sentence)
    yield " ".join(combined)


text_splitter = RecursiveCharacterTextSplitter(length_function=len, chunk_size=100, chunk_overlap=20,
                                               split_text_function=split_into_sentences,
                                               combine_chunks_function=combine_sentences_into_chunk,
                                               chunk_combine_similarity_function=similarity)
texts = ""  # ovde stavimo svoje tekstove
documents = text_splitter.split_documents(texts)


