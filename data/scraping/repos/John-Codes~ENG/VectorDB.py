
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# class VectorDB:
    # def __init__(self):
    #     pass

def split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def convert_text_to_embeddings(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def SaveVectorDB(string1, string2):
    # Split strings into text chunks
    chunks1 = split_text_into_chunks(string1)
    chunks2 = split_text_into_chunks(string2)
    #output to console chunks 1 and 2
    print("Chunks 1 and 2")
    print(chunks1)  
    print(chunks2)

    
    # Concatenate the chunks from both strings
    all_chunks = chunks1 + chunks2
    #print convert_text_to_embeddings(all_chunks)
    embeedings= convert_text_to_embeddings(all_chunks)
    # print(embeedings.aadd_texts)
    # Convert text chunks into embeddings and store them in a single vector store
    vectorstore = embeedings

    # Additional processing or operations

    
    return vectorstore

def GetVectorDB(input_string, vectorstore):
    # Convert the input string into embeddings
    embeddings = OpenAIEmbeddings()
    input_embedding = embeddings.embed(input_string)

    # Perform nearest neighbor search in the vector store
    num_neighbors = 5  # Adjust this as needed
    similar_vectors = vectorstore.retrieve_nearest_neighbors(input_embedding, num_neighbors)

    # Retrieve and process the similar vectors
    for similar_vector in similar_vectors:
        index = similar_vector["index"]
        distance = similar_vector["distance"]
        vector = vectorstore.get_vector(index)

        # Do something with the retrieved vector and metadata
        
    return similar_vectors