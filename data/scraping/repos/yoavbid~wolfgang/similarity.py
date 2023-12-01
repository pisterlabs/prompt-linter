from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader


def generate_similarity_index(data_path, index_path):
    data_files_list= list((data_path).glob("**/*.srt"))
    data_files_list.extend(list((data_path).glob("**/*.txt")))

    documents = []
    for data_file in data_files_list:
        loader = TextLoader(data_file)
        documents.extend(loader.load())
        
    textSplitter = CharacterTextSplitter(chunk_size=2000, separator="\n")

    docs = textSplitter.split_documents(documents)

    store = FAISS.from_documents(docs, OpenAIEmbeddings())
        
    store.save_local(index_path)
    
def load_similarity_index(index_path):
    store = FAISS.load_local(index_path, OpenAIEmbeddings())
        
    return store


def test_similarity_index(text, index_path):
    store = load_similarity_index(index_path)

    docs = store.similarity_search(text, k=3) #similarity_search(text)
    for i, doc in enumerate(docs):
        print('%d. ' % i, doc.page_content, '\n\n')
        
        
def main():
    from simply_tutor.src import simply_tutor_path
    generate_similarity_index(data_path=simply_tutor_path / "training_data/",
                              index_path=simply_tutor_path / "faiss_index")
    
    test_similarity_index('lean on me', index_path=simply_tutor_path / "faiss_index")
    
        
if __name__ == "__main__":
    main()