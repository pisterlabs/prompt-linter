

def load_vector_memory_from_dir(dir_path):
    from langchain.document_loaders import DirectoryLoader
    loader = DirectoryLoader(dir_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return FAISS.from_documents(texts, OpenAIEmbeddings())

def get_answer_from_vector_memory(vector_memory, query):
    from langchain.agents.agent_toolkits import (
        create_vectorstore_agent,
        VectorStoreToolkit,
        VectorStoreInfo,
    )
    vectorstore_info = VectorStoreInfo(
        name="software_requirement_specification",
        description="software requirement specification and other things you want to know",
        vectorstore=vector_memory
    )
    toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)
    agent_executor = create_vectorstore_agent(
        llm=ChatOpenAI(temperature=0),
        toolkit=toolkit,
        verbose=True
    )
    answer = agent_executor.run(query)
    return answer

def get_answer_from_vector_memory_and_web(text):
    pass

if __name__ == "__main__":
    vector_store = load_vector_memory_from_dir("../../docs")
    get_answer_from_vector_memory(vector_store, "What will be changed in the next version?")
