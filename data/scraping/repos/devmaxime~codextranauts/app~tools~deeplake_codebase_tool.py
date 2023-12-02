from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DeepLake
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import Tool

def search_deeplake_codebase(question):
    """
    Search in the DeepLake codebase for answers to your questions.

    Parameters:
        question (str): The question to ask.

    Returns:
        str: The answer to your question.    
    """
    
    # Define the retriever
    retriever = DeepLake(
        dataset_path="hub://devmaxime/langchain-code",
        read_only=True,
        embedding_function=OpenAIEmbeddings(),
    ).as_retriever()

    # Define the chain
    chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo"), retriever=retriever)

    return chain({"question": question, "chat_history": []})["answer"]

def get_deeplake_codebase_tool():
    """
    Create a custom tool based on the search_deeplake_codebase function.
    
    Returns:
        Tool: The custom tool.
    """
    search_codebase_tool = Tool.from_function(
        func=search_deeplake_codebase,
        name="Search Codebase",
        description="Search the codebase for answers to your question."
    )

    return search_codebase_tool