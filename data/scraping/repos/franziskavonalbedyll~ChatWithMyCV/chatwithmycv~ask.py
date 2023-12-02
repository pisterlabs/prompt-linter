"""
Generate answer to user question using the OpenAI API.

Usage:
- Ensure that the OpenAI API key is stored in a `.env` file.
- Use the `ask_question` function to interact with the chatbot and retrieve
  answers based on the provided question and embeddings.

Example:
    >>> from langchain.vectorstores import Chroma
    >>> embeddings = Chroma(...)
    >>> question = 'What is the capital of France?'
    >>> answer = ask_question(question, embeddings)
    >>> print(answer)
"""
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

MEMORY = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True, output_key="answer"
)


def ask_question(question: str, embeddings: Chroma) -> str:
    """Generate answer to user question.

    :param question: question to ask
    :param embeddings: word embeddings
    :return: answer
    """
    chat = ChatOpenAI()
    chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=embeddings.as_retriever(),
        memory=MEMORY,
        max_tokens_limit=4096,
    )

    response = chain({"question": question})
    answer = response["answer"]

    return answer
