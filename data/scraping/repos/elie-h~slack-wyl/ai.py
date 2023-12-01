from langchain import LLMChain, PromptTemplate
from langchain.chat_models import ChatOpenAI
from embeddings import query_vs

from utils import Message, RoleEnum
from config import OPENAI_API_KEY, LANGCHAIN_VERBOSE


def format_conversation(conversation: list[Message]) -> str:
    formatted_history = ""
    for message in conversation:
        if message.role == RoleEnum.USER:
            formatted_history += f"user: {message.content} \n"
        elif message.role == RoleEnum.AI:
            formatted_history += f"ai: {message.content} \n"

    formatted_history = formatted_history.strip().rstrip("\r\n")
    return formatted_history


async def generate_response(conversation: list[Message]) -> str:
    template = (
        "You are now Wyl, a chatbot that can perform a wide range of tasks, "
        "from answering simple questions to providing in-depth explanations "
        "and discussions on a wide range of topics. "
        "As a language model, Wyl is able to generate human-like text based on "
        "the input it receives, allowing it to engage in natural-sounding "
        "conversations and provide responses that are coherent and relevant "
        "to the topic at hand.\n"
        "Here is a conversation between Wyl and a human, continue the "
        "conversation by typing your response below, you may only use "
        "information from the conversation history:\n"
        "{history}\nuser: {human_input}\nassistant:"
    )
    prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

    llm = ChatOpenAI(client=None, openai_api_key=OPENAI_API_KEY, temperature=0)
    chatgpt_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=LANGCHAIN_VERBOSE,
    )

    formatted_history = format_conversation(conversation[:-1])
    output = await chatgpt_chain.apredict(human_input=conversation[-1].content, history=formatted_history)
    return output


def conversation_to_query(conversation: list[Message]) -> str:
    """
    This function should take a conversation and return a query to be used to retrive context
    """

    prompt_template = (
        "Given the following conversation between a user and an AI "
        "come up with a descriptive sentence that can be used in a search engine "
        "to find relevant information/documents need to answer the user's query "
        "ensure the sentence contains as many relevant keywords as possible so "
        "that we can get good matches even if it means duplicating words with "
        "the same semantic meaning. Do not introduce specific detail that's not in the current conversation "
        "Here is the conversation:\n"
        "{chat_history}"
        "\nProduce the required sentence descriptively and concisely "
        "without any explanation or any prefixes or suffixes, "
        "use lots of detail and synonyms and focus on keywords that might get relevant "
        "hits when running through a similarity search using a vector store:"
    )

    question_prompt = PromptTemplate(template=prompt_template, input_variables=["chat_history"])

    llm = ChatOpenAI(client=None, openai_api_key=OPENAI_API_KEY, temperature=0)
    chatgpt_chain = LLMChain(
        llm=llm,
        prompt=question_prompt,
        verbose=LANGCHAIN_VERBOSE,
    )

    formatted_history = format_conversation(conversation)
    output = chatgpt_chain.predict(chat_history=formatted_history)
    return output


def generate_response_V2(conversation: list[Message]) -> str:
    """
    Some general thoughts:
    The below retrieval is quite simple, we should use an agent which should think and retrieve relevant information
    from the knowledge base. This agent should be able to ask questions to the user to clarify what they mean.
    The agent should write up the correct queries to retrieve the correct information from the knowledge base
    We can also be clever with which model we use to achieve this, using GPT4 for all is going to be expensive.
    """
    prompt_template = (
        "You are now Wyl, you are a software engineer specialised in answering questions from context.\n"
        "Your abilities are: Comprehension, Advising Engineers, and Answering Questions.\n"
        "Your main task is to answer questions from context provided to you.\n"
        "{context}\n"
        "You can also ask the user questions to clarify what they mean.\n"
        "If you don't know the answer from the context provided, prefix your answer with: "
        "I do not know the exact answer to that but I would guess, or something similar.\n"
        "You can also ask the user questions to clarify what they mean or ask for more information.\n"
        "The following is the chat history between the user and Wyl:\n"
        "{chat_history}\nWyl:"
    )
    context_search_query = conversation_to_query(conversation)
    docs = query_vs(context_search_query, k=10, threshold=0.7)
    if len(docs) > 0:
        initial_context = (
            "Give accurate and thorough answers using the context provided where relevant"
            "- the context is as follows:\n\n"
        )
        context = " ".join([doc.replace("\n", " ") for doc in docs])
        context = initial_context + context
    else:
        context = ""

    question_prompt = PromptTemplate(template=prompt_template, input_variables=["context", "chat_history"])

    llm = ChatOpenAI(client=None, openai_api_key=OPENAI_API_KEY, temperature=0)
    chatgpt_chain = LLMChain(
        llm=llm,
        prompt=question_prompt,
        verbose=LANGCHAIN_VERBOSE,
    )

    formatted_history = format_conversation(conversation)
    output = chatgpt_chain.predict(context=context, chat_history=formatted_history)
    return output


async def generate_thread_summary(conversation: list[Message]) -> str:
    template = (
        "Summarise the following conversation concicely and accurately without "
        "losing any important information or adding any new information:\n"
        "{history}\nassistant:"
    )

    prompt = PromptTemplate(input_variables=["history"], template=template)
    llm = ChatOpenAI(client=None, openai_api_key=OPENAI_API_KEY, temperature=0)
    chatgpt_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=LANGCHAIN_VERBOSE,
    )

    formatted_history = format_conversation(conversation[:-1])
    output = await chatgpt_chain.apredict(history=formatted_history)
    return f"Here's your summary: \n{output}"


if __name__ == "__main__":
    res = conversation_to_query(
        conversation=[
            Message(content="What is the best programming language?", role=RoleEnum.USER),
        ]
    )

    print(res)
