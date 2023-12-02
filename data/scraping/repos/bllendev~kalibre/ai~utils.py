import openai
from ai.models import Message

import tiktoken
from openai.embeddings_utils import get_embedding


def fx_query_openai(**kwargs):
    """
    - queries openai with the given query and messages
    - returns the response from openai
    """
    # get the kwargs or set defaults
    query = kwargs.get('query')
    user_messages = kwargs.get('user_messages', list())
    ai_messages = kwargs.get('ai_messages', list())
    system_prompt = kwargs.get('system_prompt', "")
    temperature = kwargs.get('temperature', 0.7)

    # add system prompt to the beginning of the conversation
    conversation_list = [Message.get_message(system_prompt, role="system")]

    # add existing (if available) user and ai messages to the conversation
    for idx, user_msg in enumerate(user_messages):
        user_msg = Message.get_message(user_msg, role="user")
        if user_msg:
            conversation_list.append(user_msg)

        ai_msg = Message.get_message(ai_messages[idx], role="ai")
        if ai_msg:
            conversation_list.append(ai_msg)

    # add final query to the conversation
    conversation_list.append(Message.get_message(query, role="user"))

    # query openai with the conversation
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_list,
        max_tokens=2000,
        temperature=temperature,
    )
    return response['choices'][0]['message']['content'].strip()


def get_openai_embeddings(texts, embedding_model="text-embedding-ada-002", max_tokens=8000):
    """Compute embeddings for a list of texts using OpenAI's API.

    Args:
    texts (list of str): The texts to compute embeddings for.
    embedding_model (str, optional): The OpenAI model to use for computing embeddings.
    max_tokens (int, optional): The maximum number of tokens per text.

    Returns:
    list of embeddings: The embeddings computed by OpenAI's API.
    """
    pass
    # embeddings = []
    # tokenizer = tiktoken.get_encoding("cl100k_base")  # moved outside of the loop

    # for text in texts:
    #     token_ids = list(tokenizer.encode(text))
    #     n_tokens = len(token_ids)
    #     if n_tokens <= max_tokens:
    #         embedding = get_embedding(text, engine=embedding_model)
    #         embeddings.append(embedding)

    # return embeddings
