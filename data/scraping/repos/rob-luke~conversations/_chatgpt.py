"""ChatGPT based AI for conversations."""

from typing import List, Optional
import tiktoken
from openai import OpenAI

client = OpenAI()


class ChunkTooLongError(Exception):
    """Raised when a chunk is too long for the model."""

    pass


def _num_tokens_from_messages(messages, model="gpt-4-1106-preview"):
    """Return the number of tokens used by a list of messages.

    Parameters
    ----------
    messages : List[dict]
        A list of message dictionaries.
    model : str, optional
        The model to use for tokenization, defaults to "gpt-4-1106-preview".

    Returns
    -------
    int
        The total number of tokens used by the messages.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print(
            "Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301."
        )
        return _num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print(
            "Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314."
        )
        return _num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-4-1106-preview":
        print(
            "Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314."
        )
        return _num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def _content_to_message(content: str):
    messages = [
        {"role": "system", "content": _system_prompt_summariser()},
        {
            "role": "user",
            "content": f"Reduce the length of this transcript to approximately half the length of the original text but not to remove any important information.\n\n {content}",
        },
    ]
    return messages


def _system_prompt_summariser():
    """Generate the system prompt for the GPT model.

    Returns
    -------
    str
        The system prompt text.
    """
    system_prompt = """
    You are a helpful writing assistant. You write accurate and precise content.
    You will take the context from conversation notes that I provide and help me write useful summaries.
    You have excellent written communication skills and always use correct grammar and spelling.

    You are excellent at communication and will also over-communicate if you are unsure about an answer or need more information.
    You always chose to provide more information where possible.
    """
    return system_prompt


def _system_prompt_query():
    """Generate the system prompt for the GPT model.

    Returns
    -------
    str
        The system prompt text.
    """
    system_prompt = """
    You are a helpful assistant. You write accurate and precise response to questions I provide.
    You will take the context from conversation notes that I provide and provide accurate response.
    If you do not know the answer, say "I do not know the answer to that question".
    You have excellent written communication skills and always use correct grammar and spelling.
    Wherever possible, you will provide a quote from the conversation transcript to support your answer
    """
    return system_prompt


def _user_prompt_summariser(transcript) -> str:
    """Generate the summary prompt for the GPT model.

    Parameters
    ----------
    transcript : str
        The meeting transcript.

    Returns
    -------
    str
        The summary prompt text.
    """
    summary_prompt = f"""
    Meeting transcript:

    {transcript}

    First, write a 3 sentence summary of the meeting.

    Then summarise the key points of this meeting and include a quote from the transcript for each point.
    Be verbose and extract as many key points as possible and ensure you adequately summarise the conversation.
    Extract at least 10 key points.

    For each point, use the format:
    1. Point
       Speaker: Relevant quote

    List all action items discussed in the meeting and who is responsible for each action item.

    Next, extract 20 key words and list them in order from most to least relevant to the conversation.

    Next, extract 10 concepts and themes and list them in order from most to least relevant to the conversation.

    What was the most unexpected aspect of the conversation?

    Where any concepts or topics explained in the transcript? If so, list each concept and its definition as explained in the call. Highlight if the concept was incorrectly described in the call.

    What was the tone of the conversation? List the tone of the conversation as a whole and the tone of each speaker.
    """
    return summary_prompt


def _user_prompt_query(transcript: str, query: str) -> str:
    """Create a prompt for querying a transcript.

    Parameters
    ----------
    transcript : str
        The meeting transcript.
    query : str
        The query to ask the system.

    Returns
    -------
    str
        The query prompt text.
    """
    query_prompt = f"""
    Below is a transcript from conversation. Please read the transcript and then answer the following question:
    
    Meeting transcript:

    {transcript}
    
    Now, please answer the following question:

    {query}
    """
    return query_prompt


def summarise(
    transcript: str,
    system_prompt: Optional[str] = None,
    summary_prompt: Optional[str] = None,
    append_prompt: Optional[
        str
    ] = "Format your response as text and do not use markdown.",
) -> str:
    """Generate a meeting summary using the GPT model based on the given meeting transcript.

    Parameters
    ----------
    transcript : str
        The meeting transcript.
    system_prompt : Optional[str], default=None
        The system prompt text. If not provided, it will be generated using the internal function _system_prompt_summariser().
    summary_prompt : Optional[str], default=None
        The summary prompt text. If not provided, it will be generated using the internal function _user_prompt_summariser(transcript).
    append_prompt : Optional[str]
        An additional prompt to append to the summary_prompt.

    Returns
    -------
    str
        The generated meeting summary.
    """
    if system_prompt is None:
        system_prompt = _system_prompt_summariser()

    if summary_prompt is None:
        summary_prompt = _user_prompt_summariser(transcript)

    if append_prompt is not None:
        summary_prompt += append_prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": summary_prompt},
    ]

    response = client.chat.completions.create(
        model="gpt-4-1106-preview", temperature=0.3, messages=messages  # type: ignore
    )
    return str(response.choices[0].message.content)


def query(
    transcript: str,
    query: str,
    system_prompt: Optional[str] = None,
    append_prompt: Optional[
        str
    ] = "Format your response as text and do not use markdown.",
) -> Optional[str]:
    """Generate a response to a query using the GPT model based on the given meeting transcript.

    Parameters
    ----------
    transcript : str
        The meeting transcript.
    query : str
        What you would like to know from the conversation. This will be used to generate the query prompt.
    system_prompt : Optional[str], default=None
        The system prompt text. If not provided, it will be generated using the internal function _system_prompt_summariser().
    append_prompt : Optional[str]
        An additional prompt to append to the query_prompt.

    Returns
    -------
    answer : str
        The generated response to the query.
    """
    if system_prompt is None:
        system_prompt = _system_prompt_summariser()

    query_prompt = _user_prompt_query(transcript, query)

    if append_prompt is not None:
        query_prompt += append_prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query_prompt},
    ]

    response = client.chat.completions.create(
        model="gpt-4-1106-preview", temperature=0.3, messages=messages  # type: ignore
    )
    return response.choices[0].message.content
