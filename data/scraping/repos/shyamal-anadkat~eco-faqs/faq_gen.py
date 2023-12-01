import nltk
import pandas as pd
import os
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
from typing import List
import numpy as np
import streamlit as st

nltk.download("punkt")

openai.api_key = os.getenv("OPENAI_API_KEY")


# openai.organization = os.getenv("OPENAI_ORGANIZATION")


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
@st.cache(suppress_st_warning=True, persist=True)
def get_embedding(
    text: str, engine="text-similarity-davinci-001"
) -> List[float]:
    """
    It takes a string of text and returns embeddings for the text

    :param text: The text to embed
    :type text: str
    :param engine: The name of the engine to use, defaults to text-similarity-davinci-001 (optional)
    :return: A list of floats.
    """
    # replace newlines, which can negatively affect performance.
    text = text.replace("\n", " ")

    return openai.Embedding.create(input=[text], engine=engine)["data"][0][
        "embedding"
    ]


def cosine_similarity(a, b):
    """
    It takes two vectors, a and b, and returns the cosine of the angle between them

    :param a: the first vector
    :param b: the number of bits to use for the hash
    :return: The cosine similarity between two vectors.
    """
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_df_with_chunks_embedded(text: str) -> pd.DataFrame:
    """
    It splits the text into chunks, embeds each chunk, and returns a dataframe with the chunks and their embeddings

    :param text: The text to be split into chunks
    :type text: str
    :return: A dataframe with the chunks as rows and the search embedding as a column.
    """
    # Split the text into sentences (could also use TextWrap or something similar)
    sentences = nltk.sent_tokenize(text)

    # Determine the chunk size (empirically here 6 worked for this corpus)
    """ 
    You can also use GPT2 tokenizer in a smart way to chunk the test for example: 
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text))
    """
    chunk_size = len(sentences) // 6

    # Split the list of sentences into chunks
    chunks = [
        sentences[i : i + chunk_size]
        for i in range(0, len(sentences), chunk_size)
    ]

    # Concatenate the sentences in each chunk into a single string
    chunks = [" ".join(chunk) for chunk in chunks]

    # Create a dataframe with the chunks as rows
    df_with_chunks = pd.DataFrame(chunks, columns=["chunk"])

    df_with_chunks["search"] = df_with_chunks.chunk.apply(
        lambda x: get_embedding(x, engine="text-search-davinci-doc-001")
    )

    return df_with_chunks


@st.cache(suppress_st_warning=True, persist=True)
def search_material(df: pd.DataFrame, query: str, n=3) -> pd.DataFrame:
    """
    It takes a query and a dataframe of search embeddings, and returns the top n most similar documents

    :param df: the dataframe containing the search column
    :type df: pd.DataFrame
    :param query: the query string
    :type query: str
    :param n: the number of results to return, defaults to 3 (optional)
    :return: A dataframe with the top n results from the search query.
    """
    embedding = get_embedding(query, engine="text-search-davinci-query-001")

    df["similarities"] = df.search.apply(
        lambda x: cosine_similarity(x, embedding)
    )

    res = df.sort_values("similarities", ascending=False).head(n)

    return res


def join_chunks(chunks) -> str:
    return "\n".join(chunks)


@st.cache(suppress_st_warning=True, persist=True)
def generate_questions(
    chunks: str, topic: str, audience: str, num_questions=3
) -> list:
    """
    It takes a chunk of text, a topic, and an audience, and returns a list of questions that can be answered by the text

    :param chunks: the text to generate questions from
    :type chunks: str
    :param topic: The topic of the FAQ
    :type topic: str
    :param audience: The audience for the questions
    :type audience: str
    :param num_questions: The number of questions to generate, defaults to 3 (optional)
    :return: A list of questions
    """
    response = completion_with_backoff(
        model="text-davinci-002",
        prompt=f'Use the excerpt below to generate {num_questions} FAQ questions for {audience} related to the topic of "{topic}".'
        f"Each question should be answerable based on the information in the excerpt.\n###\nExcerpt:{chunks}\n###\n-",
        temperature=0,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    questions = response["choices"][0]["text"].strip().split("\n")
    response = list(map(lambda s: s.replace("-", " ").strip(), questions))
    print(f"Generated questions: {response}")
    return response


@st.cache(suppress_st_warning=True, persist=True)
def generate_qna(info: str) -> str:
    response = completion_with_backoff(
        model="text-davinci-002",
        prompt=f"First write five high level questions, then answer each one\nUse the following format:\nQuestions:\n1. "
        f"<question_1>\n2. <question_2>\n...\n5. <question_5>\n\nAnswers:\n1. <answer_1>\n2. <answer_2>\n...\n5. "
        f"<answer_5>\n\n\n{info}",
        temperature=0,
        max_tokens=970,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["choices"][0]["text"].strip()


@st.cache(suppress_st_warning=True, persist=True)
def generate_answer_from_question(info: str, q: str) -> str:
    """
    It takes a question and relevant corpus, and returns a generated answer
    """
    response = completion_with_backoff(
        model="text-davinci-002",
        # Tip: Ask the model to answer as if it were an expert
        # For example: "The following answer is correct, high-quality and written by an expert in the field of <topic>."
        prompt=f"You are a climate change educator. Using only the information and facts provided below, "
        f"provide a comprehensive and truthful answer to the following question.\n\n{info}\n\nQuestion:{q}\n###\n\nAnswer:",
        temperature=0,
        max_tokens=970,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response["choices"][0]["text"].strip()


def get_qa_pair(questions: list, df: pd.DataFrame) -> list:
    """
    It takes a list of questions and a corpus, and returns a list of dictionaries, each of which contains
    a question and its corresponding answer

    :param questions: list of questions
    :type questions: list
    :param df: the dataframe containing the corpus
    :type df: pd.DataFrame
    :return: A list of dictionaries.
    """
    qa_pair = []
    for question in questions:
        res = search_material(df, question)
        blurb = join_chunks(res["chunk"])
        pair = {
            "question": question.replace("-", " ").strip(),
            "answer": generate_answer_from_question(blurb, question),
        }
        qa_pair.append(pair)
    return qa_pair


@st.cache(suppress_st_warning=True, persist=True)
def is_flagged_for_content_violation(input: str) -> bool:
    """
    It takes in a string, sends it to the moderation API, and returns whether
    or not the string was flagged for a content violation

    :param input: The text you want to check for content violations
    :type input: str
    :return: A boolean value.
    """
    response = moderation_with_backoff(input=input)
    output = response["results"][0]
    return output["flagged"]


# URLs will expire after an hour.
def generate_illustration(prompt: str, size: str = "256x256") -> str:
    """
    It takes a prompt, and returns a URL to the DALL-E generated image

    :param prompt: the text you want to generate an image for
    :type prompt: str
    :param size: The size of the image to return, defaults to 256x256
    :type size: str (optional)
    :return: A URL to an image
    """
    response = image_gen_with_backoff(
        prompt=f"{prompt}, watercolor illustration", n=1, size=size
    )
    return response["data"][0]["url"]


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
@st.cache(suppress_st_warning=True, persist=True)
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
@st.cache(suppress_st_warning=True, persist=True)
def moderation_with_backoff(**kwargs):
    return openai.Moderation.create(**kwargs)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def image_gen_with_backoff(**kwargs):
    return openai.Image.create(**kwargs)
