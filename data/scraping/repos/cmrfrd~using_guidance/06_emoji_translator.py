import json
import re
from enum import Enum
from typing import Any, Generator, Optional

import backoff
import numpy as np
import numpy.typing as npt
import openai
from emoji.unicode_codes import get_emoji_unicode_dict
from guidance import Program, llms
from pydantic import BaseModel
from tqdm import tqdm


class Emoji(BaseModel):
    name: str
    value: str


class EmojiEmbeddings(BaseModel):
    emojis: list[Emoji]
    embeddings: list[list[float]]

    class Config:
        json_loads = json.loads
        json_dumps = json.dumps


class EmojiVectorDatabase(BaseModel):
    emojis: list[Emoji]
    embeddings: npt.NDArray[np.float32]

    class Config:
        allow_mutation = False
        arbitrary_types_allowed = True

    def top_k(self, query: str, k: int = 10) -> list[Emoji]:
        query_embedding_response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=query,
        )
        query_embedding: npt.NDArray[np.float32] = np.array(
            query_embedding_response["data"][0]["embedding"]
        )

        # cosine similarity
        similarity: npt.NDArray[np.float32] = np.dot(query_embedding, self.embeddings.T) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(self.embeddings, axis=1)
        )

        # sort by similarity
        sorted_similarity_indices: npt.NDArray[np.int64] = np.argsort(similarity)[::-1]

        # get top k
        top_k_emojis: list[Emoji] = [self.emojis[i] for i in sorted_similarity_indices[:k]]

        return top_k_emojis


def chunks(lst: list, n: int) -> Generator[list, None, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


@backoff.on_exception(backoff.expo, Exception, max_time=60)
def backoff_embeddings(**kwargs) -> Any:
    return openai.Embedding.create(**kwargs)


def get_emoji_embeddings() -> EmojiEmbeddings:
    EMOJI_EMBEDDINGS_FILENAME = "./vecdbs/emoji_name_value_embeddings.json"

    try:
        with open(EMOJI_EMBEDDINGS_FILENAME, "r") as f:
            return EmojiEmbeddings.parse_raw(f.read())
    except FileNotFoundError:
        pass

    emojis: list[Emoji] = [
        Emoji(name=name, value=value) for name, value in get_emoji_unicode_dict("en").items()
    ]
    embeddings: list[list[float]] = []
    chunk_size: int = 8
    for _, e_chunk in tqdm(enumerate(chunks(emojis, chunk_size)), total=len(emojis) // chunk_size):
        embeddings_results = [
            backoff_embeddings(
                model="text-embedding-ada-002",
                input=f"{e.name} {e.value}",
            )
            for e in e_chunk
        ]
        embeddings.extend(
            [embedding_result["data"][0]["embedding"] for embedding_result in embeddings_results]
        )

    emoji_embeddings: EmojiEmbeddings = EmojiEmbeddings(emojis=emojis, embeddings=embeddings)

    ## Save to file
    with open(EMOJI_EMBEDDINGS_FILENAME, "w") as f:
        f.write(emoji_embeddings.json())
    return emoji_embeddings


def translate_to_emojis(
    input_text: str, emoji_vector_database: EmojiVectorDatabase, top_k: int = 50
) -> str:
    """
    Translate a string to emojis.

    Args:
        input_text: The string to translate.

    Returns:
        The string translated to emojis.
    """
    smile_emoji = "😀"
    sad_emjoi = "😞"
    smile_or_sad = "{{#select}}" + smile_emoji + "{{or}}" + sad_emjoi + "{{/select}}"

    top_k_emojis: list[str] = [e.value for e in emoji_vector_database.top_k(input_text, k=top_k)]
    emoji_select = "{{#select}}" + "{{or}}".join(top_k_emojis) + "{{/select}}"
    gen_emoji_select = (
        "{{#geneach 'emojis'}}{{#unless @first}}{{/unless}}" + emoji_select + "{{/geneach}}"
    )
    test = "{{#geneach 'emojis'}}" + emoji_select + "{{/geneach}}"

    base_template = f"""
    # Goal
    
    You are an AI emoji translator. Your goal is to read a input text and translate it to emojis.
    
    # Constraints
    
    - The output emoji sequence must match the semantic meaning of the text.
    - Use your internal knowledge of what emojis mean to translate the text. 
    - Make sure that a human would agree that the emoji sequence matches the semantic meaning of the input text.
    - Use as few emojis as possible.

    # Input/Output
    
    Input: {input_text}
    Output: {test}
    """

    program_result: Program = Program(
        base_template, llm=llms.OpenAI("text-davinci-003"), caching=False, silent=True
    )()
    print(program_result)
    return str(program_result.emojis)


emoji_embeddings: EmojiEmbeddings = get_emoji_embeddings()
emoji_vector_database: EmojiVectorDatabase = EmojiVectorDatabase(
    emojis=emoji_embeddings.emojis,
    embeddings=np.array(emoji_embeddings.embeddings),
)

print(translate_to_emojis("Harry Potter's wand", emoji_vector_database))
