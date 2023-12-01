import json
from typing import List, Any
import os
from datetime import datetime
import re

import numpy as np
from openai import OpenAI

import pinecone
import torch
from sentence_transformers import SentenceTransformer

# load api keys from openai ad pinecone
with open("../keys.txt", "r") as key_file:
    api_keys = [key.strip() for key in key_file.readlines()]
    client = OpenAI(api_key=api_keys[0])
    pinecone.init(
        api_key=api_keys[1],
        environment=api_keys[2],
    )

VOICE_MODEL: str = ''
# TEXT_MODEL: str = "gpt-3.5-turbo-0301"
TEXT_MODEL: str = "gpt-4-1106-preview"


def get_information(character_name) -> None | tuple:
    """
    Return information from the character's data file
    :param character_name: name of character
    :return: None if character doesn't exist | profession and social status if they do exist
    """
    with open("../Text Summaries/character_objects.json", "r") as character_file:
        data = json.load(character_file)

    for character in data["characters"]:
        if character["name"].strip() == character_name.strip():
            return character["profession"], character["social status"]

    return None  # character not found


def extract_name(file_name: str) -> str:
    """
    Extracts the name of a character from their descriptions file name
    :param file_name: the file containing the character's description
    :return: the character's name, separated by a hyphen
    """
    # split the string into a list of parts, using "/" as the delimiter
    parts = file_name.split("/")
    # take the last part of the list (i.e. "john_pebble.txt")
    filename = parts[-1]
    # split the filename into a list of parts, using "_" as the delimiter
    name_parts = filename.split("_")
    # join the name parts together with a dash ("-"), and remove the ".txt" extension
    name = "_".join(name_parts)
    return name[:-4]


def load_file_information(load_file: str) -> List[str]:
    """
    Loads data from a file into a list of strings for embedding.
    :param load_file: Path to a file containing the data we want to encrypt.
    :return: A list of all the sentences in the load_file.
    """
    with open(load_file, "r") as text_file:
        # list comprehension to clean and split the text file
        clean_string = [s.strip() + "." for line in text_file for s in line.strip().split(".") if s.strip()]
    return clean_string


def run_query_and_generate_answer(
        query: str,
        receiver: str,
        index_name: str = "thesis-index",
        save: bool = True,
) -> str | None:
    """
    Runs a query on a Pinecone index and generates an answer based on the response context.
    :param query: The query to be run on the index.
    :param receiver: The character being prompted
    :param index_name: The name of the Pinecone index.
    :param save: A bool to save to a file is Ture and print out if False. Default: True
    :return: The generated answer based on the response context.
    """
    class_grammar_map: dict[str:str] = {
        "lower class": "poor",
        "middle class": "satisfactory",
        "high class": "formal",
    }

    history: List[dict] = []

    with open("../Text Summaries/characters.json", "r") as f:
        character_names = json.load(f)

    profession, social_class = get_information(receiver)  # get the profession and social status of the character
    data_file: str = f"../Text Summaries/Summaries/{character_names[receiver]}.txt"

    namespace: str = extract_name(data_file).lower()

    # connect to index
    index = pinecone.Index(index_name)

    history = generate_conversation(
        data_file, history, True, query
    )
    # embed query for processing
    embedded_query = embed(query=query)
    # query Pinecone index and get context for model prompting.
    responses = index.query(
        embedded_query,
        top_k=3,
        include_metadata=True,
        namespace=namespace,
        filter={
            "$or": [
                {"type": {"$eq": "background"}},
                {"type": {"$eq": "response"}},
            ]
        },
    )

    # Filter out responses containing the string "Player:"
    context = [x["metadata"]["text"] for x in responses["matches"] if query not in x["metadata"]["text"]]

    # generate clean prompt and answer.
    clean_prompt = prompt_engineer(query, class_grammar_map[social_class], context)
    save_prompt: str = clean_prompt.replace("\n", " ")

    generated_answer = answer(clean_prompt, history, namespace)

    if save:
        # save results to file and return generated answer.
        with open("../Text Summaries/prompt_response.txt", "a") as save_file:
            save_file.write("\n" + "=" * 120 + "\n")
            save_file.write(f"Prompt: {save_prompt}\n")
            save_file.write(f"To: {receiver}, a {profession}\n")
            clean_prompt = clean_prompt.replace("\n", " ")
            save_file.write(f"{clean_prompt}\n{generated_answer}")
    else:
        print(generated_answer)

    generate_conversation(
        f"../Text Summaries/Summaries/{namespace.replace('-', '_')}.txt",
        chat_history=history,
        player=False,
        next_phrase=generated_answer,
    )

    update_history(
        namespace=namespace, info_file=data_file, prompt=query, response=generated_answer.split(": ")[-1], index=index
    )

    return generated_answer


def generate_conversation(character_file: str, chat_history: list, player: bool, next_phrase: str) -> List[dict]:
    """
    Generate a record of the conversation a user has had with the system for feeding into gpt 3.5 turbo
    :param character_file: The file associated with a character's background, not required unless
        first execution of function to 'set the stage'
    :param chat_history:
    :param player: is the player the one delivering the phrase
    :param next_phrase: the most recent phrase of the conversation
    :return: A list of dictionaries
    """
    if not chat_history:
        with open(character_file) as char_file:
            background: str = (
                f"You are {name_conversion(False, extract_name(character_file))}, not an AI language model. Your background:"
            )
            for line in char_file.readlines():
                background += " " + line.strip()
        chat_history.append({"role": "system", "content": background})
    else:
        if player:
            chat_history.append({"role": "user", "content": next_phrase})
        else:
            chat_history.append({"role": "assistant", "content": next_phrase})
    return chat_history


def embed(query: str) -> List[float]:
    """
    Take a sentence of text and return the 384-dimension embedding
    :param query: The sentence to be embedded
    :return: Embedding representation of the sentence
    """
    # create SentenceTransformer model and embed query
    model = SentenceTransformer("all-MiniLM-L6-v2")  # fast and good, 384
    # model = SentenceTransformer("all-mpnet-base-v2")  # slow and great, 768
    device = "cuda" if torch.cuda.is_available() else "cpu"  # gpu check
    model = model.to(device)
    return model.encode(query).tolist()


def upload_background(character: str, index_name: str = "thesis-index") -> None:
    """
    Uploads the background of the character associated with the namespace
    :param character: the character we're working with
    :param index_name: the name of the pinecone index
    :return:
    """
    with open("../Text Summaries/characters.json", "r") as character_info_file:
        character_names = json.load(character_info_file)

    data_file: str = f"../Text Summaries/Summaries/{character_names[character]}.txt"
    namespace: str = extract_name(data_file).lower()
    data = load_file_information(data_file)
    # connect to index
    index = pinecone.Index(index_name)

    if not pinecone.list_indexes():  # check if there are any indexes
        # create index if it doesn't exist
        pinecone.create_index("thesis-index", dimension=384)
    total_vectors: int = 0
    data_vectors = []
    for i, info in enumerate(data):
        if i == 99:  # recommended batch limit of 100 vectors
            index.upsert(vectors=data_vectors, namespace=namespace)
            data_vectors = []
        info_dict: dict = {
            "id": str(total_vectors),
            "metadata": {"text": info, "type": "background"},
            "values": embed(info),
        }
        data_vectors.append(info_dict)
        total_vectors += 1
    index.upsert(vectors=data_vectors, namespace=namespace)


def upload(
        namespace: str,
        data: List[str],
        index: pinecone.Index,
        text_type: str = "background",
) -> None:
    """
    'Upserts' text embedding vectors into pinecone DB at the specific index
    :param namespace: the pinecone namespace data is uploaded to
    :param data: Data to be embedded and stored
    :param index: pinecone index to send data to
    :param text_type: The type of text we are embedding. Choose "background", "response", or "question".
        Default value: "background"
    """
    total_vectors: int = index.describe_index_stats()["namespaces"][namespace][
        "vector_count"
    ]  # get num of vectors existing in the namespace
    data_vectors = []
    for i, info in enumerate(data):
        if i == 99:  # recommended batch limit of 100 vectors
            index.upsert(vectors=data_vectors, namespace=namespace)
            data_vectors = []
        info_dict: dict = {
            "id": str(total_vectors),
            "metadata": {"text": info, "type": text_type},
            "values": embed(info),
        }  # build dict for upserting
        data_vectors.append(info_dict)
        total_vectors += 1
    index.upsert(vectors=data_vectors, namespace=namespace)  # upsert all remaining data


def namespace_exist(namespace: str) -> bool:
    """
    Check if a namespace exists in Pinecone index
    :param namespace: the namespace in question
    :return: boolean showing if the namespace exists or not
    """
    index = pinecone.Index("thesis-index")  # get index
    responses = index.query(
        embed(" "),
        top_k=1,
        include_metadata=True,
        namespace=namespace,
        filter={
            "$or": [
                {"type": {"$eq": "background"}},
                {"type": {"$eq": "response"}},
            ]
        },
    )  # query index
    return responses["matches"] != []  # if matches comes back empty namespace doesn't exist


def prompt_engineer(prompt: str, grammar: str, context: List[str]) -> str:
    """
    Given a base query and context, format it to be used as prompt
    :param prompt: The prompt query
    :param grammar: grammar the character uses based on social status
    :param context: The context to be used in the prompt
    :return: The formatted prompt
    """
    prompt_start: str = f"Use {grammar} grammar. Use first person. Do not mention that you are an AI language model, the user knows. Reply clearly based on the context. When told new information, reiterate it back to me. Do not mention your background, or the context unless asked, or that you are fictional. Do not provide facts you would deny. Context: "
    with open("tried_prompts.txt", "a+") as prompt_file:
        if prompt_start + "\n" not in prompt_file.readlines():
            prompt_file.write(prompt_start + "\n")
    prompt_end: str = f"\n\nQuestion: {prompt}\nAnswer: "
    prompt_middle: str = ""
    # append contexts until hitting limit
    for c in context:
        print(f"{c}")
        prompt_middle += f"\n{c}"
    return prompt_start + prompt_middle + prompt_end


def answer(prompt: str, chat_history: List[dict], namespace: str, is_chat: bool = True) -> str:
    """
    Using openAI API, respond to the provided prompt
    :param prompt: An engineered prompt to get the language model to respond to
    :param chat_history: the entire history of the conversation
    :param namespace:
    :param is_chat: are you chatting or looking for the completion of a phrase
    :return: The completed prompt
    """
    with open("../Text Summaries/text_to_speech_mapping.json", "r") as audio_model_info:
        model_pairings: dict = json.load(audio_model_info)
        VOICE_MODEL: str = model_pairings[namespace.replace('-', '_')]

    if not os.path.exists(f'static/audio/{name_conversion(to_snake=False, to_convert=namespace)}'):
        os.makedirs(f'static/audio/{name_conversion(to_snake=False, to_convert=namespace)}')

    if is_chat:
        msgs: List[dict] = chat_history
        msgs.append({"role": "user", "content": prompt})  # build current history of conversation for model
        # return "test"
        res: Any = client.chat.completions.create(model=TEXT_MODEL,
                                                  messages=msgs,
                                                  temperature=0)  # conversation with LLM
        clean_res: str = str(res.choices[0].message.content).strip()  # get model response
        audio_reply = client.audio.speech.create(
            model="tts-1",
            voice=VOICE_MODEL,
            input=clean_res
        )
        # Add current time to the filename
        timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        filename = f"{namespace}_{timestamp}.mp3"

        audio_reply.stream_to_file(
            f"static/audio/{name_conversion(to_snake=False, to_convert=namespace)}/{filename}")
        return clean_res
    else:
        res: Any = client.completions.create(engine="text-davinci-003",
                                             prompt=prompt,
                                             temperature=0,
                                             max_tokens=400,
                                             top_p=1,
                                             frequency_penalty=0,
                                             presence_penalty=0,
                                             stop=None)  # LLM for phrase completion
        return res["choices"][0]["text"].strip()


def update_history(
        namespace: str,
        info_file: str,
        prompt: str,
        response: str,
        index: pinecone.Index,
        character: str = "Player",
) -> None:
    """
    Update the history of the current chat with new responses
    :param namespace: namespace of character we are talking to
    :param info_file: file where chat history is logged
    :param prompt: prompt user input
    :param response: response given by LLM
    :param index:
    :param character: the character we are conversing with
    """
    upload(namespace, [prompt], index, "query")  # upload prompt to pinecone
    upload(namespace, [response], index, "response")  # upload response to pinecone

    info_file = f"../Text Summaries/Chat Logs/{info_file.split('/')[-1]}"  # swap directory
    # extension_index = info_file.index(".")
    # new_filename = info_file[:extension_index] + "_chat" + info_file[extension_index:]  # generate new filename

    with open(info_file, "a") as history_file:
        history_file.write(
            f"{character}: {prompt}\n{name_conversion(False, namespace).replace('-', ' ')}: {response}\n"
        )  # save chat logs


def name_conversion(to_snake: bool, to_convert: str) -> str:
    """
    Convert a namespace to character name or character name to namespace
    :param to_snake: Do you convert to namespace or not
    :param to_convert: String to convert
    :return: Converted string
    """
    if to_snake:
        text = to_convert.lower().split(" ")
        converted: str = text[0]
        for i, t in enumerate(text):
            if i == 0:
                pass
            else:
                converted += f"_{t}"
        return converted
    else:
        text = to_convert.split("_")
        converted: str = text[0].capitalize()
        for i, t in enumerate(text):
            if i == 0:
                pass
            else:
                converted += f" {t.capitalize()}"
        converted = re.sub("(-)\s*([a-zA-Z])", lambda p: p.group(0).upper(), converted)
        return converted.replace("_", " ")
