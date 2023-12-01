import os
import openai
import requests
import nltk
import json

nltk.download('punkt')

# Setup environment variables.
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
openai.api_key = key

def parseSentences(file, strip_newlines=True) -> list[str]:
    """
    Parse the sentences from a file.

    Args:
        file (str): The path to the file.
        strip_newlines (bool, optional): Whether to strip newlines from the file contents. Defaults to True.

    Returns:
        list[str]: The list of parsed sentences.
    """
    with open(file, "r") as f:
        contents = f.read().replace("\n", " ")
    return nltk.sent_tokenize(contents)

def textual_inversion(query, sentences, threshold) -> list[str]:
    """
    Use textual inversion via embeddings to find which sentences are relevant to your query, return all sentences above a specified threshold.
    ENSURE THIS IS CORRECT.
    """
    embeddings = openai.Embed(embedding_model="ada")
    query_embedding = embeddings(query)
    relevant_sentences = []
    for sentence in sentences:
        sentence_embedding = embeddings(sentence)
        similarity = query_embedding.cosine_similarity(sentence_embedding)
        if similarity > threshold:
            relevant_sentences.append(sentence)
    return relevant_sentences

def make_completion(messages, model=None, url=None, verbose=False):
    """
    Function to make a completion request to the OpenAI API.

    Args:
        messages (list): A list of message objects for the chat conversation.
        model (str, optional): The name of the model to use for the completion.
        url (str, optional): The URL to make the request to.
        verbose (bool, optional): If True, print additional information. Defaults to False.

    Returns:
        str: The content of the message from the completion, or None if the request was not successful.
    """
    header = {"Authorization": f"Bearer {key}"}
    if not url:
        url = "https://api.openai.com/v1/chat/completions"
    if not model:
        model = 'gpt-3.5-turbo'
    data = {'model': model, 'messages': messages}
    response = requests.post(url, headers=header, json=data)
    success = response.status_code != '401'
    response_json = response.json()
    if verbose:
        print("Request Success?", success)
        print(response_json)
    if success:
        completion = response_json['choices'][0]['message']['content']
        if verbose:
            print('completion:', completion)
    else:
        return None
    return completion

print(*parseSentences('test_paper.txt'), sep="\n|||")


















if False:
    Increase_syntax_understanding = "Sections of user input are separated by ###."
    # Increase_syntax_understanding = ""
    Increase_uncertainty = "If relationships is at all unclear do not fill response with guesses."
    Increase_overall_understanding = "mark the start of your parseable response with a single line of the form '=RESPONSE START>' "
    system_p = f"""
    """

    text_to_analyze = """
    """
    with open('test_paper.txt', 'r') as f:
        contents = f.read()
        text_to_analyze = contents

    target_relations = """
    Male gender identity <relation with> Substance Abuse
    """

    prompt = f"""
    You are an excellent systems thinker new to a job. For this job, you should be able to correctly and accurately detmine variable relationships as implied in \
    any text given to you. You will output your discoveries in rows of text. Each row will be of the format \
    Variable 1: replace with name | correlation: element of {{direct/inverse/negligible}} | Variable 2: replace with name | causation: element of {{var 1 -> var 2, var 2 -> var 1, bi-directional, unknown}}\
    The identity of Variable 1 and Variable 2 will be explicit asked for in the user's prompt. {Increase_uncertainty} Follow syntax strictly please, and thank you for helping me with this!

    ###Analyze the follow paper for relations between
    {target_relations}
    ###Use the following text
    {text_to_analyze}
    Thank you.
    """

    message = [{"role": "system", "content": system_p}, {"role": "user", "content": prompt}]
    completion = make_completion(message, verbose=False)
    print(completion)