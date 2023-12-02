import os

import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4"


def get_disclaimer():
    '''
    Retrieve the disclaimer text.
    '''
    with open("./src/prompts/disclaimer.txt", "r", encoding="utf-8") as disclaimer_file:
        disclaimer = disclaimer_file.read()
        disclaimer = disclaimer.replace("{{GITHUB_USERNAME}}", os.getenv("GITHUB_USERNAME"))

    return "\n\n" + disclaimer


def draft_tweet(seed):
    """
    Generate a draft tweet using the provided seed.
    """
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": """
             You will be given a topic to tweet about.
             You will also be provided context for the tweet as well as recommended hashtags.
             Your task is to return a well formatted tweet within 280 characters.
             """},
            {"role": "user", "content": seed}],
    )
    return response.choices[0].message.content.strip(), None


def check_tweet_length(tweet, max_length):
    if len(tweet) > max_length:
        return False
    return True


def generate_shorter_tweet(conversation_history):
    conversation_history.append({
        "role": "user",
        "content": "The tweet is too long. Please generate a shorter tweet, making it sound more natural and less buzzword-heavy. Feel free to drop emojis or hashtags if needed."
    })
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=conversation_history
    ).choices[0].message
    conversation_history.append(response)
    return response, conversation_history


def structure(concept):
    """
    Format the concept into a tweet using the Structure Prompt.
    """
    with open("./src/prompts/structure.txt", "r", encoding="utf-8") as structure_prompt_file:
        prompt = structure_prompt_file.read().replace("{{CONCEPT}}", draft_tweet(concept)[0])

    # Inject the tweet length into the prompt
    tweet_length = 280 - len(get_disclaimer()) - 10
    prompt = prompt.replace("{{TWEET_LENGTH}}", str(tweet_length))

    conversation_history = []
    conversation_history.append({"role": "user", "content": prompt})

    tweet_draft = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=conversation_history,
    ).choices[0].message

    conversation_history.append(tweet_draft)

    if not check_tweet_length(tweet_draft.content, tweet_length):
        final_tweet, _ = generate_shorter_tweet(conversation_history)
    else:
        final_tweet = tweet_draft

    return final_tweet.content.strip(), conversation_history
