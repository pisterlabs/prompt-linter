
import os
import shutil
import json
import random
import logging

import openai
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")


# ---------------------------------------------------------------------------- #
#                                    Logger                                    #
# ---------------------------------------------------------------------------- #
logger = logging.getLogger("tweet")

# ---------------------------------------------------------------------------- #
#                                   Variables                                  #
# ---------------------------------------------------------------------------- #
IDEA_BANK = "idea_bank"
USED_IDEAS = "used_ideas"
MODEL_NAME = "gpt-4"

# ---------------------------------------------------------------------------- #
#                                   Functions                                  #
# ---------------------------------------------------------------------------- #


def fetch_idea():
    '''
    Retrieve an idea from the idea bank if available.
    '''
    ideas = os.listdir(IDEA_BANK)
    if not ideas:
        return None

    idea = random.choice(ideas)
    with open(f"./{IDEA_BANK}/{idea}", "r", encoding="utf-8") as idea_file:
        idea_content = idea_file.read()

    # Move the used idea to a separate folder
    os.makedirs(USED_IDEAS, exist_ok=True)
    shutil.move(f"{IDEA_BANK}/{idea}", f"{USED_IDEAS}/{idea}")

    return idea_content


def generate_concept():
    '''
    Generate the concept for the tweet.
    '''
    with open("./src/prompts/concept_system.txt", "r", encoding="utf-8") as prompt_file:
        system_prompt = prompt_file.read()

    initial_idea = fetch_idea()
    user_content = "What should I tweet about?" if initial_idea is None else f"I have an idea for a tweet. {initial_idea}"

    conversation_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]

    concept_draft = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=conversation_history,
        temperature=1.2
    ).choices[0].message

    conversation_history.append(concept_draft)

    # Check if the concept has been used recently
    if os.path.exists("tweet_history.json"):
        with open("tweet_history.json", "r", encoding="utf-8") as tweet_history_file:
            tweet_history = json.load(tweet_history_file)

        last_20_tweets = tweet_history[-20:]

        conversation_history.append({
            "role": "user",
            "content": f"""
                Is the proposed topic similar to any of the following tweets?
                {last_20_tweets}
                Please reply with "yes" or "no" only.
            """
        })

        review_result = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=conversation_history
        ).choices[0].message

        conversation_history.append(review_result)

        if "yes" in review_result.content.lower():
            conversation_history.append({
                "role": "user",
                "content": "Provide a revised topic for the tweet that is not too similar to the most recent tweets.",
            })
            final_concept = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=conversation_history,
                presence_penalty=0.5
            ).choices[0].message
        else:
            final_concept = concept_draft
    else:
        final_concept = concept_draft

    logger.debug("Conversation history: %s", conversation_history)

    return final_concept.content.strip(), conversation_history


if __name__ == "__main__":
    concept, message_history = generate_concept()
    print(concept)
    print(json.dumps(message_history, indent=4, sort_keys=True))
