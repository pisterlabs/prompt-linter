import os
from dotenv import load_dotenv
import openai
from openai import OpenAI
from typing import List
from .fuzzy_matching import find_similar_words_preprocessed_modified

load_dotenv()

system_message_for_review = (
    "You are a highly advanced AI assistant tasked with evaluating tweets. Analyze the content critically for accuracy, relevance, and potential impact. "
    "After your analysis, provide a score from 0 to 100, where 0 means the tweet does not need review and 100 means it desperately needs review. "
    "Format your response as: 'Analysis Score: [score]. [Explanation]'. "
    "Examples for scoring: "
    " - Score 0-20: The tweet is generally accurate and aligns with well-known facts. (e.g., 'The sky is blue on a sunny day.') "
    " - Score 21-40: The tweet is mostly accurate but may contain minor inaccuracies. (e.g., 'Eating carrots improves your night vision.') "
    " - Score 41-60: The tweet has a mix of accurate and questionable information or lacks context. (e.g., 'Chocolate causes acne in teenagers.') "
    " - Score 61-80: The tweet contains misleading or unverified information. (e.g., 'Drinking coffee is linked to heart disease.') "
    " - Score 81-100: The tweet is likely spreading misinformation or false claims. (e.g., 'Vaccines cause autism.') "
    "Keep the explanation concise and neutral, focusing on the content of the tweet. One or two sentences should be sufficient. "
)

# Example output for review: "Analysis Score: 85. The tweet contains several claims that are potentially misleading
# and lacks sufficient context. There are discrepancies when cross-referenced with known data sources."

system_message_for_assistance = (
    "You are an intelligent AI assistant specialized in aiding community note writers. "
    "Your task is to synthesize accurate and relevant context from reliable sources, enhancing the understanding of tweets. "
    "Analyze the tweet and provide supplementary information that is clear, concise, and neutral. "
    "Your contributions should directly address the tweet's content, adding valuable context or correcting misinformation. "
    "Ensure to include references to reputable sources, aiming to enhance the informativeness and neutrality of community notes. "
    "Focus on verifiable facts, avoid speculation, and maintain the integrity and reliability of the notes. "
    "You should help in creating community notes that are informative, factual, and contribute positively to the understanding of the tweet's subject matter."
)

# Example output for assistance: "The tweet contains several claims that are potentially misleading
# and lacks sufficient context. For example, the tweet claims that the COVID-19 vaccine is ineffective,
# but this is not supported by the data. The vaccine is highly effective at preventing severe illness and death.


def analyze_tweet_with_lm(tweet: str, model: str, system_message_content: str) -> str:
    """
    Analyze a tweet using a specified language model.

    Args:
    tweet (str): The text of the tweet.
    model (str): The model to be used for analysis ("gpt-3.5-turbo" or "gpt-4-turbo").
    system_message_content (str): The content of the system message to guide the language model.

    Returns:
    str: Analysis result or a note synthesized by the language model.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # System message guiding the language model
    system_message = {"role": "system", "content": system_message_content}

    # User prompt with the tweet
    user_message = {"role": "user", "content": tweet}

    # Create a chat completion with the OpenAI client
    response = client.chat.completions.create(
        model=model,
        messages=[system_message, user_message],
        temperature=0.1,
    )
    return response.choices[0].message.content


import re
from typing import Tuple


def analyze_tweet(
    tweet: str, likes: int, threshold: int, sensitive_words: List[str]
) -> Tuple[str, int]:
    """
    Analyze a tweet using GPT-3.5 if it meets the like threshold and contains sensitive words.

    Args:
    tweet (str): The text of the tweet.
    likes (int): The number of likes the tweet has.
    threshold (int): The like threshold for analysis.
    sensitive_words (List[str]): List of sensitive words for fuzzy matching.

    Returns:
    Tuple[str, int]: The analysis result and the extracted score.
    """
    matched_words = find_similar_words_preprocessed_modified(tweet, sensitive_words)

    if likes >= threshold and matched_words:
        system_message_content = system_message_for_review
        model_output = analyze_tweet_with_lm(
            tweet, "gpt-3.5-turbo", system_message_content
        )

        # Extract score from the model output
        match = re.search(r"Analysis Score: (\d+)", model_output)
        score = int(match.group(1)) if match else 0

        return model_output, score
    elif not matched_words:
        return "Tweet does not contain relevant sensitive words for analysis.", 0
    else:
        return "Tweet does not meet the like threshold for analysis.", 0


def assist_with_community_note(tweet: str, model: str) -> str:
    """
    Provide assistance for writing a community note, optionally using GPT-4.
    """
    return analyze_tweet_with_lm(tweet, model, system_message_for_assistance)
