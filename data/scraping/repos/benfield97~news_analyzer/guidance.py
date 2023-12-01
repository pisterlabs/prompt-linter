import openai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import re

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

def format_list(input_string):
    # Split the string into a list of phrases
    phrases = input_string.split(',')

    # Remove leading/trailing whitespace, quotation marks and final punctuation
    cleaned_phrases = [re.sub(r'^["\s]+|["\s]+$|[.,;:!?"]$', '', phrase) for phrase in phrases]

    return cleaned_phrases

def get_article_text(input, format = 'url'):
    # Send a request to the website
    if format == 'url':
        r = requests.get(input)

    elif format == 'html':
        r = input

    # Parse HTML and save to BeautifulSoup object
    soup = BeautifulSoup(r.text, "html.parser")

    # Find article text and combine it into one string
    article_text = ' '.join([p.text for p in soup.find_all('p')])

    return article_text


def article_detection(article_text):
    message = [{"role": "system", "content": "You are an expert on journalism."}]

    prompt = f"""
        Please assess the following body of text, which is delimited by triple backticks.

        Determine if you believe this is an article, as in a piece of writing included with others in a newspaper, magazine, or other print or online publication.

        If it is an article, format your response by only printing: True
        If it is not an article, format your response by only printing: False

        Article: ```{article_text}```
        """

    response = get_completion(prompt, message)

    if 'True' in response:
        return True
    else:
        return False


def get_completion(prompt, messages, model=4):
    messages.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(
        model=f"gpt-{model}",
        messages= messages
    )

    messages.append({"role": "system", "content": response['choices'][0]['message']['content']})

    return response['choices'][0]['message']['content']



def get_emotive_list(article_text, messages):
    prompt = f"""
    Extract all examples of emotive language used in the 
    following article, which is delimited by triple backticks.

    Format your response as a list of items separated by commas.

    Article: '''{article_text}'''
    """

    response = get_completion(prompt, messages)

    response = format_list(response)

    return response


def get_emotive_rating(messages):
    prompt = """
        Based strictly on the presence of emotive language, can you rate on a scale of 1-10 how emotive the article is.
        
        Please format your response as an integer only
        """

    response = get_completion(prompt, messages)

    try:
        return int(response)

    except:
        prompt = """
        Please try again and format this response as an integer only.
        """

        response = get_completion(prompt, messages)

        return int(response)


def get_political_bias_list(article_text, messages):
    prompt = f"""
        You are evaluating in which ways the article below, delimited by triple backticks, is politically biased, specifically, biased to 
        either the left-wing or the right-wing.
        
        Extract all examples of politically biased phrases used in the article.

        Format your response as a list of items separated by commas.
        
        Article: ```{article_text}```
        """

    response = get_completion(prompt, messages)
    response = format_list(response)

    return response


def get_political_bias_rating(messages):
    prompt = """
        You are evaluating in which political direction the previous article is biased.
        
        On a scale from 1 (strongly left-wing) to 10 (strongly right-wing) can you rate the article for the position of it's political bias.

        Please format your response as an integer only.
        """

    response = get_completion(prompt, messages)

    try:
        return int(response)

    except:
        prompt = """
        Please try again and format this response as an integer only.
        """

        response = get_completion(prompt, messages)

        return int(response)


def get_establishment_list(article_text, messages):
    prompt = f"""
            You are evaluating in which ways the article below, delimited by triple backticks, is biased in a manner that is either pro-establishment or anti-establishment.

            Extract all examples of politically biased phrases used in the article.

            Format your response as a list of items separated by commas.
            
            Article: ```{article_text}```
            """

    response = get_completion(prompt, messages)
    response = format_list(response)

    return response


def get_establishment_bias_rating(messages):
    prompt = """
        You are evaluating in which direction the previous article is biased, in regards to its stance on the establishment.

        On a scale from 1 (strongly anti-establishment) to 10 (strongly pro-establishment) can you rate the article for the position of it's establishment bias.

        Please format your response as an integer only.
        """

    response = get_completion(prompt, messages)

    try:
        return int(response)

    except:
        prompt = """
        Please try again and format this response as an integer only.
        """

        response = get_completion(prompt, messages)

        return int(response)



article = get_article_text('https://www.foxnews.com/politics/biden-admin-quietly-reverses-trump-era-rule-bans-transporting-fossil-fuels-train')

is_article = article_detection(article)

emo_msgs = [{"role": "system", "content": "You are an expert on journalism. You specialise in assessing how emotive language is used to position readers"}]
emotive_list = get_emotive_list(article, emo_msgs)


def run(url):
    article = get_article_text(url)

    is_article = article_detection(article)

    emo_msgs = [{"role": "system", "content": "You are an expert on journalism. You specialise in assessing how emotive language is used to position readers"}]
    emotive_list = get_emotive_list(article, emo_msgs)
    emotive_rating = get_emotive_rating(emo_msgs)

    pol_msgs = [{"role": "system", "content": "You are an expert on journalism and politics. You specialise in assessing the presence of political bias in articles."}]
    political_list = get_political_bias_list(article, pol_msgs)
    political_rating = get_political_bias_rating(pol_msgs)

    est_msgs = [{"role": "system", "content": "You are an expert on journalism and politics. You specialise in assessing the presence of pro or anti establishment bias in articles."}]
    establishment_list = get_establishment_list(article, est_msgs)
    establishment_bias_rating = get_establishment_bias_rating(est_msgs)

    return {
        'is_article': is_article,
        'emotive_list': emotive_list,
        'emotive_rating': emotive_rating,
        'political_list': political_list,
        'political_rating': political_rating,
        'establishment_list': establishment_list,
        'establishment_bias_rating': establishment_bias_rating
    }
