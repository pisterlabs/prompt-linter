import tiktoken
import openai
import newspaper
from langchain.text_splitter import TokenTextSplitter
import re
from youtube_transcript_api import YouTubeTranscriptApi
import requests


def estimate_input_cost_optimized(model_name, token_count):
    model_cost_dict = {
        "gpt-3.5-turbo-0613": 0.0015,
        "gpt-3.5-turbo-16k-0613": 0.003,
        "gpt-4-0613": 0.03,
        "gpt-4-32k-0613": 0.06,
    }

    try:
        cost_per_1000_tokens = model_cost_dict[model_name]
    except KeyError:
        raise ValueError(f"The model '{model_name}' is not recognized.")

    estimated_cost = (token_count / 1000) * cost_per_1000_tokens

    return estimated_cost


def estimate_input_cost(model_name, token_count):
    cost_per_1000_tokens = 0.0015  # default
    if model_name == "gpt-3.5-turbo-0613":
        cost_per_1000_tokens = 0.0015
    if model_name == "gpt-3.5-turbo-16k-0613":
        cost_per_1000_tokens = 0.003
    if model_name == "gpt-4-0613":
        cost_per_1000_tokens = 0.03
    if model_name == "gpt-4-32k-0613":
        cost_per_1000_tokens = 0.06

    estimated_cost = (token_count / 1000) * cost_per_1000_tokens
    return estimated_cost


def count_tokens(text, selected_model):
    encoding = tiktoken.encoding_for_model(selected_model)
    num_tokens = encoding.encode(text)
    return len(num_tokens)


openai.api_key = "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"


def generate_text_with_openai(user_prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # you can replace this with your preferred model
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )
    return completion.choices[0].message.content


max_tokens_per_chunk = 1000


def split_text_into_chunks(text, max_tokens):
    text_splitter = TokenTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    return chunks


def get_article_from_url(url):
    try:
        # Scrape the web page for content using newspaper
        article = newspaper.Article(url)
        # Download the article's content with a timeout of 10 seconds
        article.download()
        # Check if the download was successful before parsing the article
        if article.download_state == 2:
            article.parse()
            # Get the main text content of the article
            article_text = article.text
            return article_text
        else:
            print("Error: Unable to download article from URL:", url)
            return None
    except Exception as e:
        print("An error occurred while processing the URL:", url)
        print(str(e))
        return None


def get_video_transcript(video_url):
    match = re.search(r"(?:youtube\.com\/watch\?v=|youtu\.be\/)(.*)", video_url)
    if match:
        VIDEO_ID = match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")

    video_id = VIDEO_ID

    # Fetch the transcript using the YouTubeTranscriptApi
    transcript = YouTubeTranscriptApi.get_transcript(video_id)

    # Extract the text of the transcript
    transcript_text = ""
    for line in transcript:
        transcript_text += line["text"] + " "
    return transcript_text


def check_domain_authority(domain):
    url = "https://domain-authority1.p.rapidapi.com/GetDomainAuthority"

    querystring = {"url": domain}

    headers = {
        "X-RapidAPI-Key": "a617d6467dmshac84323ce581a72p11caa9jsn1adf8bbcbd47",
        "X-RapidAPI-Host": "domain-authority1.p.rapidapi.com",
    }
    response = requests.get(url, headers=headers, params=querystring)
    data = response.json()
    return data["result"]["domainAuthority"]
