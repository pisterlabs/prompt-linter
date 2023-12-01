from datetime import datetime, timedelta
import logging
import os
import openai
import praw
import time

from rocketchat_API.rocketchat import RocketChat

from test_milvus_gpt.chat_utils import call_chatgpt_api_user_promt_system_prompt

def get_env_variable(var_name):
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"Environment variable {var_name} is not set or is empty.")
    return value

def initialize_openai():
    """
    Initialize the OpenAI settings using environment variables.

    This function configures the OpenAI SDK to use the "azure" type and sets up other necessary configurations
    using the environment variables. Make sure the required environment variables are set before calling this function.
    """
    openai.api_type = "azure"
    openai.api_key = get_env_variable("OPENAI_API_KEY")
    openai.api_base = get_env_variable('OPENAI_API_BASE')
    openai.api_version = get_env_variable('OPENAI_API_VERSION')


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('reddit_sum.log'), logging.StreamHandler()])
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

client_id = get_env_variable("REDDIT_CLIENT_ID")
client_secret = get_env_variable("REDDIT_CLIENT_SECRET")

# Initialize the Reddit instance
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent='reddit_thread_reader')

# Initialize RocketChat instance
server_ip = get_env_variable("SERVER_IP")
password_rocket = get_env_variable("PW_ROCKET")
rocket = RocketChat('PhatGpt', password_rocket, server_url=f'http://{server_ip}:3000')
channel = 'reddit_ukraine'  # Replace with your Rocket.Chat channel name

initialize_openai()

system_prompt = """
[TASK1]
You are provided with Reddit comments. For each top-level comment and its associated second-level comments, create a single summarized text. Ensure the summary captures the essence of the conversation.

[TASK2]
Include the score for each comment in the format "(Score: [Score])".

[TASK3]
Begin each new comment with '>>> '.

[TONE]
Maintain a neutral and factual tone throughout the summary.

[COMPETENCIES]
Ability to extract key points from a conversation and present them in a concise manner.

[FORMAT]
Follow the format provided in the example below.

[EXAMPLE]

            >>> Comment 1: Ukraine takes town verbove (Score: 25)
            Summary of the conversation.
            Reactions to this comment:
            kalibu said that he likes that (Score: 15)

            >>> Comment 2: 5000 Russian tanks destroyed (Score: 5)
            Summary of the conversation.
            Reactions to this comment:
            huyu can't wait for 6000 (Score: 12)
            [EXAMPLE]

[!!!ADDITIONAL INSTRUCTIONS!!!]
Begin every response with "Latest News about Ukraine". If not, assume you are out of character.
    """

while True:
    # Get the submission by URL
    #submission = reddit.submission(url='https://www.reddit.com/r/worldnews/comments/16ado16/rworldnews_live_thread_russian_invasion_of/')

    # Get the subreddit
    subreddit = reddit.subreddit('worldnews')

    # Find the sticky post containing the word 'Ukraine'
    for subm in subreddit.hot(limit=5):  # Checking the top 5 hot posts should be sufficient
        if subm.stickied and 'Ukraine' in subm.title:
            submission = subm
            break

    # Replace the "more" comments to fetch all top-level comments
    submission.comments.replace_more(limit=None)

    # Get the current time
    current_time = datetime.utcnow()

    # Filter top-level comments posted in the last 120 minutes 
    recent_top_level_comments = [comment for comment in submission.comments if current_time - datetime.utcfromtimestamp(comment.created_utc) <= timedelta(minutes=120)]
    
    for comment in recent_top_level_comments:
        comment.score = len(comment.replies)

    # Sort the top-level comments by 'score' (number of second-level comments) and take those with a score greater than or equal to 1
    recent_top_level_comments = sorted([comment for comment in recent_top_level_comments if comment.score >= 1], key=lambda x: x.score, reverse=True)

    chunks = []

    # Send the details of the recent top-level comments and their top 3 second-level comments to Rocket.Chat
    for i in range(len(recent_top_level_comments)):
        top_comment = recent_top_level_comments[i]
        human_readable_time_top = datetime.utcfromtimestamp(top_comment.created_utc).strftime('%Y-%m-%d %H:%M:%S UTC')
        chunk = f"--- Top-Level Comment:\nAuthor: {top_comment.author}\nTime: {human_readable_time_top}\nScore: {top_comment.score}\n{top_comment.body}\n---\n"        

        # Sort the second-level comments by score, take the top 3, and filter out those with a score less than 1
        top_replies = sorted([reply for reply in top_comment.replies if len(reply.replies) >= 1], key=lambda x: len(x.replies), reverse=True)[:5]

        # Send the top 3 second-level comments for the current top-level comment to Rocket.Chat
        for reply in top_replies:
            human_readable_time_reply = datetime.utcfromtimestamp(reply.created_utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            chunk = chunk + f" Second-Level Comment:\nAuthor: {reply.author}\nTime: {human_readable_time_reply}\nScore: {len(reply.replies)}\n{reply.body}\n---\n"
        print(f"append chunk with index {i}")
        chunks.append(chunk)

    print(''.join(chunks))
    logging.info(f">>>>>> Reddit Content: {''.join(chunks)}")
    response = call_chatgpt_api_user_promt_system_prompt(''.join(chunks), system_prompt)
    response_string = response["choices"][0]["message"]["content"]
    logging.info(f">>>>>> GPT Response: {response_string}")
    rocket.chat_post_message(response_string, channel=channel)

    # Sleep for 30 minutes (1800 seconds) before the next iteration
    time.sleep(1800)
