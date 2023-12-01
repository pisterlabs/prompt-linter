import asyncpraw
import json
import zstandard as zstd
import datetime
import os
import sys

import openai

import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')

import string

from langdetect import detect

import asyncio
import aiofiles

from tqdm.asyncio import tqdm_asyncio
from tqdm import tqdm

# Gather credentials from config
with open('config.json') as f:
    config = json.load(f)

client_id = config['reddit_client_id']
client_secret = config['reddit_client_secret']
user_agent = config['reddit_user_agent']
username = config['reddit_username']
password = config['reddit_password']
openai_api_key = config['openai_api_key']

# Authenticate OpenAI API
openai.api_key = openai_api_key

time_filter = 'all'  # Can be one of: 'hour', 'day', 'week', 'month', 'year', 'all'

# Set the number of posts to grab
num_posts = 500
# Set the number of comments to retrieve
k = 5
# Set the minimum upvotes
min_upvotes = 1000

# Convert date to ignore data after to Unix timestamp
date_obj = datetime.datetime.strptime('2022-10-31', "%Y-%m-%d")
unix_timestamp = int(date_obj.timestamp())

# Define the subreddits you want to scrape
subreddit_names = sorted(input('Enter subreddits (space separated): ').split())
# explainlikeimfive askscience askhistorians

# Set the maximum number of requests allowed per minute
max_requests_per_minute = 3000

# Counter for tracking the number of requests made
request_counter = 0

class ProgressBar:
    def __init__(self, total):
        self.pbar = tqdm(total=total)

    def update(self, value):
        self.pbar.update(value)

    def close(self):
        self.pbar.close()


async def process_post(post_data, pbar):
    global request_counter

    retry_attempts = 5
    for _ in range(retry_attempts):
        try:
            if request_counter >= max_requests_per_minute:
                print("Reached maximum requests per minute. Waiting for 1 minute...")
                await asyncio.sleep(60)
                request_counter = 0

            # Generate response using GPT-3.5 API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a frequent user of the subreddits {' '.join(subreddit_names)}. Answer anything relevant."},
                    {"role": "user", "content": post_data['title']}
                ],
                temperature=0.7,
                max_tokens=100, 
                timeout=10
            )
            generated_response = response.choices[0].message.content

            sentences = sent_tokenize(generated_response)
            complete_sentences = [sentence for sentence in sentences if sentence.endswith('.')]
            post_data['gpt'] = ' '.join(complete_sentences)

            request_counter += 1

            # If everything succeeded, break out of the retry loop
            pbar.update(1)
            return post_data

        except openai.error.RateLimitError as e:
            print(f"GPT rate limit exceeded. Waiting for 1 minute...")
            await asyncio.sleep(60)
        except openai.error.APIError as e:
            print(f"Error occurred: {e}. Retrying in 1 minute...")
            await asyncio.sleep(60)
        except asyncio.exceptions.TimeoutError as e:
            print('Timeout error. Retrying in 1 minute...')
            await asyncio.sleep(60)

    print(f"Exceeded maximum retry attempts for post: {post_data['title']}")
    pbar.update(1)
    return None if post_data['gpt'] != '' else post_data



async def process_posts(posts, pbar):

    tasks = [asyncio.create_task(process_post(post_data, pbar)) for post_data in posts]

    results = []
    for task in tasks:
        value = await task
        results.append(value)

    return results


async def process_subreddit(subreddit):
    top_posts = []
    # Counter for the number of filtered posts
    count = 0

    retrieved_post_ids = set()

    # Initial number of posts pulled in each iteration
    num_posts_per_search = num_posts

    pbar = ProgressBar(num_posts)

    # Continue until desired number of posts or number of posts searched is too large
    while count < num_posts and num_posts_per_search < 10000:
        async for post in subreddit.top(time_filter=time_filter, limit=num_posts_per_search):

            if count >= num_posts:
                break

            # If this post has been seen before
            if post.id in retrieved_post_ids:
                continue

            # Get the top posts that satisfy criteria below
            # print(post.title)
            # print(post.created_utc < unix_timestamp, post.author, detect(post.title), post.over_18, post.score, '\n')
            if post.score > min_upvotes \
                    and post.created_utc < unix_timestamp \
                    and not post.over_18 \
                    and detect(post.title) == 'en' \
                    and post.author is not None \
                    and '?' in post.title \
                    and len(post.title) > 1:
                
                post_data = {
                    'title': post.title,
                    'score': post.score,
                    'subreddit': post.subreddit.display_name,
                    'post_id': post.id,
                    'gpt': '', 
                    'comments': []
                }

                await post.load()

                # No comments to gather, so we don't want the post
                if not post.comments:
                    continue
                
                await post.comments.replace_more(limit=0)
                comments = post.comments.list()
                comments_sorted = sorted(comments, key=lambda comment: getattr(comment, 'score', 0), reverse=True)

                comments_stored = 0
                for comment in comments_sorted:
                    try:
                        if detect(comment.body) == 'en' and comment.author is not None and len(comment.body) > 1:
                            comment_data = {
                                'score': comment.score,
                                'body': comment.body
                            }
                            post_data['comments'].append(comment_data)
                            comments_stored += 1
                            # If we have stored k comments
                            if comments_stored >= k:
                                break
                    except:
                        print('Encountered invalid comment body, skipping to next comment...')
                        continue
                
                top_posts.append(post_data)
                retrieved_post_ids.add(post.id)
                count += 1
                pbar.update(1)
        num_posts_per_search *= 2
        
        # await asyncio.sleep(0.05)
    print(f'Gathered {len(top_posts[:num_posts])} posts from {subreddit.display_name}')
    pbar.close()
    return top_posts[:num_posts]


async def write_data_to_file(file_path, data):
    compressed_data = zstd.compress(json.dumps(data).encode('utf-8'))

    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(compressed_data)


async def main():

    # Authenticate using your Reddit account credentials
    reddit = asyncpraw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        username=username,
        password=password, 
        timeout=60
    )

    subreddits = [await reddit.subreddit(name) for name in subreddit_names]

    tasks = []
    for subreddit in subreddits:
        task = asyncio.create_task(process_subreddit(subreddit))
        tasks.append(task)

    print(f'\nGathering top {num_posts} posts from each subreddit which satisfy criteria...')

    top_posts = []
    for task in asyncio.as_completed(tasks):
        result = await task
        top_posts.extend(result)
    
    print(f'\nRetrieving top {k} comments from each post and GPT response...')

    pbar = ProgressBar(len(top_posts))

    # Process the posts and retrieve the data
    results = None
    try:
        results = await process_posts(top_posts, pbar)
    except Exception as e:
        if len(results):
            print(f'Encountered exception: {str(e)}. Writing existing {len(results)} posts...')
        else:
            print(f'Encountered exception: {str(e)}')
            await reddit.close()
            return
    
    pbar.close()

    # Create dataset folder if it doesn't exist
    if not os.path.exists('datasets/reddit_datasets'):
        os.makedirs('datasets/reddit_datasets')

    print('\nWriting data to compressed file...')
    subreddit_name_string = '+'.join(subreddit_names)
    file_path = f'datasets/reddit_datasets/{subreddit_name_string}_{date_obj.date()}_top-{k}-comments_json.zst'
    await write_data_to_file(file_path, results)

    await reddit.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"An error occurred: {e}")

