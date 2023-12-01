import re
from models.event_model import Event
from models.tweet_model import TweetEvent
import openai

def process_batch_tweets(contents,publish_dates,parser,prompt,gpt_list):
    tweet_contents = [get_twitter_contents(content.extract) for content in contents]

    _inputs = [prompt.format_prompt(tweet=content['tweet'],publish_date=publish_date,preference="tech").to_string() for content,publish_date in zip(tweet_contents, publish_dates)] 
    # TODO: Add dynamic preference

    # TODO: Convert this to the ChatCompletionAPI, which is the new API for GPT-3
    responses = openai.Completion.create(
        temperature=0,
        model="gpt-3.5-turbo-instruct",
        prompt=_inputs,
        max_tokens=2000,
    )

    # ChatGPT 3's api indexing is weird, so we have to do this to get the responses in the right order
    events = [""] * len(_inputs)
    for choice in responses.choices:
        events[choice.index] = choice.text
    
    # Parse the event descriptions and create TweetEvent objects
    for (tweet_content,event,content) in zip(tweet_contents,events,contents):
        try:
            output = parser.parse(event)
            output = TweetEvent.fromEvent(event = output, retweet_count = tweet_content['retweets'], favorite_count = tweet_content['favorites'], reply_count = tweet_content['replies']).dict()
            output['url'] = content.url
            output['source'] = 'Twitter'
            output['twitter_id'] = get_twitter_id(content.url)
            gpt_list.append(output)
        except Exception as e:
            #TODO: Add logging
            print(f"Error Parsing Request:{e}")
            pass

# This function extracts the tweet text and counts from a tweet HTML extract
def get_twitter_contents(extract: str):
    tweet = extract.split('| created_at')[0].replace("<div>", "", 1)
    retweet_count = int(re.search(r'retweet_count: (\d+)', extract).group(1))
    reply_count = int(re.search(r'reply_count: (\d+)', extract).group(1))
    favorite_count = int(re.search(r'favorite_count: (\d+)', extract).group(1))
    return {'tweet': tweet, 'retweets': retweet_count, 'replies': reply_count, 'favorites': favorite_count}

def get_twitter_id(url: str):
    match = re.search(r'/status/(\d+)', url)
    if match:
        return match.group(1)
    else:
        return None
