from praw import Reddit
from os import getenv
import openai
import markovify
from db import DB

REDDIT_USERNAME = 'jrtbot'
REDDIT_PASSWORD = getenv('REDDIT_PASSWORD')   
REDDIT_ID = getenv('REDDIT_ID')
REDDIT_SECRET = getenv('REDDIT_SECRET')
USER_AGENT = 'talkingbot by u/jrtbot'
OPENAI_ORGANIZATION = getenv('OPENAI_ORGANIZATION')
OPENAI_KEY = getenv('OPENAI_KEY')

def gen_nietzsche_talk() -> str:
    openai.organization = OPENAI_ORGANIZATION
    openai.api_key = OPENAI_KEY 
    completion = openai.Completion.create(
        model='text-davinci-002',
        prompt='Write something that sounds like Nietzsche.',
        max_tokens=100,
        temperature=1.0
    )
    return completion['choices'][0]['text'].strip('\n')

def gen_trump_talk() -> str:
    with open('trump.txt') as f:
        trump_text = f.read()
    trump_text_model = markovify.Text(trump_text)
    return trump_text_model.make_short_sentence(max_chars=500)

def main():
    reddit = Reddit(
        username=REDDIT_USERNAME,
        password=REDDIT_PASSWORD,
        client_id=REDDIT_ID,
        client_secret=REDDIT_SECRET,
        user_agent=USER_AGENT
    )
    print('listening for !trumptalk and !nietzschetalk comments')
    db = DB()
    for comment in reddit.subreddit('all').stream.comments():
        if '!nietzschetalk' in comment.body and db.comment_is_new(comment):
            comment.reply(body=gen_nietzsche_talk())
            db.ack_comment(comment)
            print(f'made nietzschetalk comment on {comment.submission.url}{comment.id}')
        elif '!trumptalk' in comment.body  and db.comment_is_new(comment):
            comment.reply(body=gen_trump_talk())
            db.ack_comment(comment)
            print(f'made trumptalk comment on {comment.submission.url}{comment.id}')
    db.close()

if __name__ == '__main__':
    while True:
        try:
            main()
        except Exception as e:
            print(f"talkbot: An exception occured while running: {e}")
            continue