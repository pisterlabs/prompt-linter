import re,time,os
import tweepy
import openai

auth = tweepy.OAuthHandler(os.environ['CONSUMER_KEY'], os.environ['CONSUMER_SECRET'])
auth.set_access_token(os.environ['ACCESS_TOKEN'], os.environ['ACCESS_TOKEN_SECRET'])
openai.api_key = os.environ["OPENAI_API_KEY"]

hashtag = '#' + os.environ["HASHTAG"]
replied = []
toreply = []

api = tweepy.API(auth, wait_on_rate_limit=True)

while True:
    input = tweepy.Cursor(api.search_tweets, q=hashtag, lang='en').items(1)
    tweetid = 0
    prompt = ''
    for i in input:
        if i.id not in replied:
            replied.append(i.id)
        prompt = i.text
        tweetid = i.id

    if tweetid not in toreply:
        cleaned_prompt = prompt.strip()
        for i in range(len(cleaned_prompt)):
            if cleaned_prompt[i] == '#':
                hash = cleaned_prompt[i:i+13]
                break
        cleaned_prompt = cleaned_prompt.replace(hash, '')

        def generateResponse():
            response = openai.Completion.create(
            engine="davinci-instruct-beta-v3",
            prompt=cleaned_prompt,
            temperature=0.5,
            max_tokens=500,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            )
            return response['choices'][0]['text'].strip()

        reply = generateResponse()
        api.update_status(status = reply[0:279], in_reply_to_status_id = tweetid, auto_populate_reply_metadata=True)
        toreply.append(tweetid)
    else:
        time.sleep(5) # wait 5 seconds before checking again
        continue
