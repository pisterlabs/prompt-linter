import openai, tweepy, random

class TwitterBot():

    api_key  ="YOUR API KEY FROM TWITTER"
    api_key_secret ="YOUR API KEY SECRET FROM TWITTER"
    access_token="YOUR ACCESS TOKEN FROM TWITTER"
    access_token_secret="YOUR ACCESS TOKEN SECRET FROM TWITTER"

    
    #OpenAI
    openai.api_key = "OPEN AI KEY"

    # Connecting to Twitter
    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    # Creating Tweets
    prompts = [
        {
            "hashtag": "#meme",
            "text": "post a meme on twitter bots"
        },
        {
            "hashtag": "#technology",
            "text": "tweet something cool for #technology"
        },
        {
            "hashtag": "@kachkol_asa",
            "text": "tweet something cool for @kachkol_asa from twitter"
        },
    ]


    def __init__(self):
        error = 1
        while(error == 1):
            tweet = self.create_tweet()
            try:
                error = 0
                self.api.update_status(tweet)
            except:
                error = 1
    
    def create_tweet(self):
        chosen_prompt = random.choice(self.prompts)
        text = chosen_prompt["text"]
        hashtags = chosen_prompt["hashtag"]

        response = openai.Completion.create(
            engine="text-davinci-001",
            prompt=text,
            max_tokens=64,
        )

        tweet = response.choices[0].text
        tweet = tweet + " " + hashtags
        return tweet


twitter = TwitterBot()
twitter.create_tweet()

