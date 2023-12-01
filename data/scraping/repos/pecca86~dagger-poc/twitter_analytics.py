import autogen
# from langchain.document_loaders.csv_loader import CSVLoader


class TwitterAnalytics:
    def __init__(self):
        print("TwitterAnalytics")

    def top_tweets(self):
        result = self._retrieve_tweet_data()
        return result
    
    def twitter_data(self):
        # loader = CSVLoader(
        #     file_path="./analytics_data/likes_per_tweet.csv",
        #     csv_args={"delimiter": "|", "quotechar": '"', "fieldnames": ["tweet_id", "tweet_text", "timestamp", "likes"]},
        # )

        # data = loader.load()
        # return data
        return self._retrieve_tweet_data()


    def _retrieve_tweet_data(self):
        """
        Call the twitter API to retrieve the data based on the CSV of tweets.
        After this save the data to a CSV file for further analytics that can be passed on to the tweeter.
        """
        return [
            "hahaa, I love Gin! Especially the first sip of the day!",
            "Gin is the best drink ever! Especially when you drink it with your friends! #luvthattaste #ginislife",
            "I just looooove this recepie! 20cl of gin, some ice, and a slice of lemon! #gin #ginlovers #ginislife",
        ]
    
    def _top_theme_related_tweets(self, theme: str):
        """
        Call twitter API and retrieve n tweets related to the theme.
        Save this to a CSV file for further analytics that can be passed on to the researcher.
        """
        return [
            "This {theme} is so interesting! I just love it!",
            "I just looooove this {theme}!",
        ]
    
    
    def _top_historically_popular_tweets(n: int):
        """
        Historically most successful own tweets.
        Looks at the CSV file of tweets and retrieves the n most successful ones.
        """
        return [
            "hahaa, I love Gin! Especially the first sip of the day!",
            "Gin is the best drink ever! Especially when you drink it with your friends! #luvthattaste #ginislife",
            "I just looooove this recepie! 20cl of gin, some ice, and a slice of lemon! #gin #ginlovers #ginislife",
        ]

    

