import cohere
from cohere.classify import Example
import requests
import more_itertools
import os
from dotenv import load_dotenv

load_dotenv()

co = cohere.Client(os.getenv("COHERE_API_KEY"))
twitter_api_url = "https://api.twitter.com/2/tweets/search/recent"
twitter_headers = {
    "Authorization": "Bearer {}".format(
        os.getenv("TWITTER_BEARER_TOKEN"))
}
request_params = {
    'query': '(React.js OR Next.js.js OR Angular.js OR Vue OR node.js OR ember.js) lang:en',
    'max_results': 100,
}


class AnalysisBot():
    retrieved_tweets = []
    classified_tweets = []
    results = {
        'react': {
            'positive': 0,
            'mentions': 0
        },
        'next': {
            'positive': 0,
            'mentions': 0
        },
        'angular': {
            'positive': 0,
            'mentions': 0
        },
        'vue': {
            'positive': 0,
            'mentions': 0
        },
        'node': {
            'positive': 0,
            'mentions': 0
        },
        'ember': {
            'positive': 0,
            'mentions': 0
        }
    }

    def __init__(self) -> None:
        fetch_count = 0

        while (fetch_count < 100):
            print(
                'FETCHING BACTH: #{}. {} tweets retrieved.'
                .format(fetch_count, len(self.retrieved_tweets))
            )
            tweets = requests.request(
                "GET",
                url=twitter_api_url,
                headers=twitter_headers,
                params=request_params
            ).json()

            token = tweets['meta']['next_token']

            if (token):
                request_params["next_token"] = token

            for item in tweets['data']:
                self.retrieved_tweets.append(item['text'])

            fetch_count = fetch_count + 1

    def classify_tweets(self):
        tweet_items = list(more_itertools.chunked(self.retrieved_tweets, 32))

        for idx, tweets in enumerate(tweet_items):
            print("PROCESSING:", idx)
            response = co.classify(
                model='medium',
                taskDescription='Classify tweets on JavaScript frameworks retrieved from the Twitter V2 Search API',
                outputIndicator='Classify retrieved tweets to determine developers stance on framework',
                inputs=tweets,
                examples=[
                    # positive tweets about frameworks
                    Example("React is still the best JS front-end library/framework", "positive review"),
                    Example("Vue is an amazing beginner framework. I've become competent in it, and the goal initially was to learn Vue then move to React... but I don't even know if I want to move to React anymore. Vue is great!",
                           "positive review"),
                    Example("Angular is such a great framework. My Front-End code has never been this tidy and organized. Google did a good job.", "positive review"),

                    # negative tweets about frameworks
                    Example("React/Redux is a bad framework choice for a complex application AND if you are developing w/ React/Redux < 6 months per year.",
                            "negative review"),
                    Example("Had a play with ember.js this morning. The code is not nice, the documentation horrific, and it does bad things to my markup.", "negative review"),

                    # neutral tweets about frameworks
                    Example("I dont know nodejs! I dont know javascript! I dont even know what my nodejs version is! lol",
                            "neutral review"),
                    Example("I dont know much about nodejs but I have to learn it, such is life.", "neutral review"),
                    Example("i Really want to use #NextJs but i dont know much about next and #react yet.", "neutral review")
                ]
            )

            for classification in response.classifications:
                self.determine_results('react', classification)
                self.determine_results('next', classification)
                self.determine_results('vue', classification)
                self.determine_results('angular', classification)
                self.determine_results('node', classification)
                self.determine_results('ember', classification)

        print("RESULTS:", self.results)

    def determine_results(self, framework, classification ):
        if (framework in classification.input):
            self.results[framework]['mentions'] = self.results[framework]['mentions'] + 1

            if (classification.prediction == "positive review"):
                self.results[framework]['positive'] = self.results[framework]['positive'] + 1

classifyObj = AnalysisBot()

classifyObj.classify_tweets()