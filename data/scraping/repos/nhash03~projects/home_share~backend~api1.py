import cohere
import pandas as pd
import requests
import datetime
import re
pd.set_option('display.max_colwidth', None)
api_key = 'lxr3K9uC48BQIpPFfAVwyxtfJdy7uQPxXhPPkkxp'
co = cohere.Client(api_key)


def get_post_titles(**kwargs):
    """ Gets data from the pushshift api. Read more: https://github.com/pushshift/api """
    base_url = f"https://api.pushshift.io/reddit/search/submission/"
    payload = kwargs
    request = requests.get(base_url, params=payload)
    return [a['title'] for a in request.json()['data']]


examples = [
    ("The New York Stock Exchange, located in New York City, is the largest stock exchange in the world by market capitalization, with a value of over $25 trillion.", "New York City"),
    ("In Tokyo, the world's busiest pedestrian crossing, Shibuya Crossing, sees over 1,000 people cross at once during peak times.", "Tokyo"),
    ("Mexico City, one of the world's most populous cities, has over 21 million inhabitants in its metropolitan area.", "Mexico City"
     ), ("The population of Moscow, Russia, has grown from just over six million in 1980 to over 12 million in 2021.", "Moscow"),
    ("Los Angeles, known for its heavy traffic, has the longest commute time in the United States, with an average of over 29 minutes.", "Los Angeles"),
    ("The Empire State Building is one of the most famous skyscrapers in the world, located in New York City.", "New York City"),
    ("The Louvre Museum in Paris houses some of the most famous artworks in history.", "Paris"),
    ("Tokyo is the most populous city in Japan and one of the busiest cities in the world.", "Tokyo"),
    ("Barcelona is known for its stunning architecture, including works by Antoni Gaudí.", "Barcelona"),
    ("The CN Tower in Toronto is one of the tallest free-standing structures in the world.", "Toronto"),
    ("Berlin is a city rich in history and culture, with many museums and historical landmarks.", "Berlin"),
    ("The Petronas Towers in Kuala Lumpur are twin skyscrapers that dominate the city's skyline.", "Kuala Lumpur"),
    ("Rome is home to many historic landmarks, including the Pantheon and the Colosseum.", "Rome"),
    ("Dublin is the capital of Ireland and known for its vibrant music and pub scene.", "Dublin"),
    ("The Space Needle in Seattle is a popular tourist attraction with stunning views of the city.", "Seattle"),
    ("There is a beautiful park in Kiev right now", "Kiev")

]


class cohereExtractor():
    def __init__(self, examples, example_labels, labels, task_desciption, example_prompt):
        self.examples = examples
        self.example_labels = example_labels
        self.labels = labels
        self.task_desciption = task_desciption
        self.example_prompt = example_prompt

    def make_prompt(self, example):
        examples = self.examples + [example]
        labels = self.example_labels + [""]
        return (self.task_desciption +
                "\n---\n".join([examples[i] + "\n" +
                                self.example_prompt +
                                labels[i] for i in range(len(examples))]))

    def extract(self, example):
        extraction = co.generate(
            model='xlarge',
            prompt=self.make_prompt(example),
            max_tokens=4,
            temperature=0.01,
            stop_sequences=["\n"])
        return(extraction.generations[0].text[:-1])


cohereCityExtractor = cohereExtractor([e[0] for e in examples],
                                      [e[1] for e in examples], [],
                                      "",
                                      "extract the city title from the post:")
# print(cohereCityExtractor.make_prompt(
#     'I live in Vancouver which is a nice place'))


def doIt(txt):
    try:
        extracted_text = cohereCityExtractor.extract(txt)
        result = (extracted_text)
        return result
    except Exception as e:
        print('ERROR: ', e)


def findInts(txt):
    return int(re.search(r'\d+', txt).group())


tests = "I am from Isfahan which is a beauiful city .I live with 4 other people"
results = doIt(tests)
result2 = findInts(tests)
print(results)
print(result2)
