import requests
import json
import openai
import config

# News API
api_key = config.newsapi_key
response = requests.get(
    f"https://newsapi.org/v2/top-headlines?country=us&pageSize=5&apiKey={api_key}")
news = response.json()

# OpenAi API
openai.organization = config.openai_organizationID
openai.api_key = config.openai_key
openai.Model.list()

# See the JSON returned
# print(json.dumps(news, indent = 2))
print("Five random news from today")

for new in news["articles"]:
    news_url = new["url"]

    prompt = f"summarize:{news_url} in one sentence"

    response_ai = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=30,
        temperature=0
    )

    print()
    print(response_ai["choices"][0]["text"].strip())
