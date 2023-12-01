import config
import requests
import openai
import random
from requests.auth import HTTPBasicAuth

# News API
api_key = config.newsapi_key

response = requests.get(
    f"https://newsapi.org/v2/top-headlines?country=us&pageSize=5&apiKey={api_key}")
news = response.json()

# OpenAi API
openai.organization = config.openai_organizationID
openai.api_key = config.openai_key
openai.Model.list()

# Spotify API
user = config.spotify_clientID
password = config.spotify_clientSecret

url = "https://accounts.spotify.com/api/token"
data = {"grant_type": "client_credentials"}
auth = HTTPBasicAuth(user, password)

response = requests.post(url, data=data, auth=auth)
accessToken = response.json()["access_token"]

url = "https://api.spotify.com/v1/search"
headers = {"Authorization": f"Bearer {accessToken}"}

# Used for get random songs on each query, and not allways the same ones
offset = random.randint(0, 100)

for new in news["articles"]:
    # Get the url of a news
    news_url = new["url"]

    prompt = f"summarize:{news_url} in two words"

    # Send the url to the AI to summarize
    response_ai = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=10,
        temperature=0
    )

    # Get the result of the summarize
    words = response_ai["choices"][0]["text"].strip()

    # Spotify
    search_query = f"?q={words}&type=track&limit=1&offset={offset}"

    # Get random songs using the words
    response_spotify = requests.get(f"{url}{search_query}", headers=headers)
    song = response_spotify.json()

    song_name = song["tracks"]["items"][0]["name"]
    song_url = song["tracks"]["items"][0]["external_urls"]["spotify"]

    print(f"Song name: {song_name}")
    print("Words used:", words)
    print(song_url)
    print()
