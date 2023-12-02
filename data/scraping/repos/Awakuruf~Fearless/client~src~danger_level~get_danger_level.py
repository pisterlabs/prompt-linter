import cohere
from examples import examples
# from inputs import inputs
import requests

co = cohere.Client('Ex5i3BxYcOXkbj5oVSRB7EC0ne8GQT9lLFBxgE4D')
headers = {"x-api-key": "U9QGGZcVtlzGpN5wcH-QyL-_bh6t_042FIVBJAC8-UE"}

# uses https://app.newscatcherapi.com/dashboard/


def get_danger_level(query):
  url = "https://api.newscatcherapi.com/v2/search"

  querystring = {
    "q": f"{query} AND vancouver",
    "lang": "en",
    "sort_by": "relevancy",
    "page": "1"
  }

  response = requests.request("GET", url, headers=headers, params=querystring)

  titles = []
  if response.status_code == 200:
    data = response.json()
    seen = set()
    for article in data['articles']:
      title = article['title']
      if title not in seen:
        titles.append(title)
        seen.add(title)
        if len(titles) == 15:
          break
  else:
    print(f"Error {response.status_code}: {response.reason}")

  response = co.classify(model='large', inputs=titles, examples=examples)

  toxic_sum = 0

  for classification in response:
    prediction = classification.prediction
    confidence = classification.confidence
    if prediction == 'Benign':
      toxic_sum += 1 - confidence
    else:
      toxic_sum += confidence

  print(titles)
  danger_level = toxic_sum / len(response)
  return danger_level
