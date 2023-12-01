import openai
import requests
import base64
import os
from dotenv import load_dotenv

def get_total_pagecount():
 response = requests.get(api_url)
 pages_count = response.headers['X-WP-TotalPages']
 return int(pages_count)

def read_wordpress_posts_with_pagination(api_url):
 total_pages = get_total_pagecount()
 current_page = 1
 all_page_items_json = []
 while current_page <= total_pages:
     api_url = f"{api_url}?page={current_page}&per_page=100"
     page_items = requests.get(api_url)
     page_items_json = page_items.json()
     all_page_items_json.extend(page_items_json)
     current_page = current_page + 1
 return all_page_items_json

def create_wordpress_post(api_url, title, text):
    data = {
        'title' : title.replace('"', ''),
        'status': 'draft',
        #'slug' : 'example-post',
        'content': text
    }
    response = requests.post(api_url,headers=wordpress_header, json=data)

load_dotenv()
wordpress_user = os.getenv('WORDPRESS_USER')
wordpress_password = os.getenv('WORDPRESS_PASSWORD')
wordpress_credentials = wordpress_user + ":" + wordpress_password
wordpress_token = base64.b64encode(wordpress_credentials.encode())
wordpress_header = {'Authorization': 'Basic ' + wordpress_token.decode('utf-8')}
openai.api_key = os.getenv('API_KEY')
api_url = os.getenv('API_URL')

# Generate 6 potential topics on health, fitness, bodybuilding, nutrition, and supplements
prompt = "Give me a potential topic for a blog article on health, fitness, bodybuilding, nutrition, or supplements."

response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    max_tokens=100,
    temperature=1.2,
    n=2
)

# Request blog articles based on the 6 topic ideas
for topic in response['choices']:
    print(topic['text'])
    prompt = "Please write a concise, professional blog article, with scholarly references, and at least 150 words for the title {}".format(topic["text"])
    print(prompt)

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=1024,
    )

    articles = response["choices"]

    # Print the blog articles
    for article in articles:
        print(article["text"])
        create_wordpress_post(api_url, topic["text"], article["text"])
