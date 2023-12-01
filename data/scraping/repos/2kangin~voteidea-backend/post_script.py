from openai import OpenAI
import requests
import os
import random
import time

API_ENDPOINT = os.environ.get("API_ENDPOINT", "http://localhost:8080")

# API 키 설정
OPENAI_YOUR_KEY = os.environ.get("OPENAI_KEY", "")
client = OpenAI(api_key=OPENAI_YOUR_KEY)

def get_recent_ideas(category_id):
    response = requests.get(API_ENDPOINT + "/ideas?categoryId=" + category_id + "&size=10", verify=False)
    if response.status_code == 200:
        data = response.json()
        # Extracting the list of categories from the nested structure
        ideas = data.get('data', {}).get('items', [])
        return ideas

# 영어로 아이디어 생성 함수
def post_idea(category_name, category_id):
    print("Category Name: " + category_name + ", Category ID: " + category_id)
    english_ideas = generate_ideas_in_english(category_name)
    korean_ideas = translate_to_korean(category_name, english_ideas)

    idea = {
        "content": {"en": english_ideas, "ko": korean_ideas},
        "title": {"en": "", "ko": ""},
        "categoryId": category_id,
        "createdBy": "GPT4"
    }

    success_count = 0
    failure_count = 0

    response = requests.post(
        API_ENDPOINT + "/ideas",
        headers={"Content-Type": "application/json"},
        json=idea,
        verify=False
    )

    if response.status_code == 200:
        success_count += 1
    else:
        print(response.status_code, response.text)
        failure_count += 1

    print(idea)
    print("Successfully inserted items: ", success_count)
    print("Failed to insert items: ", failure_count)

def generate_ideas_in_english(category_name):
    existing_ideas = get_recent_ideas(category['id'])
    context = ""
    for idea in existing_ideas:
        context += idea['content']['en'] + "\n"

    completion = client.chat.completions.create(model="gpt-3.5-turbo",
    max_tokens=200,
    temperature= random.uniform(0.5, 1.5),
    messages=[
        {"role": "system", "content": "You are a content writer very proficient idea generator."},
        {"role": "user", "content": "Generate 1 concise, practical, and appealing idea for " + category_name + " within 20 words in English."
         + "Write The Unique idea In Your Own Words Rather Than Copying And Pasting From Other Sources."
         + "Write In A Conversational Style As Written By A Human (Use An Informal Tone, Utilize Personal Pronouns, Keep It Simple, Engage The Reader, Use The Active Voice, Keep It Brief, Use Rhetorical Questions, and Incorporate Analogies And Metaphors)"
         + "The content in the triple quotation marks are the 10 answers you've already suggested. Avoid duplicating the content. " + "\"\"\"" + context + "\"\"\""
         },
    ])
    return completion.choices[0].message.content

# 영어 텍스트를 한국어로 번역하는 함수
def translate_to_korean(category_name, text):
    completion = client.chat.completions.create(model="gpt-3.5-turbo",
    max_tokens=200,
    messages=[
        {"role": "system", "content": "당신은 OpenAI 모델에 의해 생성된 영어 아이디어를 한국어 사용자에게 자연스럽고 매력적으로 다가갈 수 있도록 재해석하는 전문가입니다."},
        {"role": "user", "content": category_name + " 카테고리에 대한 원문을 재해석한 내용만을 20 단어 정도로 정리해서 주세요. 다음은 원문입니다: \"" + text + "\"" },
    ])
    return completion.choices[0].message.content

# Load categories
response = requests.get(API_ENDPOINT + "/categories?size=300", verify=False)

# Ensure the request was successful
if response.status_code == 200:
    data = response.json()
    # Extracting the list of categories from the nested structure
    categories = data.get('data', {}).get('items', [])
else:
    print(f"Failed to fetch data from API. Status code: {response.status_code}")
    categories = []

def translate_to_korean_deepl(text):
    API_ENDPOINT = 'https://api-free.deepl.com/v2/translate'
    AUTH_KEY = '934503c4-14a6-672a-75c0-d082ef132907:fx'
    headers = {
        'Authorization': f'DeepL-Auth-Key {AUTH_KEY}'
    }
    data = {
        'text': text,
        'target_lang': 'KO'
    }

    response = requests.post(API_ENDPOINT, headers=headers, data=data)

    if response.status_code == 200:
        json_data = response.json()
        translated_text = json_data['translations'][0]['text']
        return translated_text
    else:
        print(f"Error: {response.status_code}")
        return None


for category in categories:
    for _ in range(1):  # 1회 반복
        if category['createdBy'] != 'GPT-4':
            continue
        post_idea(category['name']['en'], category['id'])
        time.sleep(1)

