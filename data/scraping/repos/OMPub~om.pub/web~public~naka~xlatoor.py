import openai
import json
import os

# Replace 'your_api_key' with your actual OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

with open("list-of-langs.json", "r") as file:
    data = json.load(file)


def filter_data_by_id_range(data, min_id, max_id):
    filtered_data = {}
    for key, value in data.items():
        if min_id <= value["id"] <= max_id:
            filtered_data[key] = value
    return filtered_data

def translate(text, target_language):
    prompt = f"Translate the following English text to {target_language}:\n\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant translator that answers with only the translated text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.6,
    )

    translation = response.choices[0].message.get("content", "").strip()
    return translation


min_id = 21  # Replace with your desired minimum ID
max_id = 100  # Replace with your desired maximum ID

filtered_data = filter_data_by_id_range(data, min_id, max_id)# Iterate over the data and translate the "title" and "explainer" fields

for language_code, language_data in filtered_data.items():
    if language_code != 'en':
        print(language_code + '...')
        target_language = language_data["languageEn"]
        language_data["title"] = translate(language_data["title"], target_language)
        language_data["explainer"] = translate(language_data["explainer"], target_language)

print(json.dumps(data, indent=2))
