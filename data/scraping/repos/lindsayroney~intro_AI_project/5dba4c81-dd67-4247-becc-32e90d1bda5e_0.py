import os
import requests
import json
import openai

# Set your API keys
TYPEFORM_API_KEY = os.getenv("TYPEFORM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Set your form ID
FORM_ID = "Your_Form_ID"  # replace with your form ID

# Set the Typeform API endpoint
TYPEFORM_API = f"https://api.typeform.com/forms/{FORM_ID}/responses"

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Headers for the Typeform API
headers = {
    "Authorization": f"Bearer {TYPEFORM_API_KEY}",
}

def get_responses(since=None):
    params = {}
    if since:
        params['since'] = since

    response = requests.get(TYPEFORM_API, headers=headers, params=params)
    return response.json()

def get_summarized_points(text):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Summarize the following text into 3 key points"},
            {"role": "user", "content": text}
        ]
    )
    return completion.choices[0].message['content']

def get_classification_groups(responses):
    text = "; ".join(responses)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Come up with 5 phrases that can be used to semantically group the following form responses"},
            {"role": "user", "content": text}
        ]
    )
    return completion.choices[0].message['content']

def main():
    summarized_responses = []
    response_data = get_responses()

    # Loop through pages of responses
    while True:
        for item in response_data['items']:
            text_responses = [answer['text'] for answer in item['answers'] if answer['type'] in ['text', 'short_text', 'long_text']]
            response_text = " ".join(text_responses)
            summarized_response = get_summarized_points(response_text)
            summarized_responses.append(summarized_response)

        if response_data['page_count'] == 1:
            break
        else:
            response_data = get_responses(response_data['items'][-1]['submitted_at'])

    groups = get_classification_groups(summarized_responses)
    print(groups)

if __name__ == "__main__":
    main()
