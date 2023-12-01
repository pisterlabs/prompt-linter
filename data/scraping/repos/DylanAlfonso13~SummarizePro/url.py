import requests
import openai
from bs4 import BeautifulSoup


def grabText(url):
    # Have the limit at 10000 characters as of right now
    max_test_size = 10000
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    main_tag = soup.find('main')
    if main_tag is None:
        raise Exception('Unable to locate main content for this link')
    text = main_tag.get_text(separator=" ")
    text = text[:max_test_size]
    return text


def gen_summary(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
                "content": f"Summarize the following article:\n{text}"},
            {"role": "assistant", "content": "Summary:"}
        ],
        temperature=0.5,
    )
    # Pulls just the summary
    summary = response.choices[-1].message.content.strip()
    return summary
