import argparse
import requests
from requests.exceptions import RequestException
from bs4 import BeautifulSoup
import openai
import re

# Inputs through command line arguments
parser = argparse.ArgumentParser(description="Process and analyze a given URL and text.")
parser.add_argument('--openai_key', required=True, help="The API key for OpenAI.")
parser.add_argument('--webhook_url', required=True, help="The webhook endpoint URL.")
parser.add_argument('--source', required=True, help="The source account.")
parser.add_argument('--text', required=True, help="The title and URL.")
args = parser.parse_args()

# Use the parsed arguments
openai.api_key = args.openai_key
SLACK_WEBHOOK = args.webhook_url

def handle_chain_of_events(url, text, source):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        cleaned_content = soup.get_text(separator=' ', strip=True)
        if not cleaned_content:
            print("Error: Couldn't extract content from the page.")
            return
        print(cleaned_content)
        openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant, with high quality expertise in molecular biology, biochemistry, and nanotechnology. The response should be in Japanese and following format: \n\n**Title:[English title]** \n\n**Authors:**[Authors list with short affiliation (university or institute name and country)] \n\n**Abstract:**[日本語翻訳されたabstract] \n\n**要点:**[Japanese translation of the important background and key messages of the research]"
                },
                {
                    "role": "user",
                    "content": f"このドキュメントは学術論文の記事で、そのタイトルとURLは{text} です。このドキュメントはWeb PageのHTMLから抽出され、Beautifl Soup4 によって整形されました。ドキュメント中からAbstractを抽出して日本語翻訳してください。また、タイトルと著者名、所属を冒頭に記してください。レスポンスの最後には重要な背景情報とキーメッセージを箇条書きを付すことで、ひとめ見て研究内容が理解できるように工夫してください。 \n\n{cleaned_content}"
                }
            ]
        )
        assistant_message = openai_response['choices'][0]['message']['content']

        print(assistant_message)

        headers_for_slack = {'Content-Type': 'application/json'}
        payload = {
            "text": assistant_message,
            "username": source,
            "icon_emoji": ":robot_face:"
        }
        requests.post(SLACK_WEBHOOK, json=payload, headers=headers_for_slack)

    except RequestException as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    urls = re.findall(url_pattern, args.text)
    print(urls[0])

    handle_chain_of_events(urls[0], args.text, args.source)
else:
    headers_for_slack = {'Content-Type': 'application/json'}
    payload = {
        "text": "Error in __main__",
        "username": "MyBot",
        "icon_emoji": ":robot_face:"
    }
    requests.post(SLACK_WEBHOOK, json=payload, headers=headers_for_slack)
