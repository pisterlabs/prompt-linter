import requests
from datetime import datetime, timedelta
import dateutil.parser
import pytz
import openai

GITHUB_TOKEN = "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"
GITHUB_API_URL = "https://api.github.com"
GITHUB_REPOSITORY = "mikezamayias/healpen"
NOTION_API_URL = "https://api.notion.com/v1"
NOTION_TOKEN = "secret_ucvdiN832KtFsGb6WPgXaKQOqUv5ulTDTy3ovCtrHcg"
NOTION_DATABASE_ID = "e6bd47eb50a7471c86a3343728c42081"
OPENAI_API_KEY = "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"

# Set your OpenAI API key
openai.api_key = OPENAI_API_KEY

github_headers = {"Authorization": f"token {GITHUB_TOKEN}"}
notion_headers = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json",
}


def get_database_entries():
    response = requests.post(
        f"{NOTION_API_URL}/databases/{NOTION_DATABASE_ID}/query",
        headers=notion_headers,
    )
    pages = response.json().get("results", [])
    entries = {}
    for page in pages:
        date_property = (
            page.get("properties", {}).get("Date", {}).get("date", {}).get("start", "")
        )
        date = dateutil.parser.parse(date_property)
        entries[date.strftime("%Y-%m-%d")] = page["id"]
    return entries


def generate_summary(commit_messages):
    prompt = "Summarize the work done in these commits in a concise and straightforward way. The summary should be no more than 18 words and in Greek language. Ensure accuracy of spelling, grammar, and punctuation, while preserving the original meaning and length of the text. Speak with a confident, formal, and friendly tone."
    chat_message = {
        "role": "system",
        "content": "\n".join(commit_messages) + f"\n\n{prompt}",
    }

    response = openai.ChatCompletion.create(
        model="gpt-4", messages=[chat_message], max_tokens=100, temperature=0.1
    )

    summary = response.choices[0].message.content.strip()
    return summary


def create_database_entry(date, commits):
    if not commits:
        payload = {
            "parent": {"database_id": NOTION_DATABASE_ID},
            "properties": {
                "title": {"title": [{"text": {"content": "Απουσία"}}]},
                "Date": {"date": {"start": date.strftime("%Y-%m-%d")}},
            },
        }
    else:
        commit_urls = [commit["html_url"] for commit in commits]
        commit_messages = [commit["commit"]["message"] for commit in commits]

        content = "\n".join(commit_messages)

        summary = generate_summary(commit_messages)

        children = []
        for url, message in zip(commit_urls, commit_messages):
            child_block = {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {"content": message, "link": {"url": url}},
                        },
                    ],
                },
            }
            children.append(child_block)

        payload = {
            "parent": {"database_id": NOTION_DATABASE_ID},
            "properties": {
                "title": {"title": [{"text": {"content": summary}}]},
                "Date": {"date": {"start": date.strftime("%Y-%m-%d")}},
            },
            "children": children,
        }

    print(f"Creating database entry for {date.strftime('%Y-%m-%d')}")
    print("Payload:")
    print(payload)

    response = requests.post(
        f"{NOTION_API_URL}/pages",
        json=payload,
        headers=notion_headers,
    )

    if response.status_code == 200:
        print(f"Created database entry for {date.strftime('%Y-%m-%d')}")
    else:
        print(f"Failed to create database entry for {date.strftime('%Y-%m-%d')}")
        print(response.text)


database_entries = get_database_entries()

start_date = datetime(2023, 6, 1)
end_date = datetime(2023, 11, 30)

delta = timedelta(days=1)

while start_date <= end_date:
    date_str = start_date.strftime("%Y-%m-%d")
    if date_str not in database_entries:
        since = start_date.isoformat()
        until = (start_date + delta).isoformat()
        response = requests.get(
            f"{GITHUB_API_URL}/repos/{GITHUB_REPOSITORY}/commits?since={since}&until={until}",
            headers=github_headers,
        )

        commits = sorted(response.json(), key=lambda c: c["commit"]["author"]["date"])
        if commits:
            commit_texts = [commit["commit"]["message"] for commit in commits]
            summary = generate_summary(commit_texts)
            print(f"Generated summary for {date_str}")
            print(summary)
            create_database_entry(start_date, commits)
        else:
            print(f"No commits available for {date_str}.")
            create_database_entry(start_date, commits)

    start_date += delta
    print(f"Processed {date_str}")

print("Script execution completed.")
