import os
import requests
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import openai
import re
from typing import Dict, List, Any, Optional

load_dotenv()

NOTION_API_TOKEN = os.getenv("NOTION_API_TOKEN")
NOTION_DATABASE_ID = os.getenv("DATABASE_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NOTION_API_BASE_URL = "https://api.notion.com/v1"

NOTION_API_HEADERS = {
    "Authorization": f"Bearer {NOTION_API_TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}

def send_request(method: str, endpoint: str, json: Optional[Dict[str, Any]]=None) -> Dict[str, Any]:
    url = f"{NOTION_API_BASE_URL}/{endpoint}"
    return requests.request(method, url, headers=NOTION_API_HEADERS, json=json).json()

def completion(text: str) -> str:
    openai.api_key = OPENAI_API_KEY

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=[
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0]["message"]["content"].strip()

def parse_to_notion_blocks(markdown_text: str) -> List[Dict[str, Any]]:
    lines = markdown_text.split("\n")
    notion_blocks = []

    for line in lines:
        # Check if the line is a bulleted list item
        if re.match(r"^(\*|\-|\d\.)\s", line):
            # Remove the bullet point and extra space
            content = re.sub(r"^(\*|\-|\d\.)\s", "", line)
            
            # Create a Notion block for this line
            notion_block = {
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": content,
                            }
                        }
                    ],
                    "color": "default",
                }
            }
            notion_blocks.append(notion_block)
    
    return notion_blocks


def create_prompt(transcript: str) -> str:
    return f"""以下は、ある会議の書き起こしです。

{transcript}

この会議のサマリーを作成してください。サマリーは、以下のような形式で書いてください。

- 会議の目的
- 会議の内容
- 会議の結果

サマリー:
"""

def create_prompt_act(transcript: str) -> str:
    return f"""以下は、ある会議の書き起こしです。

{transcript}

---

この会議の次に行うべきアクションを作成してください。アクションの記述は以下のルールに従ってください。

・リスト形式で出力する (先頭は - を使う)
・簡潔に表現する

アクション:
"""

def get_notion_user_id() -> Dict[str, Any]: 
    response_data = send_request("GET", "users")

    return response_data

def convert_milliseconds(ms: int) -> str:
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f'{int(hours)}:{int(minutes):02d}:{int(seconds):02d}'
    elif minutes > 0:
        return f'{int(minutes)}:{int(seconds):02d}'
    else:
        return f'{int(seconds)}'

def patch_transcript(page_id: str,file_path: str) -> Dict[str, Any]:
    transcript = pd.read_csv(file_path)

    prompt = create_prompt(transcript.to_string())
    summary = completion(prompt)
    act_prompt = create_prompt_act(transcript.to_string())
    action = completion(act_prompt)

    print("## Summary\n\n" + summary + "\n\n## Action\n\n" + action)

    summary = parse_to_notion_blocks(summary)
    print(summary)
    action = parse_to_notion_blocks(action)

    children_paragraph = []

    for _, row in transcript.iterrows():
        start = convert_milliseconds(row["start"])
        end = convert_milliseconds(row["end"])
        text = row["text"]
        rich_text = f"{start} - {end}: {text}"

        paragraph_block = {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [
                    {
                        "type": "text",
                        "text": {
                            "content": rich_text,
                        }
                    }
                ]
            }
        }

        children_paragraph.append(paragraph_block)

    payload = {
        "children": [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{ "type": "text", "text": { "content": "会議の結果" } }]
                }
            },
        ] + summary + [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{ "type": "text", "text": { "content": "次にするアクション" } }]
                }
            },
        ] + action + [
            {
                "object": "block",
                "type": "heading_2",
                "heading_2": {
                    "rich_text": [{ "type": "text", "text": { "content": "会議のトランスクリプト" } }]
                }
            },
        ] + children_paragraph
    }
    response_data = send_request("PATCH", f"blocks/{page_id}/children", json=payload)

    return response_data

def get_emoji_for_month(month: int) -> str:
    month_emoji = {
        1: "❄️",
        2: "⛄️",
        3: "🌸",
        4: "🌷",
        5: "🦋",
        6: "☀️",
        7: "🏖️",
        8: "🍉",
        9: "🍁",
        10: "🎃",
        11: "🍂",
        12: "🎄",
    }
    return month_emoji.get(month, "❓")

def get_or_create_today_page_id() -> str:
    today = datetime.now().date()
    today_str = today.isoformat()  

    # Payload with filter to match today's date
    payload_for_get = {
        "filter": {
            "property": "日付",
            "date": {"equals": today_str}
        }
    }

    response_data = send_request("POST", f"databases/{NOTION_DATABASE_ID}/query", json=payload_for_get)

    if response_data['results']:
        return response_data['results'][0]["id"]
    else:
        payload_for_post = {
            "parent": {"database_id": NOTION_DATABASE_ID},
            "icon": {
                "emoji": get_emoji_for_month(today.month)
            },
            "properties": {
                "日付": {
                    "type": "date",
                    "date": {"start": today_str}
                },
                "title": {
                    "type": "title",
                    "title": [
                        {
                            "type": "mention",
                            "mention": {
                                "type": "date",
                                "date": {
                                "start": today_str,
                                "end": None,
                                }
                            },
                        }
                    ]
                }
            }
        }

        response_data = send_request("POST", "pages", json=payload_for_post)

        return response_data["id"]

if __name__ == "__main__":
    page_id = get_or_create_today_page_id()
    response = patch_transcript(page_id,"./content/transcript.csv")
    if (response):
        print("Success！")