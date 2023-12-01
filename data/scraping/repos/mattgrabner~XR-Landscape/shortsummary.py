import os
import requests
import openai
import sys
from dotenv import load_dotenv

load_dotenv()

def main(databaseID):
    NOTION_API_KEY = os.getenv('NOTION_API_KEY')
    
    if NOTION_API_KEY is None:
        raise ValueError("No Notion API key provided. Please set the NOTION_API_KEY environment variable.")
    
    headers = {
        "Authorization": "Bearer " + NOTION_API_KEY,
        "Content-Type": "application/json",
        "Notion-Version": "2022-02-22"
    }

    companies = readDatabase(databaseID, headers)

    print(f"Processing {len(companies)} companies...")

    skip_count = 0
    no_description_count = 0

    for page_id, company_name, new_category, description, shortversion, active in companies:
        if not description:
            print(f"Skipping {company_name} as there is no description provided.")
            no_description_count += 1
            continue
        
        if shortversion or active.lower() == 'no':
            print(f"Skipping {company_name} as 'shortversion' is already filled or company is not active.")
            skip_count += 1
            continue

        print(f"Generating short summary for {company_name}...")
        try:
            generated_description = generateCompanyDescription(company_name, new_category, description)
            updateContent(page_id, generated_description)
            print(f"Updated content for {company_name}")
        except Exception as e:
            print(f"Failed to update content for {company_name}: {str(e)}")

    print(f"Total number of skipped companies due to existing 'shortversion' or inactive status: {skip_count}")
    print(f"Total number of skipped companies due to missing descriptions: {no_description_count}")

def readDatabase(databaseID, headers):
    readUrl = f"https://api.notion.com/v1/databases/{databaseID}/query"
    has_more = True
    start_cursor = None
    data_list = []

    while has_more:
        payload = {}
        if start_cursor:
            payload['start_cursor'] = start_cursor
        res = requests.post(readUrl, headers=headers, json=payload)
        data = res.json()
        for page in data.get('results', []):
            properties = page.get('properties', {})
            page_id = page.get('id')
            company_title_list = properties.get('Company / Organization', {}).get('title', [{}])
            company = company_title_list[0].get('plain_text', '') if company_title_list else ''
            new_category = [option.get('name', '') for option in properties.get('Category', {}).get('multi_select', [])]
            description = properties.get('Description', {}).get('rich_text', [{}])[0].get('plain_text', '') if properties.get('Description', {}).get('rich_text') else ''
            shortversion = properties.get('shortversion', {}).get('rich_text', [{}])[0].get('plain_text', '') if properties.get('shortversion', {}).get('rich_text') else ''
            active = properties.get('Active', {}).get('select', {}).get('name', '')
            data_list.append((page_id, company, new_category, description, shortversion, active))

        has_more = data.get('has_more')
        start_cursor = data.get('next_cursor')

    return data_list


def generateCompanyDescription(company, category, description):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
        {
          "role": "system",
          "content": "You are an experienced copywriter specializing in detailed, technical, and non-promotional descriptions. Your task is to write a concise, accurate, and understandable description for companies in the XR landscape, an overview of organizations in the XR ecosystem. Your descriptions should focus on what the company is actually doing in the XR (augmented reality, virtual reality, and extended reality) field, while avoiding marketing jargon and superlatives."
        },
        {
          "role": "user",
          "content": f"Write a 50-character max description for the company {company}, categorized under \"{category}\", based on the following description: \"{description}\". The description should start with \"{company}:\" and then describe their work in XR/AR/VR. It should not sound like a marketing pitch but should clearly convey what the company does. Do not make full sentences, create something more similar to headlines or taglines!"
        }
      ],
      temperature=1,
      max_tokens=64,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    # Remove surrounding quotes if present
    content = response.choices[0].message['content'].strip('"')
    return content


def updateContent(page_id, generated_description):
    NOTION_API_KEY = os.getenv('NOTION_API_KEY')
    
    if NOTION_API_KEY is None:
        raise ValueError("No Notion API key provided. Please set the NOTION_API_KEY environment variable.")
    
    headers = {
        "Authorization": "Bearer " + NOTION_API_KEY,
        "Content-Type": "application/json",
        "Notion-Version": "2022-02-22"
    }

    data = {
        "properties": {
            "shortversion": {
                "type": "rich_text",
                "rich_text": [
                    {
                        "text": {
                            "content": generated_description
                        }
                    }
                ]
            }
        }
    }

    updateUrl = f"https://api.notion.com/v1/pages/{page_id}"
    res = requests.patch(updateUrl, headers=headers, json=data)

    if res.status_code != 200:
        print(f"Failed to update page: {res.content}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python shortsummary.py <Notion Database ID>")
        print("Please provide the Notion Database ID as an argument when calling the script.")
        sys.exit(1)
    else:
        main(sys.argv[1])
