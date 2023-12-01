import json
import openai
import requests
import re
import os

from utils import *

# Configuration and Initialization
INDUSTRY_KEYWORD = os.getenv('INDUSTRY_KEYWORD')
WHOISJSONAPI = os.getenv('WHOISJSONAPI')
COMPAREPRICES = os.getenv('COMPAREPRICES')
SERP_PRICES_EXT = os.getenv('SERP_PRICES_EXT') or exit("SERP_PRICES_EXT is not defined. Please define it in .env file if you want to use this script.")
DATA_FOLDER = f"data/{INDUSTRY_KEYWORD}"
BASE_GPTV= os.environ.get('BASE_GPTV','gpt-3.5-turbo-1106')

def find_link_to_plans(serp_content, domain_data):
    """ Use GPT to find the link to the plans page from SERP content. """
    print(serp_content)
    try:
        response = openai.ChatCompletion.create(
            model=BASE_GPTV,  # Update this to the model you're using
            response_format={"type": "json_object"},
            temperature=0,
            messages=[
                {"role": "system", "content": "Find the link to the page with "+SERP_PRICES_EXT+" for "+INDUSTRY_KEYWORD+". Return JSON with 'url' field, or return 'Not found' if not found and no chances that another page like faq can contain required information."},
                {"role": "user", "content": serp_content}
            ]
        )
        if response['choices'][0]['message']['content']:
            ch = response['choices'][0]['message']['content']
            urls = json.loads(ch)
        else:
            urls = "Not found"
        return urls
    except Exception as e:
        print(f"An error occurred: {e}")
        return 'Not found'


def process_domain_data(domain, domain_data):
    """ Process the data for a single domain. """
    query = f"site:{domain} {SERP_PRICES_EXT}"
    
    organic_results = search_companies_on_google(query, 10)
    serp_content = "\n\n".join([
        f"{result.get('position', 'N/A')}. link: {result.get('link', '')}, "
        f"text: {result.get('title', '')},  "
        f"snippet: {result.get('snippet', '')}"
        for result in organic_results
    ])

    # Determine the plans URL based on the number of search results
    if organic_results:
        urls = find_link_to_plans(serp_content, domain_data)
        return urls
    else:
        return 'Not found'

   


def main():
    # Load data
    data = load_from_json_file("1companies.json", DATA_FOLDER)
    # Filter only data with nature=single project and not yet crawled
    data = {k: v for k, v in data.items() if v['nature'] == 'single project' and 'priceAndPlansCrawled' not in v  } 

    print(len(data), "domains to process.")

    for domain, domain_data in data.items():
        print(f"\n\nProcessing prices {domain}...")
        plans_url = process_domain_data(domain, domain_data)

        # Handle found plans URL
        if plans_url != 'Not found':
            plans_url = correct_url(plans_url['url'])
            domain_data["priceAndPlansCrawled"] = plans_url
            
            print(f"Crawling {plans_url}...")
            summary = extract_content(plans_url)
            if (len(summary['text_content']) < 600):
                print ("try to extract from web archive")
                url1 = get_wayback_url(domain)
                if (url1 is not None): 
                    summary = extract_content(url1)

            details = load_from_json_file(f"{domain}.json", DATA_FOLDER)
            details["priceAndPlans"] = summary['text_content']
            save_to_json_file(details, f"{domain}.json", DATA_FOLDER)
        else:
            domain_data["priceAndPlansCrawled"] = plans_url

        save_to_json_file(data, "1companies.json", DATA_FOLDER)

    print("Processing complete. Next step: 5")

if __name__ == "__main__":
    main()
