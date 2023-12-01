import os
import re
import json
import openai
import time

from utils import *

# Environment Variables
SERPAPI_KEY = os.environ.get('SERPAPI_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
INDUSTRY_KEYWORD = os.environ.get('INDUSTRY_KEYWORD')
KEYWORD_FOR_SERP = os.environ.get('KEYWORD_FOR_SERP', INDUSTRY_KEYWORD)
BASE_GPTV= os.environ.get('BASE_GPTV','gpt-3.5-turbo')
if not SERPAPI_KEY:
    print("Please set the SERPAPI_KEY environment variable.")
    exit()

openai.api_key = OPENAI_API_KEY  # Set the OpenAI API key
def extract_company_urls_from_serp(serp_content, industry_query):
    try:
        prompt = f"Analyse SERP and Identify sites based on a given Google search query. '{industry_query}'. Return only list of urls if found (use 'urls' key). Return only JSON with urls.\n\n{serp_content}\n\n "
        
        response = openai.ChatCompletion.create(
            model=BASE_GPTV,  # Update this to the model you're using
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": prompt}
            ]
        )

        if response['choices'][0]['message']['content']:
            urls = json.loads(response['choices'][0]['message']['content'])
        else:
            #return "Not found" json
            urls = {'urls': []}
        return urls

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Not found"

def read_existing_domains(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        return {}


def main():
    industry_query = INDUSTRY_KEYWORD
    
    organic_results = search_companies_on_google(KEYWORD_FOR_SERP, 40)
    print(f"Found {len(organic_results)} results for {KEYWORD_FOR_SERP}")
    serp_content = ""
    for result in organic_results:
        if "snippet" not in result:
            result["snippet"] = ""
        serp_content += (str(result["position"]) + ". " + result["link"] + " " + result["title"] + " " + result["snippet"])+"\n\n"
            
    company_urls = extract_company_urls_from_serp(serp_content, industry_query)

    # Read existing domains
    directory_name = os.path.join('data', INDUSTRY_KEYWORD)
    file_path = os.path.join(directory_name, '1companies.json')
    existing_domains = read_existing_domains(file_path)

    # Update the list of companies, if the domain does not exist
    for url in company_urls['urls']:
        domain = extract_domain_from_url(url)
        if domain and domain not in existing_domains:
            existing_domains[domain] = {'url': url}
        else:
            print(domain + " exists")

    # Save updated data
    save_to_json_file(existing_domains, '1companies.json', directory_name)
    print(f"Company data saved to {directory_name}/1companies.json")

if __name__ == '__main__':
    main()