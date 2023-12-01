import requests
from bs4 import BeautifulSoup
import re
import openai
import os
from dotenv import load_dotenv
from cvsslib import cvss31, calculate_vector

# Load .env file
load_dotenv()

# Initialize OpenAI with the API key from .env file
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_summary(cleaned_text, description):
    # Combine the description and cleaned_text
    combined_content = description + "\n\n" + cleaned_text
    
    # Create a chat completion using OpenAI's GPT-4 model
    chat_completion = openai.ChatCompletion.create(
        model="gpt-4", 
        messages=[
            {"role": "user", "content": combined_content + "\n Clearly summarize the vulnerability based on the given details, then explain the Attack Vector, the Attack Complexity, the Privileges Required, the User Interaction, the Scope, the Confidentiality Impact, the Integrity Impact, and the Availability Impact as well as the type of CWE it is. The aim os for an organisation to clearly understand how easy it is to exloit the vulnerability (and is this software ususally exposed to the publick internat or an internal network?), and what the impact of the vulnerability is, in order to priotise it's remediation."}
        ]
    )
    
    # Return the chat completion
    return chat_completion.choices[0].message.content

def get_score(summary):
    # Create a chat completion using OpenAI's GPT-4 model
    chat_completion = openai.ChatCompletion.create(
        model="gpt-4", 
        messages=[
            {"role": "user", "content": summary + "\n Based on the above summary, please return a JSON object with CVSS 3.1 score attributes, which are the Attack Vector, the Attack Complexity, the Privileges Required, the User Interaction, the Scope, the Confidentiality Impact, the Integrity Impact, and the Availability Impact in the format for example: '{\"AV\":\"N\", \"AC\":\"L\", \"PR\":\"N\", \"UI\":\"N\", \"S\":\"C\", \"C\":\"L\", \"I\":\"L\", \"A\":\"L\"}'."}
        ]
    )
    
    # Extract the CVSS 3.1 vector from the response
    response_content = chat_completion.choices[0].message.content
    try:
        cvss_attributes = eval(response_content)  # Convert string representation of dict to actual dict
        cvss_vector = "CVSS:3.1/" + "/".join([f"{k}:{v}" for k, v in cvss_attributes.items()])
    except:
        cvss_vector = ""
    
    return cvss_vector

def fetch_cve_description(cve_id):
    # Step 1: Fetch CVE details from NVD API
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={cve_id}"
    response = requests.get(url)
    data = response.json()

    # Check if vulnerabilities key exists
    if 'vulnerabilities' not in data:
        print(f"No vulnerabilities found for {cve_id}")
        return

    description = None
    # Step 2: Extract and print the English description
    descriptions = data['vulnerabilities'][0]['cve'].get('descriptions', [])
    for desc in descriptions:
        if desc['lang'] == 'en':
            description = desc['value']
            print(description)

    # Step 3: Extract URLs and fetch content using BeautifulSoup
    references = data['vulnerabilities'][0]['cve'].get('references', [])
    reference_urls = [ref['url'] for ref in references]

    valid_url_count = 0  # Counter for URLs that return a 200 response
    for ref_url in reference_urls:
        response = requests.get(ref_url)
        if response.status_code == 200:  # Check if URL returns a 200 OK status
            valid_url_count += 1
            if valid_url_count > 2:  # If we already have two valid URLs, break
                break
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text()
            # Remove occurrences of more than 2 consecutive newlines
            cleaned_text = re.sub(r'\n{1,}', '\n\n', text)
            summary = get_summary(cleaned_text, description)
    print("\nSummary:")
    print(summary)

    # After the summary is printed, call the get_score function
    cvss_vector = get_score(summary)

    # Calculate the CVSS 3.1 score using the provided attributes
    cvss_score_values = calculate_vector(cvss_vector, cvss31)
    print("\nCVSS 3.1 Base Score:", cvss_score_values[0])
    print("CVSS 3.1 Temporal Score:", cvss_score_values[1])
    print("CVSS 3.1 Environmental Score:", cvss_score_values[2])

if __name__ == "__main__":
    cve_id = "CVE-2021-44228"
    fetch_cve_description(cve_id)
