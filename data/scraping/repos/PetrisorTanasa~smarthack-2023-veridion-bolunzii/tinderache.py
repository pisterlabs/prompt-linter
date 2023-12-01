import numpy as np
import Levenshtein as lev
import pandas as pd
import openai
import csv
import requests
import sys

file_path = './updated_file_with_embeddings.csv'
data = pd.read_csv(file_path)

def compare_embeding(target_company_embedding):
    global companies_data
    cosine_similarities = []

    # Calculate cosine similarity between the target company and all other companies
    for index, row in data.iloc[:].iterrows():
        company_embedding = row['embeddings'].strip('[]').split()
        # Convert the embeddings to numpy arrays of type float
        target_company_embedding = np.array(target_company_embedding[:], dtype=float)
        company_embedding = np.array(company_embedding, dtype=float)

        # Calculate cosine similarity
        cosine_similarity = np.dot(target_company_embedding, company_embedding) / (np.linalg.norm(target_company_embedding) * np.linalg.norm(company_embedding))
        cosine_similarities.append(cosine_similarity)

    similarity_scores = list(zip(data['company_name'].tolist(), cosine_similarities))
    # similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # top_6_similar_companies = similarity_scores[:6]

    # # Output the top 6 most similar companies starting with the second most similar company
    # for idx, (company, score) in enumerate(top_6_similar_companies[1:], start=2):
    #     print(f"Rank {idx}: Company = {company}, Similarity Score = {score}")

    return similarity_scores

def Levenshtein_compare(target_company):
    global data
    # Format the company data
    global companies_data

    # Find the target company index
    target_company_index = -1
    for idx, company in enumerate(companies_data):
        # Capitalize the company name for comparison
        # print(company["company_name"].upper().strip("\""), target_company.upper())
        if company["company_name"].upper().strip("\" ") == target_company.upper():
            target_company_index = idx
            break
    
    if target_company_index == -1:
        print("Target company not found.")
        return
    
    # Get the target company data
    company_data_1 = companies_data[target_company_index]

    # Remove the target company from the list of companies
    other_companies = companies_data[:target_company_index] + companies_data[target_company_index + 1:]



    # Store similarity scores
    similarity_scores = []

    for company in other_companies:
        similarity_score = compare_companies(company_data_1, company)
        similarity_scores.append((company, similarity_score))

    embedding_similarity = compare_embeding(company_data_1['embeddings'])

    products = []
    # for every company in the dataset calculate, using their names as key, the product between the similarity score and the cosine similarity
    scores_len = len(similarity_scores)
    similarity_len = len(embedding_similarity)
    for idx, (company, score) in enumerate(similarity_scores, start=1):
        # print(f"Rank {idx}: Company = {company['company_name']}", embedding_similarity[idx][0])
        products.append((company, score  * embedding_similarity[idx][1]))


    # sort the products based on the similarity score
    products.sort(key=lambda x: x[1], reverse=True)

    # get the top 5 products
    top_5_best_matches = products[:100]
    top_5_best_matches_names = list()
    # Output the top 5 best matches
    #print('-------------------------------------------------------------')
    for idx, (company, score) in enumerate(top_5_best_matches, start=1):
        if company['company_name'] not in top_5_best_matches_names:
            #print(f"Rank {idx}: Company = {company['company_name']}, Similarity Score = {score}")
            top_5_best_matches_names.append(company['company_name'])
        
        if len(top_5_best_matches_names) == 5:
            break
    return top_5_best_matches_names


# Function to preprocess and format company data
def format_company_data(row):
    return {
        "company_name": row['company_name'],
        "business_tags": eval(row['business_tags']),  # Convert string representation of list to actual list
        "main_business_category": row['main_business_category'].strip('"'),
        "main_industry": row['main_industry'].strip('"'),
        #"main_sector": row['main_sector'].strip('"'),
        "embeddings": row['embeddings'].strip('[]').split(),
        "sics_industry": row['sics_industry'].strip('"'),
        "sics_subsector": row['sics_subsector'].strip('"'),
        "sics_sector": row['sics_sector'].strip('"'),
        "long_description": row['long_description'],
        #"technologies": eval(row['technologies'])  # Convert string representation of list to actual list

    }

# Function to convert Levenshtein distance to similarity percentage
def levenshtein_similarity(s1, s2):
    dist = lev.distance(s1, s2)
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 100
    return (1 - dist / max_len) * 100

# Function to calculate set similarity
def calculate_set_similarity(set1, set2):
    common_elements = set1.intersection(set2)
    total_elements = set1.union(set2)
    if not total_elements:
        return 100  # Both sets are empty, so they are identical
    return (len(common_elements) / len(total_elements)) * 100

# Function to compare two companies
def compare_companies(company1, company2):
    # Calculate similarities
    business_tags_similarity = calculate_set_similarity(set(company1["business_tags"]), set(company2["business_tags"]))
    sics_industry_similarity = levenshtein_similarity(company1["sics_industry"], company2["sics_industry"])
    sics_subsector_similarity = levenshtein_similarity(company1["sics_subsector"], company2["sics_subsector"])
    sics_sector_similarity = levenshtein_similarity(company1["sics_sector"], company2["sics_sector"])
    category_similarity = levenshtein_similarity(company1["main_business_category"], company2["main_business_category"])
    industry_similarity = levenshtein_similarity(company1["main_industry"], company2["main_industry"])
    long_description_similarity = levenshtein_similarity(company1["long_description"], company2["long_description"])

    # Calculate weighted average of similarities
    similarity_score = (business_tags_similarity * 0.15 + sics_industry_similarity * 0.05 + sics_subsector_similarity * 0.05 + sics_sector_similarity * 0.05 + category_similarity * 0.3 + industry_similarity * 0.2  + long_description_similarity * 0.2)

    return similarity_score

# API details
url = 'https://data.veridion.com/match/v4/companies'
headers = {
    "x-api-key": "",  # Replace with your API key
    "Content-type": "application/json"
}

def search_and_enrich(company_name):
    payload = {
        "legal_names": [company_name],
        "phone_number": "+4073"
    }
    response = requests.post(url, json=payload, headers=headers)

    # Check if the response status code indicates success
    if response.ok:
        try:
            data = response.json()
            # Extract required fields
            # Initialize a counter for default values
            default_count = 0

            # Apply default values where necessary and increment the counter
            year_founded = data.get("year_founded")
            if year_founded is None:
                year_founded = 2015
                default_count += 1

            employee_count = data.get("employee_count")
            if employee_count is None:
                employee_count = 100
                default_count += 1

            estimated_revenue = data.get("estimated_revenue")
            if estimated_revenue is None:
                estimated_revenue = 1_000_000
                default_count += 1

            # Set the flag based on the counter
            flag = 1 if default_count > 2 else 0

            # Create the extracted_data dictionary
            extracted_data = {
                "company_name": company_name,
                "main_country": data.get("main_country"),
                "main_region": data.get("main_region"),
                "num_locations": data.get("num_locations", 1),
                "company_type": data.get("company_type"),
                "year_founded": year_founded,
                "employee_count": employee_count,
                "estimated_revenue": estimated_revenue,
                "short_description": data.get("short_description"),
                "long_description": data.get("long_description"),
                "business_tags": data.get("business_tags"),
                "main_business_category": data.get("main_business_category"),
                "main_industry": data.get("main_industry"),
                "main_sector": data.get("main_sector"),
                "website_url": data.get("website_url"),
                "flag": flag
            }
            return extracted_data
        except ValueError:
            # Handle the case where response is not in JSON format
            print(f"Non-JSON response for {company_name}: {response.text}")
            return None
    else:
        # Print response text for any HTTP errors
        print(f"Error response for {company_name}: {response.status_code}, {response.text}")
        return None

companies_data = [format_company_data(row) for index, row in data.iloc[:].iterrows()]

input_company_name = sys.argv[1]

# List of top 5 similar companies
company_names = Levenshtein_compare(search_and_enrich(input_company_name)['company_name'])
    
all_extracted_data = list()

# Making API calls for each company and printing the responses
for company in company_names:
    response_data = search_and_enrich(company)
    if response_data:
        all_extracted_data.append(response_data)
        
try:
    with open('extracted_data.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_extracted_data[0].keys())
        writer.writeheader()
        for data in all_extracted_data:
            writer.writerow(data)
    #print("Data successfully saved to 'extracted_data.csv'")
except IOError as e:
    print(f"Error writing to file: {e}")
    
def read_csv_data(file_path):
    companies = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            companies.append(row)
    return companies

companies_data = read_csv_data('extracted_data.csv')

def query_gpt_api(prompt):
    openai.api_key = ''  # Replace with your OpenAI API key

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Using the chat model
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content'].strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"

custom_prompt = "Based on the following key data, provide a risk score out of 100 for the company. Please provide only the numeric score. Please only the numeric score please pwetty please:"

for company in companies_data:
    # Include only key data points relevant to risk assessment
    key_data = ", ".join([f"{key}: {value}" for key, value in company.items() if key in ["main_country", "num_locations", "company_type", "year_founded", "employee_count", "estimated_revenue"]])
    company_prompt = f"{custom_prompt}\n\n{key_data}"
    gpt_response = query_gpt_api(company_prompt)
    
    if len(gpt_response) > 3:
        gpt_response = -1
    print(f"{company['company_name']},{gpt_response},{company['website_url']},{company['flag']}")
