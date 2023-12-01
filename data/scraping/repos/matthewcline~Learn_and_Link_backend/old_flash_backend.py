# Import Flask and request libraries
from flask import Flask, request
# Import Anthropic Claude API
from anthropic import Client

import cohere

from customized_linkedinAPI import LinkedIn
import os

from LinkedInAPI import MyLinkedInAPI

app = Flask(__name__)

# Create a Claude instance with your API key
claude_api_key = os.environ['CLAUDE_API_KEY']


USERNAME = "shah.jaidev00@gmail.com"
PASSWORD = os.environ.get('LINKEDIN_PASSWORD')

linkedinAPIinstance = MyLinkedInAPI(USERNAME, PASSWORD)

# Define a route for the app
@app.route("/", methods=["POST"])
def index():
    # Get the input paragraph from the request body
    paragraph = request.get_json().get("paragraph")
    # Call the Claude API to extract keywords and types
    keywords = client.extract_keywords(paragraph)
    # Return the keywords and types as a list of tuples
    return str(keywords)


# Define a route for getting recommended profiles
@app.route("/recommended_profiles", methods=["GET"])
def recommended_profiles():
    return recommended_profiles


# Define a route for generating a cover letter
@app.route("/cover-letter", methods=["POST"])
def cover_letter():
    # Get the input text from the request body
    user_intentions_text = request.get_json().get("text")

    #APPEND the summarized profile dict to the user_intentions_text

    #Get the job posting from the request body
    job_posting = request.get_json().get("job_posting")

    # Create a prompt for Claude to generate a cover letter
    prompt = f"Write a cover letter based on the following information about the candidate and the job posting:\n\n" \
                f"Candidate: {text}\n\n" \
                f"Job Posting: {text}" \
                "Cover Letter: "

    print(prompt)

    # Call the Claude API to generate a cover letter
    cover_letter = client.generate_text(prompt)

    # Return the cover letter as a PDF file
    return cover_letter




def summarize_linkedin_profile(profile_dict):
    #Use Claude to summarize the profile

    claude = Client(claude_api_key)   #"claude-instant-v1")
    keys_to_keep = ['industryName', 'lastName', 'firstName', 'geoLocationName', 'headline', 'experience', 'education', 'projects']
    filtered_profile_dict = {key: profile_dict[key] for key in profile_dict if key in keys_to_keep}

    # the values of the dictionary
    profile_summary = str(filtered_profile_dict) 

    #Create a prompt for Claude to summarize the profile
    prompt = f"Summarize the following LinkedIn profile to capture the key parts:\n\n" \
                f"{profile_summary}\n\n" \
                "Summary: "   

    #Call the Claude API to summarize the profile
    summary = claude.generate_text(prompt)

    #Return the summary
    return summary



def cohere_rerank():
    # initialize the Cohere Client with an API Key
    co = cohere.Client('YOUR_API_KEY')

    # define the query and the documents
    query = 'What is the capital of the United States?'
    docs = ['Carson City is the capital city of the American state of Nevada.',
            'The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.',
            'Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.',
            'Capital punishment (the death penalty) has existed in the United States since before the United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.',
            'The capital city of Canada is Ottawa, Ontario.',
            'The capital city of France is Paris.']

    # rerank the documents based on semantic relevance
    results = co.rerank(query=query, documents=docs, top_n=3, model='rerank-english-v2.0')

    # print the reranked documents and their scores
    for result in results:
        print(f"Document: {result.document}")
        print(f"Relevance Score: {result.score}")




def claude_extract_keywords():
    claude = Client(claude_api_key)   #"claude-instant-v1")
    prompt_for_keywords= f"Given this LinkedIn profile: {}, Extract the most relevant keywords from this persons profile that would help him find good connections to network with: "
    summary = claude.generate_text(prompt_for_keywords)