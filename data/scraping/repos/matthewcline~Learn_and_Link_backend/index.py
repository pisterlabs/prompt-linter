# Import Flask and request libraries
from flask import Flask, request
# Import Anthropic Claude API
import anthropic

import cohere

# from customized_linkedinAPI import LinkedIn
import os

from WrapperLinkedInAPI import MyLinkedInAPI

app = Flask(__name__)

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')

claude = anthropic.Client(os.environ.get('ANTHROPIC_API_KEY'))
co = cohere.Client(os.environ.get('COHERE_API_KEY'))


USERNAME = "test.hackgpt.lnl@gmail.com"
PASSWORD = os.environ.get('LINKEDIN_PASSWORD')

myAPIWrapper = MyLinkedInAPI(USERNAME, PASSWORD)

# Define a route for the app
@app.route("/", methods=["POST"])
def index():
    return None 




# Define a route for getting recommended profiles
@app.route("/recommended_profiles", methods=["POST"])
def recommended_profiles():
    user_free_form_text = request.get_json().get("user_free_form_text")
    user_linkedin_profile_url = request.get_json().get("user_linkedin_profile_url")
    profile_name = user_linkedin_profile_url.split("/")[-1]
    myAPIWrapper.recommended_profiles(user_free_form_text, profile_name)
    return recommended_profiles

# Define a route for generating a cover letter
@app.route("/cover-letter", methods=["POST"])
def cover_letter():
    # Get the input text from the request body
    user_free_form_text = request.get_json().get("user_free_form_text")
    user_linkedin_profile_url = request.get_json().get("user_linkedin_profile_url")

    user_profile_name = user_linkedin_profile_url.split("/")[-1]
    user_profile_summary = myAPIWrapper.get_cleaned_profile_summary(user_profile_name)

    user_profile_summary_and_intentions = user_profile_summary + "\n\n" + user_free_form_text

    #Get the job posting from the request body
    job_id = request.get_json().get("job_posting_id")

    job_posting_summary = myAPIWrapper.get_job_posting(job_id)

    # Create a prompt for Claude to generate a cover letter
    prompt = f"Write a cover letter based on the following information about the candidate and the job posting:\n\n" \
                f"Candidate Profile Summary: {user_profile_summary_and_intentions}\n\n" \
                f"Job Posting Summary: {job_posting_summary}" \
                "Cover Letter: "
    
    print(prompt)
    
    # Call the Claude API to generate an intro
    max_tokens_to_sample: int = 3000
    resp = claude.completion(
        prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-instant-v1",
        max_tokens_to_sample=max_tokens_to_sample,
    )

    print(resp['completion'])
    return resp['completion']

@app.route("/intro", methods=["POST"])
def intro():

    user_free_form_text = request.get_json().get("user_free_form_text")

    candidate_profile_url = request.get_json().get("user_linkedin_profile_url")
    candidate_profile_name = candidate_profile_url.split("/")[-1]

    lead_summary = request.get_json().get("lead_linkedin_profile_url")
    lead_profile_name = lead_summary.split("/")[-1]

    print("generating intro...")
    generated_intro = myAPIWrapper.intro_generation(candidate_profile_name, lead_profile_name, user_free_form_text)

    return generated_intro