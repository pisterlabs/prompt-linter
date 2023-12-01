import json
import openai
import random

# Set your OpenAI API key
openai.api_key = "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"

# Load candidate info from JSON
with open("E:/hackathon/info_demo.json", 'r') as json_file:
    data = json.load(json_file)
    candidate = data  # Assuming the JSON structure matches the provided data

# Define a sample job description
work_experience=data["work_experience"]
candidate_tech=data["Skills"][0]
job_description = "Frontend Web developer"
jd_tech_keywords = ["react","frontend","javascript"]

# Extract tech stack from candidate's project experience
tech_stack = "good react developer"

# Combine JD and tech stack keywords for generating questions
keywords = jd_tech_keywords + tech_stack.split()

# Generate questions using OpenAI GPT-3
prompt1 = f"Generate 5 questions based on {work_experience}"


response1 = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt1,
    max_tokens=2000,
    stop=None,
    temperature=0.7
)

generated_question1 = response1.choices[0].text.strip()

print(f"Generated questions based on work experience: {generated_question1}")

prompt2=f"generate 5 technical questions based on common techstack between {job_description} and {candidate_tech}"

response2 = openai.Completion.create(
    engine="text-davinci-003",
    prompt=prompt2,
    max_tokens=2000,
    stop=None,
    temperature=0.7
)

generated_question2 = response2.choices[0].text.strip()

print(f"Generated questions based on tech stack: {generated_question2}")





try:
    with open("E:\\hackathon\\tech_ques.json", 'w') as output_file:
        json.dump(generated_question1+generated_question2, output_file, indent=4)
    print("JSON data successfully written ")
except Exception as e:
    print(f"Error while writing JSON data: {e}")

