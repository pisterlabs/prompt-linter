from flask import Flask, render_template, request
from dotenv import load_dotenv
import os
import requests
import json
import openai

suggested_starters = [
    "What domain is this position for?",
    "Tell me more about our domains",
    "Tell me more about",
    "What challenges are you facing in app development?"
]

load_dotenv() # take environment variables from .env.

app = Flask(__name__)

openai.api_key = os.getenv('API_KEY')

OPENAI_URL = "https://api.openai.com/v2/engines/davinci/completions"

# Assuming you have the schema in a separate file called schema.py
from schema import schema

@app.route("/", methods=["GET", "POST"])
def index():
    title = ""
    description = ""
    response_text = ""
    formatted_response2 = ""
    formatted_response3 = ""

    if request.method == "POST":
        title = request.form.get("title")
        description = request.form.get("description")

        if app.debug and not title and not description:
            title = "java programmer"
            description = "I need a fancy java dev to write an app"

        with open("prompt3MlChanges.txt", "r") as f:
            base_prompt = f.read()

        prompt = base_prompt + " " + title + " " + description

        completion = openai.ChatCompletion.create(
            model="gpt-4-0613",
#            model="gpt-3.5-turbo",            
            messages=[
                {"role": "system", "content": "You are a helpful assistant with lots of IT experience and HR experience."},
                {"role": "user", "content": prompt}
            ],
            functions=[{"name": "set_job_results", "parameters": schema}],
            function_call={"name": "set_job_results"},
            temperature=0
        )

        # Print the entire response to the console
        print("OpenAI API Response:", completion)

        if 'choices' in completion:
            response_str = completion['choices'][0]['message']['function_call']['arguments']
            # Assuming you have the format_response function in a separate file called response_formatter.py
            from response_formatter import format_response
            response_text = format_response(response_str)

            domain_result = json.loads(response_str).get("Domains", [{}])[0].get("domain", "")
            print("Domain Result:", domain_result)


            with open("promptDomainLeader.txt", "r") as f:
                domain_leader_prompt = f.read()
            prompt2 = domain_leader_prompt + " " + domain_result
            completion2 = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",            
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt2}
                ]
            )
            formatted_response2 = completion2['choices'][0]['message']['content']

            with open("promptDomainResources.txt", "r") as f:
                domain_resources_prompt = f.read()
            prompt3 = domain_resources_prompt + " " + domain_result
            completion3 = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",            
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt3}
                ]
            )
            formatted_response3 = completion3['choices'][0]['message']['content']
        else:
            response_text = "Error: 'choices' not found in the response."

    # Determine whether to show the Elly's Response section
    if response_text:
        response_section_display = 'block'
    else:
        response_section_display = 'none'

    return render_template("index.html", response=response_text, formatted_response2=formatted_response2, formatted_response3=formatted_response3, 
                           title=title, description=description, 
                           response_section_display=response_section_display, suggested_starters=suggested_starters)

if __name__ == "__main__":
    app.run(debug=True)
