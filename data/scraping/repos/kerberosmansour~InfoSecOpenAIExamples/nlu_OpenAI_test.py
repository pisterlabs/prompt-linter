import os
import openai
import spacy
import datetime
import pytz
from pytz import country_timezones, timezone
from dotenv import load_dotenv
import os

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
nlp = spacy.load("en_core_web_sm")

def generate_gpt3_response(prompt):
    response = openai.Completion.create(
        model="davinci",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0,
    )

    return response.choices[0].text.strip()

def nlu_time_query(user_input):
    gpt3_prompt = f"Help an NLU try to understand the user input delimited with three backticks. The objective is to respond with a text that the NLU can process. The NLU can respond to queries about time and it understand timezones based on countries. If the user is asking about the time in a region that is not a country like a town or city include the country as well. If the user is not asking about time, respond with 'not applicable', but if they are, respond in a way that us helpful for the NLU to proccess: ```{user_input}```"


    gpt3_response = generate_gpt3_response(gpt3_prompt)
    print(gpt3_response)

    doc = nlp(gpt3_response)
    location = None

    for ent in doc.ents:
        if ent.label_ == "GPE":
            location = ent.text

    if "time" in gpt3_response.lower():
        if location:
            return get_time_in_location(location)
        else:
            return get_local_time()

    return f"Sorry, I couldn't understand your query based on the GPT-3 response: '{gpt3_response}'. Please try again."

def get_local_time():
    current_time = datetime.datetime.now()
    return current_time.strftime("%Y-%m-%d %H:%M:%S")

def get_time_in_location(location):
    try:
        country_code = None
        for code, country_name in list(pytz.country_names.items()):
            if location.lower() in country_name.lower():
                country_code = code
                break

        if country_code:
            tz = country_timezones[country_code][0]
            current_time = datetime.datetime.now(timezone(tz))
            return current_time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            return f"Sorry, I couldn't find the timezone for {location}."
    except Exception as e:
        return f"Sorry, an error occurred while processing your request: {str(e)}"

# Example usage:
user_input = "What's the current time in Paris?"
print(nlu_time_query(user_input))
