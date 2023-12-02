import time
import csv
import openai
import json
import os

openai.api_key = os.getenv('OPENAI_API_KEY')


def generate_data(input_file, output_file):
    with open(input_file, 'r') as csvinput, open(output_file, 'w') as jsonl_output:
        reader = csv.reader(csvinput)
        for row in reader:
            city_country = row[0]
            for delay_secs in (2**x for x in range(6)):  # up to 2^5=32 seconds
                try:
                    message = {
                        "role": "system",
                        "content": f"You are BourdainGPT. The user will provide a message about {city_country}, and you will respond with a three paragraph response about that city. In particular, the first paragraph will give a short summary about the city, its culture, environment and history. The next two paragraphs will be about the culinary traditions of the city, with specific examples of dishes, ingredients and festivities. Try to keep each response brief (1 sentences for first paragraph, ~3 sentences per paragraph) but informative and colorful."
                    }
                    user_message = {
                        "role": "user",
                        "content": f"Give me brief but informative description of {city_country} and its culinary tradition. Keep each paragraph to 1-3 sentences."
                    }

                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[message, user_message]
                    )
                    completion = response['choices'][0]['message']['content']
                    jsonl_output.write(json.dumps(
                        {"prompt": city_country, "completion": completion}) + '\n')
                    break

                except openai.error.OpenAIError as e:
                    print(f"Error: {e}. Retrying in {delay_secs} seconds.")
                    time.sleep(delay_secs)
                    continue  # continue to the next loop iteration


generate_data('cities.csv', 'cities.jsonl')
