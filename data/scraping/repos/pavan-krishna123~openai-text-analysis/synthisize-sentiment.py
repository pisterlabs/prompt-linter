import openai
import os
import glob
from time import time, sleep
from uuid import uuid4

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

openai.api_key = open_file('openaiapikey.txt')

def gpt3_completion(prompt, engine='text-davinci-002', temp=1.0, top_p=1.0, tokens=1000, freq_pen=0.0, pres_pen=0.0, stop=None):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response.choices[0].text.strip()
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)

def process_news_articles(folder_path):
    for file_path in glob.glob(os.path.join(folder_path, "*")):
        try:
            # Read the article text from the file
            article_text = open_file(file_path)

            # Construct the prompt
            prompt = f"Please classify the sentiment of the following news article as positive or negative:\n\n{article_text}"

            # Save the prompt to the "prompts" folder
            prompt_filename = os.path.join("prompts", f"{os.path.basename(file_path)}_{time()}.txt")
            save_file(prompt_filename, prompt)

            # Get the sentiment of the article
            sentiment = gpt3_completion(prompt)

            # Save the completion (sentiment) to the "completions" folder
            completion_filename = os.path.join("completions", f"{os.path.basename(file_path)}_{time()}.txt")
            save_file(completion_filename, f"Sentiment: {sentiment}\n")

            print(f"Processed file: {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}\n")

if not os.path.exists("prompts"):
    os.makedirs("prompts")

if not os.path.exists("completions"):
    os.makedirs("completions")

if __name__ == '__main__':
    folder_path = r"C:\Users\Pavankrishna\OneDrive\Desktop\new\test"  # Replace this with the path to your folder containing news articles
    process_news_articles(folder_path)
