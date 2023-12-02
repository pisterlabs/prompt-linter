import cohere
import os
from dotenv import load_dotenv

load_dotenv()
co = cohere.Client(os.getenv('COHERE_API_KEY','RMppgVMUjgiKZRWSIwjmmfbOwRa9YEhi1B15oxQ2'))

# we will need to connect the post request to get the text from the user here as well
text = 'The U.S. Senate has passed a bill that would make daylight time permanent, but questions remain as to whether this could be beneficial to Canadians who want to keep daylight time year-round.'


prompt = f"""Read the following:\n\n {text}.\n\nWrite a summary of the text in 200 words or less."""


# Summarize the text 

def summarize(text):
    response = co.generate(
        model='command-xlarge-nightly',
        prompt = text,
        max_tokens=200,
        temperature=0.8,
        stop_sequences=["--"])
    summary = response.generations[0].text
    # print(summary)
    return summary

# summarize(prompt)

