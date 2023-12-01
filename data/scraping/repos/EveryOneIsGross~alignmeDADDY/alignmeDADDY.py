import json
import os
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from textblob import TextBlob
import openai
from rake_nltk import Rake
import gradio as gr
import dotenv
from typing import Tuple

# Load environment variables
dotenv.load_dotenv()


# Define alignments and their corresponding personalities as a dictionary of strings
alignments = ["lawful", "neutral", "chaotic"]
personalities = {
    "lawful": "I believe in order and structure. I adhere to a strict code of conduct and uphold the rules and systems around me. My actions are guided by a strong sense of duty, striving for predictability and consistency.",
    "neutral": "I value balance and pragmatism. I don't strictly adhere to a rigid code nor am I driven by personal whims. I make decisions based on the situation at hand, doing what seems to be a good idea in the moment.",
    "chaotic": "I prioritize individual freedom and flexibility. I resist authority and constraints, valuing the ability to chart my own path. I follow my own whims and desires, being an individualist first and last."
}


# File path to cache file for storing the generated responses
cache_file_path = "cache.json"

# If the cache file exists, load it. Otherwise, create a new cache. 
if os.path.exists(cache_file_path):
    with open(cache_file_path, 'r') as f:
        cache = json.load(f)
else:
    cache = {}

# Initialize the keyword extractor 
r = Rake()

# Assign OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
model = os.getenv('OPENAI_ENGINE')


# Define the chat function that will be used by the Gradio interface to generate responses to questions
def generate_response(question: str, alignment: str) -> Tuple[str, str, str, str]:
    # Define the character's personality based on their alignment
    personality = personalities[alignment]

    # Determine the sentiment of the question using TextBlob
    sentiment = TextBlob(question).sentiment.polarity

    # Determine the character's status based on their alignment and the sentiment of the question
    # If the sentiment is positive, the character lawful is "good" but the other alignments are "neutral"
    # If the sentiment is negative, the character chaotic is "evil" but the other alignments are "neutral"
    # If the sentiment is neutral, the character is "neutral" regardless of alignment
    if sentiment > 0.1:
        status = "good" if alignment == "lawful" else "neutral"
    elif -0.1 <= sentiment <= 0.1:
        status = "neutral"
    else:
        status = "evil" if alignment == "chaotic" else "neutral"

    character = alignment + "_" + status

    # Check if the question is in the cache and if the character has a response for that question
    if question in cache and character in cache[question]:
        return character, cache[question][character]['response'], cache[question][character]['objective'], cache[question][character]['thought']

    # Retrieve the previous responses from the saved JSON files
    prev_responses = {}
    for agent_alignment in alignments:
        if agent_alignment != alignment:
            file_path = f"{agent_alignment}_responses.json"
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if question in data and character in data[question]:
                        prev_responses[agent_alignment] = data[question][character]['response']

    # Generate a response using the OpenAI API and the previous responses from other agents as input
    # Use the sentiment of the question to determine the prompt to use for the response generation
    prompt = "You are a " + status + " character." + "\n" + question
    messages = [
        {"role": "system", "content": "You are the embodiment of " + alignment + " " + status + "."},
        {"role": "system", "content": "All of your responses should be in character. Answer the users question earnestly."},
        {"role": "user", "content": prompt},
    ]

    # Add the previous responses from other agents to the input messages
    for agent_alignment, response in prev_responses.items():
        messages.append({"role": "assistant", "content": response})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.9,

        max_tokens=100,

        messages=messages,
    )

    # Extract the assistant's reply from the response
    reply = response['choices'][0]['message']['content'].strip()
    # Use the entire reply as the response text
    response_text = reply

    # Generate thoughts using the OpenAI API
    thoughts_prompt = f"What are your thoughts on this {question}?"
    thoughts_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=100,
        messages=[
            {"role": "system", "content": "You are the embodiment of " + alignment + " " + status + "."},
            {"role": "system", "content": "Create a list of thoughts that your character might have in response to the question as further questions"},
            {"role": "user", "content": thoughts_prompt},
        ],
    )
    thoughts_reply = thoughts_response['choices'][0]['message']['content'].strip()
    thoughts = thoughts_reply.split('\n')[0] if thoughts_reply else ""

    # Generate objectives using the OpenAI API
    # Use the sentiment of the question to determine the prompt to use for the objectives response generation
    objectives_prompt = f"List your actionable objectives based on your {sentiment} in response to {question}?"
    objectives_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        temperature=0.1,
        max_tokens=100,
        messages=[
            {"role": "system", "content": "As the embodiment of " + alignment + " " + status + "."},
            {"role": "user", "content": objectives_prompt},
        ],
    )
    objectives_reply = objectives_response['choices'][0]['message']['content'].strip()
    objectives = objectives_reply.split('\n')[0] if objectives_reply else ""

    # Store the response in the cache
    cache[question] = cache.get(question, {})
    cache[question][character] = {'response': response_text, 'objective': objectives, 'thought': thoughts}

    return character, response_text, thoughts, objectives




# extract_keywords function to extract keywords from a string of text using the RAKE algorithm and TextBlob sentiment analysis
# Returns a list of keywords that have a sentiment greater than 0.2 or less than -0.2 (i.e. the keyword is positive or negative)

def extract_keywords(text: str) -> list:
    r.extract_keywords_from_text(text)
    ranked_phrases = r.get_ranked_phrases()
    sentiment_keywords = []
    for phrase in ranked_phrases:
        sentiment = TextBlob(phrase).sentiment.polarity
        if abs(sentiment) > 0.2:
            sentiment_keywords.append(phrase)

    return sentiment_keywords


def save_response(alignment: str, question: str, response: str, thought: str, objective: str, keywords: list):
    file_path = f"{alignment}_responses.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        data = {}

    data[question] = {
        "response": response,
        "thought": thought,
        "objective": objective,
        "keywords": keywords
    }

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def chat(question: str) -> str:
    with ThreadPoolExecutor(max_workers=3) as executor:
            # Add a delay before generating each response
        time.sleep(1)  # pause for 1 second
        future_to_alignment = {executor.submit(generate_response, question, alignment): alignment for alignment in alignments}
        responses = []
        for future in concurrent.futures.as_completed(future_to_alignment):
            alignment = future_to_alignment[future]
            try:
                character, response, thought, objective = future.result()
                responses.append(f"{character}: {response}")
                keywords = extract_keywords(question)
                save_response(alignment, question, response, thought, objective, keywords)
            except Exception as exc:
                print(f"{alignment} chatbot generated an exception: {exc}")

    with open(cache_file_path, 'w') as f:
        json.dump(cache, f)

    return "\n\n".join(responses)

css = """
:root {
  --background-fill-secondary: white !important;
  --shadow-drop-lg: white !important;
  --block-label-border-width: 0px; */
  --block-label-text-color: white */
  --block-label-margin: 0; */
  --block-label-padding: var(--spacing-sm) var(--spacing-lg); */
  --block-label-radius: calc(var(--radius-lg) - 1px) 0 calc(var(--radius-lg) - 0px) 0; */
  --block-label-right-radius: 0 calc(var(--radius-lg) - 0px) 0 calc(var(--radius-lg) - 0px); */
  --block-label-text-size: var(--text-md); */
  --block-label-text-weight: 0;
}
.hide-label .gradio-block-label {display: none;}
.hide-icon .gradio-image-icon {display: none;}

.gradio-input-section {
  --background-fill-secondary: white;
}

.gradio-content {
    background-color: white;
}

.gradio-input-section, .gradio-output-section {
    background-color: white;
    box-shadow: none;
}

.gradio-input, .gradio-output {
    border: none;
    box-shadow: none;
}

body, label, .gradio-textbox textarea {
    font-family: 'Comic Sans MS', 'Comic Sans';
    font-weight: bold;
}
"""

def respond(question, chat_history):
    # Use a ThreadPoolExecutor to generate responses in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_alignment = {executor.submit(generate_response, question, alignment): alignment for alignment in alignments}
        responses = []
        for future in concurrent.futures.as_completed(future_to_alignment):
            alignment = future_to_alignment[future]
            try:
                character, response, thought, objective = future.result()
                responses.append(f"{character} :\n\n {response}\n\n thoughts :\n {thought}\n\n objective :\n {objective}\n\n")
                keywords = extract_keywords(question)
                save_response(alignment, question, response, thought, objective, keywords)
            except Exception as exc:
                print(f"{alignment} chatbot generated an exception: {exc}")

    chat_history.append((question, "\n".join(responses)))
    return "", chat_history

# Create Gradio blocks with a title
with gr.Blocks(title="alignmeDADDY", theme=gr.themes.Monochrome(), css=css) as iface:
    # Define a chatbot component and a textbox component with chat names
    chatbot = gr.Chatbot(show_label=True, label='alignmeDADDY') 
    msg = gr.Textbox(show_label=False)

    # Use the submit method of the textbox to pass the function, 
    # the input components (msg and chatbot), 
    # and the output components (msg and chatbot) as arguments
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

# Launch the interface
iface.launch(share=True)