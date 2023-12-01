from flask import Flask, render_template, jsonify, request
import openai
import re
from prompts import prompts
from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

app = Flask(__name__)
initialCall = True
currentDescription = ""



# Initialize OpenAI API with your API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Define a dictionary to store user progress data
user_data = {}

# Global variable to track initialization status
initialized = False

@app.route('/')
def index():
    global initialized
    global currentDescription
    if initialized:
        # Initialization has already been done, return JSON response
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(story=user_data['story'], choices=user_data['choices'])
        # Initialization has already been done, return HTML response
        else:
            return render_template('index.html', story=user_data['story'], choices=user_data['choices'])
    else:
        # Initialization code
        systemRoleAuto = prompts['index']['System']
        promptStory = prompts['index']['story']
        storyResponse = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{systemRoleAuto}"},
            {"role": "user", "content": f"{promptStory}"},
            #{"role": "assistant", "content": f"{contentAssistant}"},
        ],
        max_tokens= 1500,
    )
        story = storyResponse.choices[0].message['content']
        currentDescription = story
        choicesPrompt  = prompts['index']['choices']
        choiceResponse = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{systemRoleAuto}"},
            {"role": "user", "content": f"{story} {choicesPrompt}"},
            #{"role": "assistant", "content": f"{contentAssistant}"},
        ],
        max_tokens= 1500,
    )
        #Insert <p> tags around each paragraph
        formatted_story = format_story(story)
        user_data['story'] = formatted_story
        user_data['choices'] = choiceResponse.choices[0].message['content']
        initialized = True
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify(story=story, choices=user_data['choices'])
        else:
            return render_template('index.html', story=story, choices=user_data['choices'])

# Define a route to handle user choices and update the story
@app.route('/next-page/<choice>')
def next_page(choice):
    systemRoleAuto = prompts['next-page']['System']
    originalStory = user_data['story'] + "\n" + choice
    contentAssistant = prompts['next-page']['storyAssistant']
    contentAssistantChoices = prompts['next-page']['choicesAssistant']
    prompt_story = originalStory + "\n" + prompts['next-page']['story']
    response_story = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{systemRoleAuto}"},
            {"role": "user", "content": f"{prompt_story}"},
            {"role": "assistant", "content": f"{contentAssistant}"},
        ],
        max_tokens= 1500,
    )
    prompt_choices = originalStory + response_story.choices[0].message['content'] + "\n" + prompts['next-page']['choices']
    response_choices = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{systemRoleAuto}"},
            {"role": "user", "content": f"{prompt_choices}"},
            {"role": "assistant", "content": f"{contentAssistantChoices}"},
        ],
        max_tokens= 1500,
    )
    story = response_story.choices[0].message['content']
    choices = response_choices.choices[0].message['content']
    # get summary of previous story and actions by gpt-3.5-turbo and original story
    prompt_summary = originalStory + "\n" + prompts['next-page']['summary']
    response_summary = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{systemRoleAuto}"},
            {"role": "user", "content": f"{prompt_summary}"},
            #{"role": "assistant", "content": f"{contentAssistant}"},
        ],
        max_tokens= 1500,
    )

    formatted_story = format_story(story)
    user_data['story'] = formatted_story
    user_data['choices'] = choices
    user_data['summary'] = response_summary.choices[0].message['content']
    return jsonify(story=formatted_story, choices=choices, summary=user_data['summary'])

def format_story(story):
    # Split the text into paragraphs using a regular expression
    paragraphs = re.split(r"\n\s*\n", story)
    #Insert <p> tags around each paragraph
    formatted_story = "\n".join([f"<p>{paragraph}</p>" for paragraph in paragraphs])
    return formatted_story

if __name__ == '__main__':
    app.run(debug=True)
