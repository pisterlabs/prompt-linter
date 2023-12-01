import os
import dotenv
import openai
import json
import random
from rake_nltk import Rake
from nltk.sentiment import SentimentIntensityAnalyzer

dotenv.load_dotenv()
CHATCOMPLETE_API_KEY = os.getenv("CHATCOMPLETE_API_KEY")
sia = SentimentIntensityAnalyzer()

class ChatCompleteAPI:
    def __init__(self, prompt):
        self.prompt = prompt
        self.max_tokens = 100
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.openai_api_key

    def get_response(self):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": self.prompt}
                ],
                max_tokens=self.max_tokens
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            return str(e)

class Chatbot:
    def __init__(self, name, character, prompt, next_node_positive=None, next_node_negative=None):
        self.name = name
        self.character = character
        self.prompt = prompt
        self.next_node_positive = next_node_positive
        self.next_node_negative = next_node_negative
        self.api = ChatCompleteAPI(prompt)
        self.rake = Rake()

    def extract_keywords(self, text):
        self.rake.extract_keywords_from_text(text)
        return self.rake.get_ranked_phrases()

    def respond(self, user_input, previous_responses=[]):
        # Prepend the character role to the user's input
        prompt = f"You are {self.character}. {user_input}"

        # Include previous responses if available
        if previous_responses:
            prompt = ' '.join(previous_responses) + ' ' + prompt

        self.api.prompt = prompt
        response = self.api.get_response()
        sentiment = sia.polarity_scores(response)  # change to response
        sentiment_label = 'positive' if sentiment['compound'] > 0 else 'negative'
        response = f"{self.name} detected {sentiment_label} sentiment: {response}"
        self.next_node = self.next_node_positive if sentiment_label == 'positive' else self.next_node_negative
        return response

    def record_to_json(self, filename, user_input, response):
        with open(filename, 'a') as f:
            json.dump({
                'name': self.name,
                'user_input': user_input,
                'keywords': self.extract_keywords(user_input),
                'response': response,
                'response_keywords': self.extract_keywords(response),
            }, f)

def get_input(prompt, default):
    user_input = input(prompt)
    return user_input if user_input else default

def create_chatbots(num_bots, user_input, summary_bot_1, summary_bot_2):
    chatbots = [None] * num_bots  # Create a list with num_bots elements, all None
    for i in range(num_bots):
        name = get_input(f"Enter a name for chatbot {i+1}: ", default_values['name'] + f" {i+1}")
        character = get_input(f"Enter a character role for chatbot {i+1}: ", default_values['character'] + f" {i+1}")
        temperature = float(get_input(f"Enter a temperature for chatbot {i+1}: ", default_values['temperature']))
        chatbots[i] = Chatbot(name, character, user_input)  # Create the chatbot
    # Now that all the chatbots are created, assign the next_node_positive and next_node_negative
    for i in range(num_bots):
        chatbots[i].next_node_positive = chatbots[(i + 1) % num_bots] if i != num_bots - 1 else summary_bot_1
        chatbots[i].next_node_negative = chatbots[(i + 1) % num_bots] if i != num_bots - 1 else summary_bot_2
    return chatbots


def create_summary_bot(summary_bot_number):
    summary_bot_name = get_input(f"Enter a name for the summary bot {summary_bot_number}: ", default_values[f'summary_bot_{summary_bot_number}_name'])
    summary_bot_character = get_input(f"Enter a character role for the summary bot {summary_bot_number}: ", default_values[f'summary_bot_{summary_bot_number}_character'])

    summary_bot = Chatbot(summary_bot_name, summary_bot_character, user_input)
    summary_bot.api.max_tokens = 50
    summary_bot.api.openai_api_key = CHATCOMPLETE_API_KEY
    return summary_bot

test_questions = {
    'philosophy': 'What is the meaning of life?',
    'poetry': 'Write me a poem. About ' + f'{random.choice(["love", "nature", "life", "death", "the universe", "the human condition"])}',
    'todo': 'create a todo list',
    'fact': 'what is an interesting fact?',
    'joke': 'Tell me a joke.'
}

default_values = {
    'name': 'Skeleton',
    'character': 'You are a skeleton only concerned about your bones.',
    'temperature': 0.5,
    'max_tokens': 100,
    'summary_bot_1_name': 'OldSkeleton',
    'summary_bot_1_character': 'You are a wise skeleton who has seen many things.',
    'summary_bot_1_temperature': 0.5,
    'summary_bot_1_max_tokens': 100,
    'summary_bot_2_name': 'YoungSkeleton',
    'summary_bot_2_character': 'You are a young skeleton who is eager to learn.',
    'summary_bot_2_temperature': 0.5,
    'summary_bot_2_max_tokens': 100,
}

user_input = get_input("Enter your query: ", random.choice(list(test_questions.values())))
num_bots = int(get_input("Enter the number of chatbots: ", '1'))

summary_bot_1 = create_summary_bot(1)
summary_bot_2 = create_summary_bot(2)
chatbots = create_chatbots(num_bots, user_input, summary_bot_1, summary_bot_2)

filename = 'chatbot_responses.json'
current_node = chatbots[0]
previous_responses = []  # Track previous responses for summary bots
while current_node is not None:
    response = current_node.respond(user_input, previous_responses)
    print(response)
    current_node.record_to_json(filename, user_input, response)
    previous_responses.append(response)  # Store the response to pass to the next node
    current_node = current_node.next_node


# Once done with chatbots, move to the summary bots
while current_node in [summary_bot_1, summary_bot_2]:
    response = current_node.respond(user_input, previous_responses)
    print(response)
    current_node.record_to_json(filename, user_input, response)
    previous_responses.append(response)
    current_node = current_node.next_node


