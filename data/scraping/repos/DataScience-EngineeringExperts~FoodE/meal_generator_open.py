import os
import openai
from pymongo import MongoClient
from bson import ObjectId
from json import JSONEncoder
import logging
from datetime import datetime
from dotenv import load_dotenv
import json
import requests

# Load environment variables from .env file
load_dotenv()

# Accessing and validating environment variables
mongo_conn_string = os.getenv('mongo_conn_string')
openai_api_key = os.getenv('openai_api_key')
webhook_url = os.getenv('webhook_url')

if not mongo_conn_string or not openai_api_key:
    raise ValueError("Environment variables for MongoDB and OpenAI API key are required.")

# Configure OpenAI API key
openai.api_key = openai_api_key

# Global MongoDB connection
client = MongoClient(mongo_conn_string)
db = client.menumaker
log_collection = db.logs

class MongoDBHandler(logging.Handler):
    def emit(self, record):
        # Format the log record using the formatter
        log_entry = self.format(record)

        # Create a log document with a 'datetime' field
        log_document = {
            "log": log_entry,
            "datetime": datetime.now()  # Add the current datetime
        }

        # Insert the log document into MongoDB
        log_collection.insert_one(log_document)

# Create a custom logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a MongoDB log handler
mongo_handler = MongoDBHandler()
logger.addHandler(mongo_handler)

# Formatter for logs
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
mongo_handler.setFormatter(formatter)

class JSONEncoderCustom(JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return super(JSONEncoderCustom, self).default(o)


# Directly use provided lists
HealthLabel = [
    "fat-free", "low-fat-abs", "sugar-conscious", "low-sugar", "low-potassium", "kidney-friendly",
    "keto-friendly", "plant-based", "vegan", "vegetarian", "pescatarian", "paleo", "specific-carbs",
    "Mediterranean", "DASH", "dairy-free", "gluten-free", "wheat-free", "egg-free", "milk-free", "peanut-free",
    "tree-nut-free", "soy-free", "fish-free", "shellfish-free", "pork-free", "red-meat-free", "crustacean-free",
    "celery-free", "mustard-free", "sesame-free", "lupine-free", "mollusk-free", "alcohol-free", "no-oil-added",
    "no-sugar-added", "sulfite-free", "fodmap-free", "kosher", "alcohol-cocktail", "immuno-supportive"
]

DietLabel = [
    "balanced", "high-protein", "high-fiber", "low-fat", "low-carb", "low-sodium"
]

CautionLabel = [
    "gluten", "wheat", "eggs", "milk", "peanuts", "tree-nuts", "soy", "fish", "shellfish", "sulfites", "fodmap"
]

acceptable_tags = HealthLabel + DietLabel
avoidance_tags = CautionLabel

def send_teams_notification(message, webhook_url):
    """
    Sends a notification message to a Microsoft Teams channel.
    """
    headers = {"Content-Type": "application/json"}
    payload = {"text": message}
    try:
        response = requests.post(webhook_url, json=payload, headers=headers)
        response.raise_for_status()
        # logger.info(f"Message successfully sent to Teams: {message}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error sending message to Teams: {e}")


def format_meal_suggestions(doc):
    """
    Formats the meal suggestions document for a readable Teams message.
    """
    message_lines = [f"New meal suggestions generated for {doc['username']}:"]
    for suggestion in doc.get('suggestions', []):
        meal = suggestion.get('meal', 'No meal')
        sides = ', '.join(suggestion.get('sides', [])) or 'No sides'
        message_lines.append(f" - Meal: {meal}, Sides: {sides}")
    message_lines.append(f"Timestamp: {doc.get('timestamp', 'No timestamp')}")
    return '\n'.join(message_lines)


def generate_meals(num_days=1, username=None):
    meal_suggestions_list = []

    # Define 'current_dir' at the start of the function
    current_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        results_collection = db.results
        user_summary_collection = db.user_summary

        # Fetch the user summary for the specific user
        user_summary_data = user_summary_collection.find_one({'username': username})

        if not user_summary_data:
            logger.error(f"No user summary found for username: {username}")
            return

        # Convert user summary data to JSON string
        user_summary = json.dumps(user_summary_data, cls=JSONEncoderCustom)

        # Fetch the last 2 meal suggestions for the specific user
        last_2_documents = list(results_collection.find({'username': username}).sort([('_id', -1)]).limit(5))

        # Initialize an empty list to store meal history
        meal_suggestions_history = []

        # Load template for suggestions using JSON file
        with open(os.path.join(current_dir, 'meal_suggestions.json'), 'r') as file:
            meal_template = json.load(file)
            meal_suggestions = json.dumps(meal_template, cls=JSONEncoderCustom)
        # logger.debug(f'Meal Suggestions Template is as follows: {meal_suggestions}')

        if len(last_2_documents) > 0:
            for doc in last_2_documents:
                if 'suggestions' in doc:
                    for suggestion in doc['suggestions']:
                        if 'meal' in suggestion:
                            meal_suggestions_history.append(suggestion['meal'])

        meal_suggestions_history = '\n'.join(meal_suggestions_history)

    except IOError as e:
        logger.error(f"File error: {e}")
        return
    except Exception as e:
        logger.error(f"MongoDB error: {e}")
        return

    for _ in range(num_days):
        try:

            completion = openai.ChatCompletion.create(
                model="gpt-4-1106-preview",
                response_format={"type": "json_object"},
                messages=[ENTER YOUR PROMPT HERE]
            )

            # Parse the response
            #print(completion)
            parsed_response = json.loads(completion.choices[0].message['content'])
            #logger.debug(f'OpenAI Response Content: {completion.choices[0].message["content"]}')

        except Exception as e:
            logger.error(f"API call or JSON parsing error: {e}")
            continue

        try:
            # If parsed_response is a list, then it's already what we need
            if isinstance(parsed_response, list):
                meal_suggestions = parsed_response
            # If parsed_response is a dictionary, then extract the "suggestions" key
            elif isinstance(parsed_response, dict):
                meal_suggestions = parsed_response['suggestions']
            else:
                logger.error(f"Unknown data type for parsed_response: {type(parsed_response)}")
                continue

            meal_suggestions_list.extend(meal_suggestions)

            # Create a single document for the user with all suggestions
            current_time = datetime.now()

            user_meal_suggestion_doc = {
                'username': username,
                'suggestions': meal_suggestions_list,
                'timestamp': current_time  # Use the formatted datetime
            }

            collection = db.results
            # logger.debug(f"Successfully connected to MongoDB for {user_meal_suggestion_doc}")
            # print(f"User Doc saved was: {user_meal_suggestion_doc}")

            # insert the document into MongoDB
            collection.insert_one(user_meal_suggestion_doc)
            #logger.info("Successfully inserted meal suggestions into MongoDB")

            # Format the document for readability
            formatted_message = format_meal_suggestions(user_meal_suggestion_doc)

            # send a formatted notification to Teams
            send_teams_notification(formatted_message, webhook_url)

        except KeyError as e:
            logger.error(f"Key not found in JSON: {e}")
            continue
        except Exception as e:
            logger.error(f"General error with JSON: {e}")
            continue

        logger.debug(f"Full API response: {json.dumps(completion)}")

    # Close MongoDB connection
    client.close()

    # print("successful transmit of data...")
    return meal_suggestions_list


def main():
    # defaults to admin user if no username found
    user = "admin"

    # Number of days for which you want to generate meal plans
    num_days = 1

    # Call the function
    generate_meals(num_days, user)


if __name__ == "__main__":
    main()
