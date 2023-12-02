from datetime import datetime
from chad_app.functions.openai_class import OpenAIGPT


# Make the api instances for the account lists
def make_user_api(api_details):
	# Setup instances and build the dictionaries
	openai_app = OpenAIGPT()
	openai_app.setup_api(api_details[0], api_details[1])

	return openai_app