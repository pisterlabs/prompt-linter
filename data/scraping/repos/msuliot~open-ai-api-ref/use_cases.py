import Helper
import json
import openai

# get keys from .env file
import os
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
#####################################################

# baseSentimentPrompt = "create an abstract painting expressing the sentiment of "
# baseSummaryPrompt = "create a oil painting showing the "

# # 1
# transcript = Helper.audio_to_text("audio/happy.m4a") # convert audio to text
# response = Helper.get_chat_completion(Helper.create_prompt(transcript)) # create prompt with text and instructions on how to respond
# data = json.loads(response)
# sentiment = data["sentiment"]
# Helper.create_image("create",baseSentimentPrompt + sentiment) # create image from response


# # 2
# transcript = Helper.audio_to_text("audio/mad.m4a")
# response = Helper.get_chat_completion(Helper.create_prompt(transcript))
# data = json.loads(response)
# summary = data["summary"]
# Helper.create_image("create",baseSummaryPrompt + summary) # create image from response


# 3 - Youtube video to text
transcript = Helper.audio_to_text("audio/bard.m4a")
print(transcript)
response = Helper.get_chat_completion(Helper.create_prompt_for_video(transcript))
print(response)


# 4 
# response = Helper.get_chat_completion("was there a housing crash around 2006 and what caused it?","gpt-4",0)
# print(response)


# 5 
# your_name = "Michael"
# response = Helper.get_chat_completion("Create a short story about a human named, " + your_name + ", trying to teach people the benefits of artificial intelligence","gpt-3.5-turbo",1.1)
# print(response)





