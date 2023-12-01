# from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
# from langchain import OpenAi
import subprocess
import json
import sys
import os
os.environ["OPENAI_API_KEY"]="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"

# Execute the Node.js script as a subprocess
node_output = subprocess.check_output(['node', 'nodescript.js'], universal_newlines=True)

# Parse the output from Node.js
data = json.loads(node_output)

# Access the exported data
restaurant = data['Restaurants']
restaurantImage = data['RestaurantImage']
review = data['Reviews']
reviewImage = data['ReviewImages']
user = data['Users']

# Use the data as needed in your Python script
print(restaurant)
print(restaurantImage)
print(review)
print(reviewImage)
print(user)

