import os
import openai

prompt = """Please filter following data from the following text and put it in a table. The table should have following columns:

Official Company Name | Tax number/UID | Company number | Telephone | Website | Email | Address Street | Address PLZ | Address City | Address Country | Revenue | Employees | sector

If the data is not available please leave the field empty.
The text starts here:
"""

# Function to send a message to ChatGPT and get a response
def send_message(message):
    response = openai.Completion.create(
        engine="text-davinci-002",  # You can choose other engines as well
        prompt=message,
        max_tokens=500,  # Adjust the max_tokens as needed
    )
    return response.choices[0].text

# Set your OpenAI API key here
api_key = ''
# read api key from file
with open('api_key', 'r') as f:
	api_key = f.read()


# locate raw txt files in folder

folder = 'companies_raw_files'

# get all files in folder
files = os.listdir(folder)

# loop through files
for file in files:
	# open file and read text
	with open(folder + '/' + file, 'r', encoding='utf-8') as f:
		text = f.read()
	# append prompt at the beginning of the text
	text = prompt + text
	# send text to chatgpt
	response = send_message(text)
	# write response to file
	with open('companies_prompt_files/' + file, 'w', encoding='utf-8') as f:
		f.write(response)
