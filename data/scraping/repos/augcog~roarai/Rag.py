import openai
import pickle
from dotenv import load_dotenv
import os

load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

user_question = 'Generate a list of questions that could be asked about this document.'

system_prompts = [
    "You are an AI assistant. Provide a detailed answer so the user doesn’t need to search outside to understand the answer.",
    "You are an AI assistant. You will be given a task. You must generate a detailed and long answer.",
    "You are an AI assistant. User will give you a task. Your goal is to complete the task as faithfully as you can. While performing the task, think step-by-step and justify your steps."
]

#function reads all .pkl files in a given directory, extracts text segments from them, and returns a concatenated string of all these segments.
def get_documents(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.pkl'):
            with open(os.path.join(directory_path, filename), 'rb') as file:
                pkl = pickle.load(file)
                for item in pkl:
                    documents.append(item['Segment_print'])
    return documents
    #return ' '.join(documents)

my_documents_directory = "/Users/arnavjain/desktop/Defi_Lab/edugpt/Scrape_rst/Sawyer"
documents = get_documents(my_documents_directory)[1]

responses = []  # A list to hold all the responses

# Loop through each user question and corresponding system prompt and document retrieval function
for system_prompt in system_prompts:
    # Construct the messages list with the current system prompt, documents, and user question
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "system", 
            "content": f"{user_question}\n---\n{documents}"
        },
    ]

    # Get the response from OpenAI's model for the current set of messages
    openai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    
    # Add the response to the responses list
    responses.append(openai_response)
    for i in responses:
        print(i["choices"][0]["message"]["content"])
