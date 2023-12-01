import openai
import json
import os
import re
import tiktoken
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Specify the file path
file_path = 'samples/microsoftTeamsGeneratedTranscript.vtt'

# set OpenAI info
openai.api_key = os.getenv("OPENAI_API_KEY")
aiModel = "gpt-3.5-turbo"
tokenLimit = 3000

# simplify the encoding capture to one time
try:
    encoding = tiktoken.encoding_for_model(aiModel)
except KeyError:
    encoding = tiktoken.get_encoding("cl100k_base")

# read VTT file
def read_clean_file(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        # Read the contents of the file into a string
        file_contents = file.read()
        
    # Split the string into lines
    lines = file_contents.split('\n')

    # Check if the first line is "WEBVTT" and remove it if so
    if lines[0].strip() == "WEBVTT":
        lines = lines[1:]

    # Join the remaining lines back into a string
    cleaned_contents = '\n'.join(lines)
    
    # Remove the timestamps using regular expression
    cleaned_contents = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3}\s-->\s\d{2}:\d{2}:\d{2}\.\d{3}\n', '', cleaned_contents)
    
    # Define the regular expression pattern for the hashes
    pattern = r'\b[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}-\d\b'

    # Remove the hashes from the content using the sub() method
    cleaned_contents = re.sub(pattern, '', cleaned_contents)
    
    # Split the string into paragraphs using double newline
    paragraphs = cleaned_contents.split('\n\n')

    # For each paragraph, split by newlines, strip, and join
    processed_paragraphs = [' '.join([line.strip() for line in paragraph.split('\n') if line.strip()]) for paragraph in paragraphs]

    # Join the processed paragraphs using a single newline
    cleaned_contents = '\n'.join(processed_paragraphs)

    return cleaned_contents

# Returns the number of tokens used by a list of messages.
def num_tokens_from_messages(messages, model="gpt-3.5-turbo"):
#   print ("Getting num tokens for messages", messages)

  if model == "gpt-3.5-turbo":
        if isinstance(messages, str):
                return len(encoding.encode(messages))
        elif isinstance(messages, list):
            num_tokens = 0
            for message in messages:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens
        else:
            raise ValueError("The input must either be a single string or a list of message dictionaries.")
  else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.
  See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")

def split_to_fit_token_limit(message, max_tokens=3000):
    valid_messages = []
    words = message.split()
    current_message = ""

    while words:
        word = words.pop(0)
        temp_message = f"{current_message} {word}".strip()

        # If adding the current word exceeds the token limit
        if num_tokens_from_messages(temp_message) > max_tokens:
            valid_messages.append(current_message)
            current_message = word
        else:
            current_message = temp_message

    # Add any remaining content
    if current_message:
        valid_messages.append(current_message)

    return valid_messages

# Will batch the calls into a single API call
def summarize_vtt(vtt_string):
    
    print("Splitting the transcript into batches of %d tokens..."%tokenLimit)
    chatInputs = split_to_fit_token_limit(vtt_string, tokenLimit)
    print("Finished splitting the transcript into batches")
    
    # where we keep track of the detected bullet points
    detectedBulletPoints = ""
    
    # get bullet points for each batch of conversation
    for i in range(len(chatInputs)):
        # print(listOfChatInputs[i])
        print("Parsing a batched call for bullet points. Token quantity of: ", num_tokens_from_messages(chatInputs[i]))
        
        batchedChatMessageTemplate = [
            {
                "role": "system",
                "content": "You work for a custom software company, Buildable, as a project manager. Your job is to create concise recaps that are shared to clients and internally. You recieve meeting transcripts and summarize the important aspects of the meeting clearly. Always use a professional tone and attitude."
            },
            {
                "role": "user",
                "content": chatInputs[i]
            },
            {
                "role": "user",
                "content": "Give me bullet points to summarize this segment of a conversation. Each bullet point should be either a unanswered question, an action item, or a key point. Label each bullet point with the appropriate label."
            },
        ]
        
        batchedResponse = openai.ChatCompletion.create(
            model=aiModel,
            messages= batchedChatMessageTemplate,
            temperature=1,
            max_tokens=200,
            top_p=1,
            frequency_penalty=0.25,
            presence_penalty=0.25
        )
        
        # concatenate the batched responses
        detectedBulletPoints += "\n%s"%batchedResponse['choices'][0]['message']['content']
    
    # Do a final summarization of the bullet points    
    finalChatMessage = [
        {
        "role": "system",
        "content": "You work for a professional software company as a program manager. You need to create concise recaps that can be shared to client and internal organizations. When provided with a transcript of a meeting, you summarize the important aspects of the meeting into key points, action items, and relevant follow up questions."
        },
        {
        "role": "user",
        "content": detectedBulletPoints
        },
        {
        "role": "user",
        "content": "First, write a clear summary of the meeting bullet points in 2-3 sentances, avoid redundant verbaige like 'key points include', 'action items include' and 'questions include'. Then, include a reorganization of the bullet points by their labels into 3 sections: key points, unanswered questions, and action items in that order. If there is any duplicated information, only include it once. Only add questions not answered by key pointst."
        },
    ]
    
    finalResponse = openai.ChatCompletion.create(
        model=aiModel,
        messages= finalChatMessage,
        temperature=1,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0
    )
    
    return finalResponse['choices'][0]['message']['content']

# Call the function to read the file into a string
print("Reading Microsoft Teams VTT transcript file...")
file_string = read_clean_file(file_path)
print("Finished cleaning transcript file...")

summarization = summarize_vtt(file_string)

print("\n\n--RESULSTING RECAP--\n\n")
print(summarization)