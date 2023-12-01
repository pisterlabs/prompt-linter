#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This updated code first reads JSON files and extracts transcriptions, 
# then calculates token counts for each transcription using the GPT2 tokenizer. 
# It calculates the costs for GPT-3.5, GPT-4-8K, and GPT-4-32K, 
# and generates refined transcripts using the GPT-3.5-turbo and GPT-4 models. 
# Finally, it exports the DataFrame to a CSV file.

import os
import json
import csv
import pandas as pd
from transformers import GPT2Tokenizer
import openai

# Set the API key for OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Get a list of all the JSON files in the output directory
output_folder_path = "./output/"
json_files = [f for f in os.listdir(output_folder_path) if f.endswith(".json")]

transcriptions = []
for json_file in json_files:
    with open(os.path.join(output_folder_path, json_file)) as f:
        data = json.load(f)
    transcription = data["transcription"]
    transcriptions.append(transcription)

# Save the transcriptions to a CSV file
with open("output.csv", "w", encoding="utf-8", newline='') as f:
    writer = csv.writer(f)
    for transcription in transcriptions:
        writer.writerow([transcription])

# Load the GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define a function to count the tokens in a string
def count_tokens(text):
    tokens = tokenizer.encode(text)
    return len(tokens)

# Read the CSV file into a DataFrame
df = pd.read_csv("output.csv", header=None, names=["transcription"])

# Apply the count_tokens function to each row of the "transcription" column
df["token_count"] = df["transcription"].apply(count_tokens)

# Add three new columns to the DataFrame for GPT-3.5, GPT-4-8K, and GPT-4-32K
df["GPT-3.5"] = df["token_count"].apply(lambda x: round(x / 500 * 0.002, 2))
df["GPT-4-8K"] = df["token_count"].apply(lambda x: round(x / 500 * 0.045, 2))
df["GPT-4-32K"] = df["token_count"].apply(lambda x: round(x / 500 * 0.09, 2))

# Calculate the sum of each column
sum_gpt_3_5 = df["GPT-3.5"].sum()
sum_gpt_4_8k = df["GPT-4-8K"].sum()
sum_gpt_4_32k = df["GPT-4-32K"].sum()

# Print the sums of each column
print("Sum of GPT-3.5 column:", sum_gpt_3_5)
print("Sum of GPT-4-8K column:", sum_gpt_4_8k)
print("Sum of GPT-4-32K column:", sum_gpt_4_32k)

# Generate GPT-3.5-turbo refined transcripts
GPT3_transcript = []
for i in range(len(df.transcription)):
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "assistant", "content": "You are going to get the raw transcript of a phone call.  You need to refine the transcript by carefully distinguishing who made the call from who received the call. Refer to the person who made the call as 'Caller' and the person who recieved the call as 'Receptionist'. Here is the raw transcript:" + df.transcription[i]}
      ]
    )
    entry = str(completion.choices[0].message.content)
    GPT3_transcript.append(entry)

# Add the GPT-3.5-turbo refined transcripts to the DataFrame
df["GPT3_transcript"] = pd.Series(GPT3_transcript)

# Generate GPT-4 refined transcripts (Replace "gpt-4" with the actual GPT-4 model name when available)
GPT4_transcript = []
for i in range(len(df.transcription)):
    completion = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
        {"role": "assistant", "content": "You are going to get the raw transcript of a phone call.  You need to refine the transcript by carefully distinguishing who made the call from who received the call. Refer to the person who made the call as 'Caller' and the person who recieved the call as 'Receptionist'. Here is the raw transcript:" + df.transcription[i]}
      ]
    )
    entry = str(completion.choices[0].message.content)
    GPT4_transcript.append(entry)

# Add the GPT-4 refined transcripts to the DataFrame (Replace "GPT4_transcript" with the actual GPT-4 model name when available)
df["GPT4_transcript"] = pd.Series(GPT4_transcript)

# Export the DataFrame to a CSV file
df.to_csv("output_with_gpt.csv", index=False)


# In[ ]:




