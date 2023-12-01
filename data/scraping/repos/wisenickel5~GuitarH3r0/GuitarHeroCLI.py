from openai import AzureOpenAI
import os
import requests
from pathlib import Path
import pandas as pd

from guitar_hero_utils import normalize_text, get_embedding, cosine_similarity
from transcript_utils import (get_transcript_data, get_transcript_turns, create_transcript_subsets,
                              extract_agent_responses, convert_subsets_to_messages)

# Setup parameters to authenticate with OpenAI API
api_key = os.getenv("AZURE_OPENAI_API_KEY")
base_url = os.getenv("AZURE_OPENAI_ENDPOINT")
client = AzureOpenAI(
    api_version="2023-07-01-preview",
    api_key=api_key,
    azure_endpoint=base_url
)

# Verify the OpenAI API can be accessed
url = f"{base_url}/openai/deployments?api-version=2022-12-01"
r = requests.get(url, headers={"api-key": api_key})
print("Confirming that an entry for Embedding & Language models exist in API response:\n", r.text)

# Parse transcript
transcript_path = Path('/Users/dylanalexander/Repos/GuitarH3r0/Call-Center-Transcript.CSV')
transcript_df: pd.DataFrame = get_transcript_data(str(transcript_path))
turns, speakers = get_transcript_turns(transcript_df)
transcript_subsets = create_transcript_subsets(turns, speakers)
actual_agent_responses = extract_agent_responses(transcript_subsets)
all_transcript_messages = convert_subsets_to_messages(transcript_subsets)

# Score-Tracking variables
num_of_subsets = len(transcript_subsets)
embedding_proximity_sum = 0

# Iterate over all messages (transcript_subsets) and actual agent responses from the transcript
for message, agent_response in zip(all_transcript_messages, actual_agent_responses):
    # Generate the next best sentence with ChatGPT
    next_best_sentence = client.chat.completions.create(model="Guitar-H3r0-GPT-Turbo", messages=message)

    print("\n\n***********************************************"
          "\nActual Agent Response: ", f'{agent_response}')
    print("'Next-Best-Sentence' According to ChatGPT:\n", next_best_sentence.choices[0].message.content)
    print("*Normalized* 'Next-Best-Sentence' According to ChatGPT:\n",
          normalize_text(next_best_sentence.choices[0].message.content))

    # Generate Embeddings for sentences
    gpt_response_embedding = get_embedding(next_best_sentence.choices[0].message.content, client)
    actual_agent_response_embedding = get_embedding(agent_response, client)

    # Calculate distance between embeddings
    dist_0_1: float = cosine_similarity(gpt_response_embedding, actual_agent_response_embedding)
    # NOTE: We are evaluating how effectively ChatGPT can replace an agent in a call center.
    # A score close to 1 would mean that ChatGPT did a good job emulating what an
    # agent would actually say in a Call Center.
    print(f'Distance (gpt_response_embedding, actual_agent_response_embedding) = {dist_0_1:0.3}')

    embedding_proximity_sum += dist_0_1

# Determine final score
final_avg = embedding_proximity_sum / num_of_subsets
print("\n\n****** Final Guitar H3r0 Score ******"
      "\nFinal Score =  Embedding Proximity Sum / [# of subsets derived from transcript]"
      f"\nFinal Score = {round(embedding_proximity_sum, 3)} / {round(num_of_subsets, 3)} = {round(final_avg, 3)}")
