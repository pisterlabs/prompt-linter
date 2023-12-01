import openai
import os
import re


def read_transcript(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def split_transcription(transcription, max_length=4000):
    # Split the transcription into chunks
    words = transcription.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) > max_length:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def parse_chunk_with_gpt(chunk, openai_api_key):
    openai.api_key = openai_api_key

    response =openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant capable of understanding and speaking Italian."},
            {"role": "user", "content": f"Trasforma il seguente verbale di riunione in un resoconto strutturato in italiano:\n\n{chunk}"}
        ]

    )

    print(response.choices[0].message.content)
    print("total tokens",response.usage.total_tokens)
    print("-------------------")

    return response.choices[0].message.content, response.usage.total_tokens

def parse_chunks_with_gpt(chunks, openai_api_key):
    openai.api_key = openai_api_key

    messages=[
        {"role": "system", "content": "You are a helpful assistant capable of understanding and speaking Italian."},
        {"role": "user", "content": f"Ti invio i pezzi della trascrizione della riunione:\n\n###{chunks[0]}###"}
    ]
    for chunk in chunks:
        #append to messages the last chunk
        messages.append({"role": "user", "content": f"###{chunk}###"})
        
    messages.append({"role": "user", "content": f"crea il verbale della riunione"})
        

    response =openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages

    )

    print(response.choices[0].message.content)
    print("total tokens",response.usage.total_tokens)
    print("-------------------")

    return response.choices[0].message.content, response.usage.total_tokens



def generate_meeting_minutes(file_path):
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OpenAI API key not found in environment variables.")

    transcription = read_transcript(file_path)
    chunks = split_transcription(transcription)
    meeting_minutes = []
    total_tokens = 0

    # for chunk in chunks:
    #     parsed_chunk, tokens = parse_chunk_with_gpt(chunk, openai_api_key)
    #     meeting_minutes.append(parsed_chunk)
    #     total_tokens += tokens

    meeting_minutes, total_tokens = parse_chunks_with_gpt(chunks, openai_api_key)

    return '\n'.join(meeting_minutes), total_tokens

# File path to the transcript
file_path = 'data\\transcript.txt'

# Generate the meeting minutes
meeting_minutes, total_tokens = generate_meeting_minutes(file_path)
cost_per_1000_tokens = 0.002  # Replace with the current cost per 1,000 tokens
cost = (total_tokens / 1000) * cost_per_1000_tokens

print(meeting_minutes)
print(f"\nTotal tokens processed: {total_tokens}")
print(f"Estimated cost: ${cost:.4f}")


#save the meeting minutes to a file
with open('meeting_minutes.txt', 'w') as file:
    file.write(meeting_minutes)
    
