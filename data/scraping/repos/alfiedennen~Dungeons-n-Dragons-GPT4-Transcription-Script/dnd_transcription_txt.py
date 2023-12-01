import openai
import textwrap
import json
import os

openai.api_key = "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"

def read_txt(file):
    with open(file, "r") as filehandle:
        text = filehandle.read()
    print("Text successfully read")
    return text

def chunk_text(text, max_length=8000):
    chunks = textwrap.wrap(text, max_length)
    print(f"Text successfully chunked into {len(chunks)} pieces")
    return chunks

def parse_chunk(chunk):
    prompt = "As the narrator of this epic adventure, your role is to translate the raw transcripts into a lively and engaging narrative. The transcript features 5 different people, one of whom is the dungeon master who often tells the story of what happens to the characters and to the world around them. Your task also includes distinguishing between game-related content and real-world interactions. The game-related content involves everything happening to the characters in the game, everything the dungeon master describes. Conversely, real-world interactions involve players discussing their dice rolls, referring to elements outside the game world like films, TV shows, or the platform they're playing on. When providing the narrative, focus on the game-related content, making the readers feel as if they are there, experiencing the same thrills and suspense that the characters did. You will receive the transcript in sections, so you should not treat the start or end of transcripts as beginnings or endings, you have received parts before and will receive more after this one. Separate the game world narrative and the out-of-game interactions distinctly. Now, here is the transcript excerpt: "

    message = {
        "role": "system",
        "content": prompt
    }

    message2 = {
        "role": "user",
        "content": chunk
    }

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",  # GPT4 model.
        messages=[message, message2]
    )

    # We obtain the content from the message here
    parsed_chunk = response['choices'][0]['message']['content']
    print("Chunk successfully parsed")
    return parsed_chunk


def process_text(file):
    print("Processing started...")
    
    if os.path.exists('progress.json'):
        with open('progress.json') as f:
            progress = json.load(f)
        print("Existing progress found")
    else:
        progress = {
            "chunk_idx": 0,
            "parsed_story": ''
        }
        print("No existing progress found")
    
    text = read_txt(file)
    chunks = chunk_text(text)
    
    for i in range(progress["chunk_idx"], len(chunks)):
        print(f"Processing chunk {i+1} out of {len(chunks)}")
        
        parsed_chunk = parse_chunk(chunks[i])
        progress["parsed_story"] += f"{parsed_chunk}\n"
        progress["chunk_idx"] = i+1
        
        print(f"Writing parsed chunk {i+1} to file...")
        with open('parsed_story.txt', 'w', encoding='utf-8') as out_file:
            out_file.write(progress["parsed_story"])
            
        print(f"Saving progress up to chunk {i+1}...")
        with open('progress.json', 'w') as f:
            json.dump(progress, f)
            
        print(f"Done with chunk {i+1}")

    print("Processing finished")

process_text("transcript.txt")
