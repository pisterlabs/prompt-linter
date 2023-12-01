import openai
from PyPDF2 import PdfReader
import textwrap
import json
import os

openai.api_key = "OPENAI_API_KEY"

def read_pdf(file):
    with open(file, "rb") as filehandle:
        pdf = PdfReader(filehandle)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    print("PDF successfully read")
    return text

def chunk_text(text, max_length=8000):
    chunks = textwrap.wrap(text, max_length)
    print(f"Text successfully chunked into {len(chunks)} pieces")
    return chunks

def parse_chunk(chunk):
    prompt = "As the narrator of this epic adventure, your role is to translate the raw transcripts into a lively and engaging narrative, keeping your response short and to the point. Assume the persona of Samgine, and weave the tale of Illion, Feng, Klagnut, Althea and yourself as you journey through realms unknown. Here, 'Illion' is Chrys Mordin, 'Feng' is J.D.P.Croy, 'Klagnut' is Allix Harrison-D'Arcy, you are 'Samgine' (Alfie Dennen), and 'Althea' is Charlotte.  Important: The narrative should focus solely on the actions and interactions of the characters, never the players themselves. Keep the narrative present tense and be sure not to introduce new context. Disregard any sections that don't advance the narrative and ignore dialogues that don't contribute to the progression of the story.  Now, using the spirit of imagination that fuels this game, bring this piece of our adventure to life in an amusing and exciting summary. Make the readers feel as if they are there, experiencing the same thrills and suspense that we did. Here is the transcript excerpt: "

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
    
    text = read_pdf(file)
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

process_text("gametranscripttest.pdf")