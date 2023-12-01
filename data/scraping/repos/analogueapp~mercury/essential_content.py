import openai
import re
import os
import argparse
from db_config import essential_contents_collection
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_KEY")

def get_essential_media(n, genre, media, modifier=''):    
    prompt_text = (f"Generate a list of the {n} most influential {genre} {media} of all time. {modifier}")
    completion = openai.Completion.create(model="gpt-3.5-turbo-instruct", prompt=prompt_text, max_tokens=4000, temperature=0.6, n=1)   
            
    raw_text = completion.choices[0].text
     
    contents = []
    if media == "books":
        contents = parse_books(raw_text)
    elif media == "films":
        contents = parse_films(raw_text)                
    elif media == "podcast shows":
        contents = parse_podcasts(raw_text)
    elif media.endswith(' podcast episodes'):
        show = media.removesuffix(' podcast episodes')
        contents = parse_episodes(raw_text, show)
    
    # Insert contents into MongoDB collection
    for item in contents:
        item['media_type'] = 'podcast episode' if media.endswith('podcast episodes') else media[:-1]
        essential_contents_collection.insert_one(item)    
    return contents

def parse_books(text):
    # Regular expression to match the pattern of the list items
    pattern = r'(?:(?<=\d\.\s")|(?<=\d\.\s))(?P<title>.*?)(?:(?=" by)|(?= by))"?\sby\s(?P<authors>.+?)(?=\n|$)'
    matches = re.findall(pattern, text)
    
    books = []
    for match in matches:
        title = match[0].strip('"')  # Remove quotes if present
        authors = [author.strip() for author in match[1].split("and")]  # Split multiple authors by "and"
        books.append({"title": title, "authors": authors})
    
    return books

def parse_films(text):
    # Regular expression to match the pattern of the list items for films
    pattern = r'(?:(?<=\d\.\s))(?P<title>.*?)(?=\s\()\s\((?P<year>\d{4})\)'
    matches = re.findall(pattern, text)
    
    films = []
    for match in matches:
        title = match[0].strip()  # Remove any extra spaces
        year = match[1]
        films.append({"title": title, "year": year})
    
    return films

def parse_podcasts(text):
    # Regular expression to match the pattern of the list items for podcast shows
    pattern = r'(?<=\d\.\s)(?P<title>.+?)(?=\n|$)'
    matches = re.findall(pattern, text)
    
    podcasts = []
    for match in matches:
        title = match.strip().strip('"')  # Remove any extra spaces
        get_essential_media(15, '', f'"{title}" podcast episodes', "Include just the title of the episode.")
        podcasts.append({"title": title})
    
    return podcasts

def parse_episodes(text, show):
    # Regular expression to match the pattern of the list items for podcast episodes
    pattern = r'(?:(?<=:\s)|(?<=\d\.\s"))(.*?)(?="|\s\(|$)'
    matches = re.findall(pattern, text)
    
    episodes = []
    for match in matches:
        title = match.strip().strip('"')  # Remove any extra spaces
        episodes.append({"title": title, "show": show.strip('"')})
    
    return episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get top media lists.')
    parser.add_argument('-n', type=int, help='Number of items in the list', default=50)
    parser.add_argument('-g', '--genre', type=str, help='Specify the genre of the media', default='')
    parser.add_argument('-m', '--media', type=str, help='Type of media')
    parser.add_argument('-mod', '--modifier', type=str, help='Additional instructions for the list')

    args = parser.parse_args()

    get_essential_media(args.n, args.genre, args.media, args.modifier)