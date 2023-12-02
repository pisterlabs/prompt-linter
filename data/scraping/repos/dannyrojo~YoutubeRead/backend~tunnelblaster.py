import os
from urllib.parse import urlparse
import yt_dlp
import requests
import urllib.request
import shutil
import xml.etree.ElementTree as ET
import argparse
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from string import Template
import re

# Functions for extracting an array of urls from a playlist (or single video)
def check_if_playlist(input):
    parsed_url = urlparse(input)
    return parsed_url.path == "/playlist"

def truncate_url(url):
    result = re.match(r'^https://www\.youtube\.com/watch\?v=[\w-]+', url)
    if result:
        return result.group(0)
    else:
        print("Invalid URL format.")
        return None

def extract_urls_from_playlist(input):
    list_of_urls = []
    ydl_opts = {'quiet': True, 'extract_flat': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            playlist_info = ydl.extract_info(input, download=False)
        if 'entries' in playlist_info:
            entries = playlist_info['entries']
            for entry in entries:
                video_url = entry['url']
                list_of_urls.append(video_url)
    except yt_dlp.utils.DownloadError as e:
        print(f"TB error extracting playlist: {str(e)}")
    return list_of_urls

def process_host_url (input):  #function for API endpoint
    if check_if_playlist(input): 
        url_array = extract_urls_from_playlist(input) # If url is a playlist, create an array of urls
    else:
        url = input
        truncated_url = truncate_url(url)
        url_array = [truncated_url] # If it is not a playlist then wrap the url into an array
    print("This is the array of urls (url_array):", url_array)
    return url_array

# Functions for extracting a summary from an url
def extract_metadata(video_url): #yt-dlp
    print(f"Processing video URL: {video_url}")  # Debug print
    ydl_opts = {'quiet':True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            metadata = ydl.extract_info(video_url, download=False)
            return metadata
        except Exception as e:
            print(f"Error extracting metadata for the following: {video_url}: {e}")
            return None
               
def get_subtitle_url(metadata):
    #wit open('metadata.txt','w') as file:
        #file.write(str(metadata))

    lang = "en-US"
    language_codes_to_check = [lang, lang.split('-')[0]] 
    for code in language_codes_to_check:
        if 'automatic_captions' in metadata and code in metadata['automatic_captions']:
                for cap in metadata['automatic_captions'][code]:
                    if cap['ext'] == 'ttml':
                        return cap['url']
        if 'subtitles' in metadata and code in metadata['subtitles']:
            for sub in metadata['subtitles'][code]:
                if sub['ext'] == 'ttml':
                    return sub['url']
    else:
        print("No subtitles found")
    return None  
                
def get_plain_text_from_ttml(url): #parser
    if url:    
        response = requests.get(url)
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            texts = [elem.text.strip() for elem in root.iter() if elem.text]
            plain_text = " ".join(texts)    
            return plain_text
        else:
            print(f"Failed to retrieve captions content. Status codeD {response.status_code}")
    else:
        print("No valid URL for captions available.")
    return None

def initialize_prompts(config):
    print("THIS IS THE CONFIG FILE:", config)
    custom_map = config['currentConfig']['mapText']
    custom_combine = config['currentConfig']['reduceText']
    mapTemplate = Template(""""$custom_map: "{text}""""")
    map_prompt=mapTemplate.substitute(custom_map=custom_map)
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])    
    print("This is the map_prompt_template:", map_prompt_template)
    
    combineTemplate = Template(""""$custom_combine: "{text}""""")
    combine_prompt=combineTemplate.substitute(custom_combine=custom_combine)
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    print("This is the combine_prompt_template:", combine_prompt_template)
        
    return combine_prompt_template, map_prompt_template

def map_reduce_and_summarize(plain_text, map_prompt_template, combine_prompt_template): #langchain API
    if plain_text:
        
        #Configure langchain
        openaikey = os.environ.get('openaikey')
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openaikey)
        num_tokens = llm.get_num_tokens(plain_text)
        print (f"Our text has {num_tokens} tokens")

        #Split the text into chunks (overlap is too high)
        text_splitter = RecursiveCharacterTextSplitter(separators=[" "], chunk_size=10000, chunk_overlap=500)
        docs = text_splitter.create_documents([plain_text])

        #Define chain type, run, return
        summary_chain = load_summarize_chain(llm=llm,
                                            chain_type='map_reduce',
                                            map_prompt=map_prompt_template,
                                            combine_prompt=combine_prompt_template,
                                            verbose=False
                                            )
        summary = summary_chain.run(docs)
        print("SUMMARY COMPLETE")
        return summary
    else:
        print("No valid plain text content available for summarization.")
        summary = "!!!!!!!!!!!!!!!!ERROR!!!!!!!!!!!!!!!!!! \n\n No valid captions for this video, so no summary.  WOMP WOMP WOMP \n\n !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        return summary
    
def get_title_and_description(metadata): #yt-dlp
    video_title = metadata.get('title')
    video_description = metadata.get('description')
    upload_date = metadata.get('upload_date')
    duration_string = metadata.get('duration_string')
    uploader_id = metadata.get('uploader_id')
    return video_title, video_description, upload_date, duration_string, uploader_id

def process_video_url(input, config):  #function for API endpoint
    try:
        metadata = extract_metadata(input)
        subtitle_url = get_subtitle_url(metadata)
        plain_text = get_plain_text_from_ttml(subtitle_url)
        map_prompt_template, combine_prompt_template = initialize_prompts(config)
        summary = map_reduce_and_summarize(plain_text, map_prompt_template, combine_prompt_template)
        video_title, video_description, upload_date, duration_string, uploader_id = get_title_and_description(metadata)
        summary = {
            'title': video_title,
            'uploader_id': uploader_id,
            'url': input,
            'upload_date': upload_date,
            'duration': duration_string,
            'summary': summary,
            'description': video_description  
        }
        return summary
    except Exception as e:
        print(f"Error processing video URL {input}: {str(e)}")

def save_info(url, summary, title, description, upload_date, duration):    
        # Weird characters be gone (causing issues with "oepn" method)!
        sanitized_video_title = title.replace('/','_').replace('\\','_')        
        # Write it down
        with open(f'{sanitized_video_title}' + '_info.md', 'w') as md_file:
            md_file.write(f"Video Title: {title}\n")
            md_file.write(f"URL: {url}\n")
            md_file.write(f"Duration: {duration}\n")
            md_file.write(f"Upload Date: {upload_date}\n\n")
            md_file.write(f"Summary: {summary}\n\n")
            md_file.write(f"Video Description: {description}\n\n")

if __name__ == '__main__':  
   
   # Declare and define argument parser for script
    parser = argparse.ArgumentParser(description = "URL")
    parser.add_argument("url", help="URL of Playlist")
    args = parser.parse_args()
    url = args.url
    
    # Broke the script, fix the argument parser later
    
    
