import base64
from datetime import datetime
import eyed3
from eyed3.id3 import ID3_DEFAULT_VERSION
from eyed3.id3.frames import ImageFrame
from internetarchive import get_session
from internetarchive import upload
import feedparser
import json
from dotenv import load_dotenv
import math
import openai
import os
import requests
from zoneinfo import ZoneInfo # Requires Python 3.9 or later.

# Helper function to format dates in a human-friendly way.
def ordinal(date_number):
    suffix = ["th", "st", "nd", "rd", "th"][min(date_number % 10, 4)]
    if 11 <= (date_number % 100) <= 13:
        suffix = "th"
    return str(date_number) + suffix


def fetch_stories(news_feed_url, limit=10):
    try:
        feed = feedparser.parse(news_feed_url)
        if feed.bozo:
            raise Exception(feed.bozo_exception)
    except Exception as e:
        print(f"Error fetching RSS feed: {e}")
        exit(1)

    stories = "".join(
        f" New Story: {item.title}. {item.description}"
        for item in feed.entries[:limit] if hasattr(item, 'description')
    )

    return stories



def generate_chat_content(stories, today):
    prompt = "Here are some news headlines and summaries for today (" + today + "). First, combine related stories into a single news story. Don't repeat similar news stories. Second, ignore any very short-term news stories such as a train delay. Third, rewrite the news stories in a discussion way, as though someone is talking about them one by one on the 'Just a Moment News' podcast in a non-judgemental way and with no follow-on discussion. Fourth, DO NOT append the term [Image] or [Source] or [Image Source]. Fifth, add an opening greeting (mentioning your name 'Erika', 'Just a Moment News' and the date) and a closing greeting (mentioning your name 'Erika' and 'Just a Moment News')."
    
    try:
        chat_output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": prompt + " NEWS STORIES: " + stories
            }],
            temperature=0.5
        )
        chat_content = chat_output.choices[0].message.content
    except openai.error.AuthenticationError as e:
        print(f"OpenAI API authentication error: {e}")
        exit(1)
        
    return chat_content


def generate_audio(chat_content, voice_id, elevenlabs_api_key, today):
    try:
        audio_output = requests.post(
            "https://api.elevenlabs.io/v1/text-to-speech/" + voice_id,
            data=json.dumps({
                "text": chat_content,
                "voice_settings": {
                    "stability": 0.3,
                    "similarity_boost": 1
                }
            }),
            headers={
                "Content-Type": "application/json",
                "xi-api-key": elevenlabs_api_key,
                "accept": "audio/mpeg"
            }
        )

        if audio_output.status_code == 200:
            output_file_name = f"momentnews_{today}.mp3"
            with open(output_file_name, "wb") as output_file:
                output_file.write(audio_output.content)
        elif audio_output.status_code == 401:
            print("ElevenLabs API authentication error: Unauthorized")
            print(audio_output.text)
            exit(1)
        else:
            print(audio_output.text)

    except requests.exceptions.RequestException as e:
        print(f"ElevenLabs API request error: {e}")
        exit(1)


def add_id3_tags(file_path, title, artist, album, year, genre, comment):
    audio_file = eyed3.load(file_path)

    if audio_file.tag is None:
        audio_file.initTag()

    audio_file.tag.title = title
    audio_file.tag.artist = artist
    audio_file.tag.album = album
    audio_file.tag.year = year
    audio_file.tag.genre = genre
    audio_file.tag.comments.set(comment)

    audio_file.tag.save()


def add_cover_art(file_path, image_path, image_type=ImageFrame.FRONT_COVER, mime_type="image/png", description="Cover Art"):
    audio_file = eyed3.load(file_path)

    if audio_file.tag is None:
        audio_file.initTag(version=ID3_DEFAULT_VERSION)

    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    img_frame = ImageFrame()
    img_frame.picture_type = image_type
    img_frame.mime_type = mime_type  # Set mime_type explicitly
    img_frame.description = description
    img_frame.image_data = image_data
    audio_file.tag.frame_set[b'APIC'] = img_frame

    audio_file.tag.save()


def create_text_files(chat_content, today, output_file_name):
    try:
        # Create transcript file
        transcript_file_name = f"momentnews_{today}.txt"
        with open(transcript_file_name, "w", encoding="utf-8") as transcript_file:
            transcript_file.write(chat_content)
    except IOError as e:
        print(f"Error creating transcript file: {e}")
        exit(1)

    try:
        # Create markdown file for Jekyll website
        markdown_file_name = f"{today}-news.md"
        with open("markdown_template.md", "r", encoding="utf-8") as template_file:
            template_content = template_file.read()

        # Get the duration of the audio file
        audio_file = eyed3.load(output_file_name)
        duration = audio_file.info.time_secs
        duration_rounded = math.floor(duration)
        
        # Convert the duration to the "MM:SS" format
        minutes = duration_rounded // 60
        seconds = duration_rounded % 60
        length = f"{minutes:02d}:{seconds:02d}"
        
        # Get the current date & time
        ct_timezone = ZoneInfo("Africa/Johannesburg")
        current_datetime_ct = datetime.now(ct_timezone)
        formatted_datetime = current_datetime_ct.strftime("%Y-%m-%d %H:%M:%S %z")
        
        # Format the date in a human-friendly format, e.g. April 5th, 2023
        today_human = current_datetime_ct.strftime("%B ") + ordinal(current_datetime_ct.day) + current_datetime_ct.strftime(", %Y")        
        
        formatted_content = template_content.format(today=today, today_human=today_human, datetime=formatted_datetime, chat_content=chat_content, duration=duration_rounded, length=length)
        
        with open(markdown_file_name, "w", encoding="utf-8") as md_file:
            md_file.write(formatted_content)
    except IOError as e:
        print(f"Error creating markdown file: {e}")
        exit(1)


def upload_to_internet_archive(ia_session, file_path, identifier, metadata):
    try:
        item = ia_session.get_item(identifier)
        response = item.upload(file_path, metadata=metadata, retries=3)
        print(f"File {file_path} uploaded to the Internet Archive with identifier {identifier}")
        return response
    except Exception as e:
        print(f"Error uploading to the Internet Archive: {e}")
        exit(1)


def read_config_from_env():
    load_dotenv()
    config = {
        'news_feed': "https://rss.app/feeds/1KgoD3402e6Lv4Ge.xml",  
        'openai_api_key': os.getenv("OPENAI_API_KEY"),
        'elevenlabs_api_key': os.getenv("ELEVENLABS_API_KEY"),
        'ia_access_key': os.getenv("IA_ACCESS_KEY"),
        'ia_secret_key': os.getenv("IA_SECRET_KEY"),
        'github_token': os.getenv("GITHUB_TOKEN"),
        'github_repo': "shauntrennery/moment"  
    }
    return config


def process_rss_feed(news_feed_url):
    print("## Processing RSS feed")
    stories = fetch_stories(news_feed_url)
    return stories


def process_chat_gpt(openai_api_key, stories, today):
    print("## Processing ChatGPT")
    openai.api_key = openai_api_key
    chat_content = generate_chat_content(stories, today)
    print(chat_content)
    return chat_content


def process_audio_and_metadata(chat_content, elevenlabs_api_key, today, voice_id="6w3m9C9QzMZKcPVeYzYr"):
    print("## Processing audio and metadata")
    output_file_name = f"momentnews_{today}.mp3"
    generate_audio(chat_content, voice_id, elevenlabs_api_key, today)

    # Add ID3 tags to the MP3 file
    title = f"Just a Moment News - {today}"
    artist = "Just a Moment News"
    album = "Just a Moment News Podcast"
    year = int(today.split("-")[0])  # Extract the year from the date string
    genre = "Podcast"
    comment = "Generated using GPT-3.5-turbo and ElevenLabs TTS API."
    add_id3_tags(output_file_name, title, artist, album, year, genre, comment)
    
    # Add cover art to the MP3 file
    cover_image_path = "momentnews_cover-image.png"
    add_cover_art(output_file_name, cover_image_path)
    
    return output_file_name


def upload_to_archive(config, output_file_name, today):
    print("## Uploading to the Internet Archive")
    ia_session = get_session(config={
        's3': {
            'access': config['ia_access_key'],
            'secret': config['ia_secret_key']
        }
    })
    identifier = f"momentnews"
    metadata = {
        "title": f"Just a Moment News - {today}",
        "creator": "Just a Moment News",
        "date": today,
        "language": "eng",
        "licenseurl": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "mediatype": "audio",
        "collection": "opensource_audio",  # Change this to your desired collection
        "subject": "podcast; daily; news"
    }
    upload_to_internet_archive(ia_session, output_file_name, identifier, metadata)

def upload_to_github(github_token, repo_name, file_path, commit_message):
    print("## Uploading markdown file to GitHub")
    try:
        # Read the file
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Prepare the API endpoint and headers
        api_base = "https://api.github.com"
        endpoint = f"{api_base}/repos/{repo_name}/contents/docs/_posts/{file_path}"
        headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github+json",
        }

        # Check if the file exists in the repo
        response = requests.get(endpoint, headers=headers)
        if response.status_code == 200:
            file_sha = response.json()["sha"]
        else:
            file_sha = None

        # Prepare the API request data
        data = {
            "message": commit_message,
            "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
            "branch": "main",  # Change this to the target branch if necessary
        }
        if file_sha:
            data["sha"] = file_sha

        # Upload the file
        response = requests.put(endpoint, json=data, headers=headers)
        if response.status_code in [201, 200]:
            print(f"File {file_path} uploaded to GitHub repository {repo_name}")
        else:
            print(f"Error uploading to GitHub: {response.status_code} {response.text}")
            exit(1)

    except Exception as e:
        print(f"Error uploading to GitHub: {e}")
        exit(1)


def main():
    config = read_config_from_env()
    today = datetime.now().strftime("%Y-%m-%d")

    stories = process_rss_feed(config['news_feed'])
    chat_content = process_chat_gpt(config['openai_api_key'], stories, today)
    output_file_name = process_audio_and_metadata(chat_content, config['elevenlabs_api_key'], today)
    create_text_files(chat_content, today, output_file_name)
    upload_to_archive(config, output_file_name, today)
    upload_to_github(config['github_token'], config['github_repo'], f"{today}-news.md", f"Add {today} Just a Moment News")
    

    print("## Processing complete")


if __name__ == "__main__":
    main()
