from googleapiclient.discovery import build
from .utils import utils
from .db import mongo
import openai       # Get Dictation

import os
from dotenv import load_dotenv
load_dotenv()

# Set Up the API Client:
YOUTUBE = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY")) #, credentials=credentials) 
openai.api_key = os.getenv("OPENAI_API_KEY")

# Store the audio files in "Youtube" subfolder
AUDIO_DIR = "Youtube"       

def search_and_get(query, view_threshold = 10000):
    search_results = search_videos(query, "relevance")
    for item in search_results:
        video_id = item['id']['videoId']
        stat = get_stat(video_id)
        view_count = int(stat["view_count"])
        # Check if view count exceeds the threshold
        if view_count > view_threshold:
            get_dictation_and_comments(video_id)

def get_dictation_and_comments(video_id):
    try: 
        video_url = utils.construct_youtube_url(video_id)
        print(f"Getting dictation for {video_id}.")
        store_dictation(video_url)
        print(f"Getting comments for {video_id}.")
        store_comments(video_id)
    except Exception as e:
        print(f"Can't get {video_id} because:", e)


#################################################################################
#                       Search
#################################################################################
def search_videos(query, order, max_results = 50):
    search_response = YOUTUBE.search().list(
        q=query,
        type="video",
        order=order,
        part="id,snippet",
        maxResults=max_results
    ).execute()

    return search_response['items']

#################################################################################
#                       Statistics
#################################################################################
def get_stat(video_id):
    stat_request = YOUTUBE.videos().list(
        part="statistics",
        id=video_id
    )
    stat_response = stat_request.execute()
    stat = stat_response['items'][0]['statistics']

    video_stat = {
        'view_count': stat['viewCount'],
        'like_count': stat['likeCount'],
        'favorite_count': stat['favoriteCount'],
        'comment_count': stat.get('commentCount', '0')  #if the the commentCount doesn't exist, set it to zero
    }
    return video_stat

#################################################################################
#                       Dictation
#################################################################################
def store_dictation(video_url):
    video_id = utils.extract_id_from_url(video_url)

    if mongo.youtube_get_dictation_by_video_id(video_id):
        print(f"dictation for {video_id} already exists in database.")
        return 

    # Attempt to get captions
    # caption_id = find_captions(video_id)
    # if caption_id:
    #     document = {"_id": video_id, "text": get_caption(video_url)}
    #     mongo.youtube_insert_dictation(document)
    # else:

    # If no captions, download audio
    audio_filepath = download_audio(video_url)
    # Get dictation from audio
    dictation = audio2text(audio_filepath)
    if dictation:
        document = {"_id": video_id, "text": dictation}
        stat = get_stat(video_id)
        document.update(stat)
        mongo.youtube_insert_dictation(document)
    else:
        raise Exception("Can't get diactation")

#################################################################################
#                       Caption
#################################################################################
def find_captions(video_id):
    caption_list = YOUTUBE.captions().list(
        part="snippet",
        videoId=video_id
    ).execute()
    
    english_caption_id = None
    any_caption_id = None
    
    for caption in caption_list['items']:
        if 'en' in caption['snippet']['language']:
            english_caption_id = caption['id']
            break
        if any_caption_id is None:
            any_caption_id = caption['id']

    return english_caption_id or any_caption_id

def get_caption(video_url):
    try:
        any_cap = get_any_caption(video_url)
        # if any_cap is None:
        #     raise Exception(f"No Any Caption for {video_url}")
    except Exception:
        raise Exception(f"Should have caption for {video_url}")

# import pysubs2

# def extract_text(subtitle_file_path, output_file_path):
#     subs = pysubs2.load(subtitle_file_path)
#     with open(output_file_path, 'w', encoding='utf-8') as file:
#         for line in subs:
#             file.write(f"{line.text}\n")


def get_any_caption(video_url):
    video_id = utils.extract_id_from_url(video_url)

    ydl_opts = {
        'skip_download': True,      # We just want the subtitle
        'writesubtitles': True,
        'subtitleslangs': ['all'],  # All
        'subtitlesformat': 'vvt',
        'outtmpl': f"{AUDIO_DIR}/{video_id}.%(ext)s",
    }

    import yt_dlp
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

        file_path = f"{AUDIO_DIR}/{video_id}"
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # return content

# def get_any_caption(video_url):
#     from pytube import YouTube
#     yt = YouTube(
#         video_url,
#         use_oauth=True, allow_oauth_cache=True
#     )

#     try:
#         # Get an iterator over the dictionary's items
#         iter_caption = iter(yt.captions.values())
#         # Get the first item
#         any_caption = next(iter_caption)
#         return any_caption.generate_srt_captions()
#     except StopIteration:
#         print(f"There is no caption for {video_url}.")
#         return None

#################################################################################
#                       Audio
# download, get dictation of audio
#################################################################################
def download_audio(video_url):
    video_id = utils.extract_id_from_url(video_url)

    from pytube import YouTube
    yt = YouTube(
        video_url,
        use_oauth=True, allow_oauth_cache=True
    )
    audio_filename = f"{video_id}.mp3"
    audio_filepath = os.path.join(AUDIO_DIR, audio_filename)

    # Check if the audio file already exists
    if os.path.exists(audio_filepath):
        print(f"Audio for {video_id} already exists at {audio_filepath}")
        return audio_filepath

    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(output_path=AUDIO_DIR, filename=audio_filename)
    print(f"Audio downloaded for {video_id}")

    return audio_filepath

def audio2text(audio_filepath):
    try:
        with open(audio_filepath, "rb") as audio_file:
            transcript = openai.Audio.translate("whisper-1", audio_file)
        return transcript
    except Exception as e:
        print(f"Failed to transcribe audio: {e}")
        return None

#################################################################################
#   Comments
#################################################################################
def store_comments(video_id):
    request = YOUTUBE.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        order="relevance",
        maxResults=100
    )
    try: 
        response = request.execute()
        for item in response["items"]:
            comment = process_comment_item(item)
            if mongo.youtube_exist_comment(comment["_id"]):
                print(f'Comment {comment["_id"]} of video {comment["video_id"]} already exists')
            else:
                mongo.youtube_insert_comments(comment)
    except Exception as e:
        print(f'can not get comment of {video_id} because:', e)

def process_comment_item(item):
    snippet = item['snippet']
    top_level_comment = snippet['topLevelComment']['snippet']
    author_channel = top_level_comment['authorChannelId']

    # Prepare a dictionary with the relevant data
    comment_data = {
        '_id': item['id'],
        'video_id': snippet['videoId'],
        'channel_id': snippet['channelId'],
        'comment_text': top_level_comment['textOriginal'],
        'author_display_name': top_level_comment['authorDisplayName'],
        'author_profile_image_url': top_level_comment['authorProfileImageUrl'],
        'author_channel_url': top_level_comment['authorChannelUrl'],
        'author_channel_id': author_channel['value'] if author_channel else None,
        'like_count': top_level_comment['likeCount'],
        'published_at': top_level_comment['publishedAt'],
        'updated_at': top_level_comment['updatedAt'],
        'can_reply': snippet['canReply'],
        'total_reply_count': snippet['totalReplyCount'],
        'is_public': snippet['isPublic']
    }

    return comment_data

# This will keep the structure of youtube comment
# def get_comments(video_id):
#     # Get the top-level comments first
#     top_level_request = youtube.commentThreads().list(
#         part="snippet",
#         videoId=video_id,
#         textFormat="plainText"
#     )
#     top_level_response = top_level_request.execute()

#     comments_data = []

#     for item in top_level_response["items"]:
#         top_level_comment = process_comment_item(item)
#         top_level_comment['parent_id'] = None  # For top-level comments, parent_id is None
#         comments_data.append(top_level_comment)

#         # Check for replies to this top-level comment
#         if item['snippet']['totalReplyCount'] > 0:
#             replies_request = youtube.comments().list(
#                 part="snippet",
#                 parentId=item['id'],
#                 textFormat="plainText"
#             )
#             replies_response = replies_request.execute()

#             for reply_item in replies_response['items']:
#                 reply_comment = process_comment_item(reply_item)
#                 reply_comment['parent_id'] = item['id']  # Set the parent_id to the top-level comment ID
#                 comments_data.append(reply_comment)

#     return comments_data
