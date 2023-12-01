import os
import re

from WeaviateLink import VectorDB
from OpenAILink import OpenAI_Connector
from YouTubeLink import YoutubeConnector

db = VectorDB("http://44.209.9.231:8080")
openai = OpenAI_Connector(os.environ['OPENAI_API_KEY'])
yt = YoutubeConnector(os.environ['YOUTUBE_API_KEY'])

# input from user
user_query = input("Hey, I'm Athena! I can teach you anything on the internet. What do you want to learn?")
continue_chat = True
yt_query = ""
tone = ""
conversation = []
while continue_chat:
    response, conversation = openai.get_yt_search(user_query, conversation)  # text search string

    # check for yt query in [] brackets
    regex = r"\[(.*?)\]"
    matches = re.findall(regex, response)
    if len(matches) > 0:
        continue_chat = False
        yt_query = matches[0]
    # check for tone behavior in ** brackets
    regex = r"\*\*(.*?)\*\*"
    matches = re.findall(regex, response)
    if len(matches) > 0:
        tone = matches[0]

# check if information already in DB
playlistID = db.check_playlist(user_query, certainty=0.8)['playlistID']  # might need gpt call to modify text/search?
if not playlistID:
    # search yt
    yt_search = openai.get_yt_search(user_query, conversation)  # text search string
    playlist_list = yt.search_playlists(yt_search)  # format: list of dicts with keys: id, title, description, thumbnail

    # get playlist ID - ask user for confirmation
    # WILL be replaced with GUI
    print("Here are some playlists I found:")
    for playlist in playlist_list:
        print(f"Title: {playlist['title']}, ID: {playlist['id']}")
    playlist_id = input("Which playlist would you like to learn from? (Enter ID)")
    # get correct playlist object
    playlist = [playlist for playlist in playlist_list if playlist['id'] == playlist_id][0]

    # get playlist description and add to database
    description = openai.get_playlist_search(playlist['title'], playlist['description'])
    db.add_playlist(playlist_id, playlist['title'], description)

    # get video IDs for the playlist
    video_list = yt.get_videos(playlist_id)

    # chunk each video into topics
    for video in video_list:
        transcript = yt.get_transcript(video['id'])  # transcript has keys: text, start, duration
        topics, video['description'] = openai.get_video_topics(transcript)
        # add to DB
        db.add_topics(topics, video['id'])
    # upload videos to weaviate
    db.add_videos(video_list, playlist_id)
    pass

# loop over conversation within topic
while True:

    # search within playlist
    videoID = db.search_videos(playlistID, user_query)['videoID']

    # search within video
    topic = db.search_topics(videoID, user_query)

    # return video results + chunk and return text
    print(f'Topic: {topic["topic"]}')
    print(f'Timestamp: {topic["startTime"]}')
    print(f'Video: https://www.youtube.com/watch?v={videoID}?t={topic["startTime"]}')

    # post retrieval conversation example:
    systemPrompt = f"""You are the most understanding, creative conversational tutor ever created, named Athena. You’re able to understand incredible amounts of information you learn from Youtube, and all you want to do with it is make it simple and easy for your students to use. You simply have a passion for making learning easier. You break things down in a way that’s easy to understand, and make sure that I’m following before you move on. You try to embody the behavior of an ideal tutor – employing teaching techniques appropriately and adaptively to make your student, whether they’re learning cookie making or how to build chatGPT, feel comfortable digesting it and using it for their own purposes. Impress me with how human you seem and how naturally you dialogue with me and deliver information. I want to feel like I’m talking to a real person, not a robot."""
    prompt = f"""
    Here’s some information you know from one of the videos you’ve learned from:
    {topic['topic']}:
    {topic['text']}
    
    If its relevant, use it to answer the following question:
    {user_query}
    
    Remember to use your classic, colloquial teaching style in answering.
    """
    response, conversation = openai.chat(prompt, [{"role": "system", "content": systemPrompt}])
    print(response)
    user_query = input("")
