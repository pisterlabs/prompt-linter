import re
from pytube import Playlist
from youtube_transcript_api import YouTubeTranscriptApi
import cohere
co = cohere.Client('rGjz0KNIMSReCgEyzpEUDQpYzxSoXb85RjjdyAel')
from pytube import YouTube
import json

# URL of the YouTube playlist
# URL_PLAYLIST = "https://www.youtube.com/playlist?list=PLEoM_i-3sen_w5IYh0d5xtnpLHJeeO8l5"

def YoutubeParse(url):
    # Retrieve URLs of videos from playlist
    playlist = Playlist(url)
    print('Number Of Videos In playlist: %s' % len(playlist.video_urls))

    urls = []
    for url in playlist:
        urls.append(url)

    # print(urls)


    pattern = r'(?<=v=)[\w-]+'

    video_idx = {}
    uid = 0


    documents = []
    for youtube_url in urls:
        match = re.search(pattern, youtube_url)

        if match:
            video_id = match.group()
        else:
            print("Video ID not found.")

        # retrieve the available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # using the srt variable with the list of dictionaries
        # obtained by the .get_transcript() function
        srt = YouTubeTranscriptApi.get_transcript(video_id)

        transcript = '\n'.join(i["text"] for i in srt)
        response = co.summarize(
        text=transcript,
        )
        video_idx[uid] = (youtube_url, transcript, response.summary)
        uid += 1

        document = {
            'title': str(uid),
            'snippet': response.summary,
        }
        documents.append(document)

    # print(video_idx)

    # Convert documents list to JSON format
    json_data = json.dumps(documents, indent=2)  # Convert list of dictionaries to JSON string

    # Print the JSON-formatted data
    print(json_data)

    # Write JSON data to a file
    output_file = 'documents_auto.json'  
    with open(output_file, 'w') as file:
        file.write(json_data)

YoutubeParse("https://www.youtube.com/playlist?list=PLEoM_i-3sen_w5IYh0d5xtnpLHJeeO8l5")