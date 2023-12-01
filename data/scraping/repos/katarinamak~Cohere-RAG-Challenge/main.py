
import re
from pytube import Playlist
from youtube_transcript_api import YouTubeTranscriptApi
import cohere
co = cohere.Client('rGjz0KNIMSReCgEyzpEUDQpYzxSoXb85RjjdyAel')
from pytube import YouTube
import json
import webbrowser


transcriptSelected = 0
globalvideomap= {}
documents = []
urls = []
transcripts = []
usermsg = ""

def storeURLS(url):
    playlist = Playlist(url)
    for url in playlist:
        urls.append(url)

def Transcript(url):
    # Retrieve URLs of videos from playlist
    playlist = Playlist(url)
    print('Number Of Videos In playlist: %s' % len(playlist.video_urls))

    # print(urls)


    pattern = r'(?<=v=)[\w-]+'

    video_idx = {}
    uid = 0


    
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
        transcripts.append(transcript)

def YoutubeParse(url):
    # Retrieve URLs of videos from playlist
    playlist = Playlist(url)
    print('Number Of Videos In playlist: %s' % len(playlist.video_urls))

    # print(urls)


    pattern = r'(?<=v=)[\w-]+'

    video_idx = {}
    uid = 0


    
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
        globalvideomap[uid] = youtube_url
        document = {
            'title: ': uid,
            'snippet': response.summary,
        }
        uid += 1

        documents.append(document)

    # print(video_idx)

    # Convert documents list to JSON format
    json_data = json.dumps(documents, indent=2)  # Convert list of dictionaries to JSON string

    # Print the JSON-formatted data
    print(json_data)

    # Write JSON data to a file
    output_file = 'documents.json'  
    with open(output_file, 'w') as file:
        file.write(json_data)



def process(response):
    citation_docs = response.citations
    # print(citation_docs)
    indexes_array = [entry['document_ids'][0] for entry in citation_docs]
    print("INDEX OF VIDEO")
    numbers_after_underscore = [int(entry.split('_')[1].split(':')[0]) if '_' in entry else None for entry in indexes_array]
    print(numbers_after_underscore[0])
    transcriptSelected = numbers_after_underscore[0]-1
    return urls[numbers_after_underscore[0] - 1]

def getdocindex(response):
    citation_docs = response.citations
    # print(citation_docs)
    indexes_array = [entry['document_ids'][0] for entry in citation_docs]
    print("INDEX OF VIDEO")
    numbers_after_underscore = [int(entry.split('_')[1].split(':')[0]) if '_' in entry else None for entry in indexes_array]
    print(numbers_after_underscore[0]-1)

    

def openVid():
    msg = input('What are you looking to learn? ')
    usermsg = msg
    co = cohere.Client('rGjz0KNIMSReCgEyzpEUDQpYzxSoXb85RjjdyAel')
    with open('./documents_auto.json', 'r') as file:
        documents = json.load(file)
    response = co.chat(
    message= msg,
    documents=documents,
        prompt_truncation= "AUTO"
    )
    
    print(response.text)

    # print(citation_docs)
    # indexes_array = [entry['document_ids'][0] for entry in citation_docs]
    # val_array = [int(entry[-1]) for entry in indexes_array]
    # print(val_array)

    vidurl = process(response)
    return vidurl

# def getLengthVideo(url):
#     # Create a YouTube object
#     yt = YouTube(url)

#     # Get the video length in seconds
#     video_length_seconds = yt.length

#     return video_length_seconds


def finalOpenTime(chunk, totalChunks, videoLength, url):
    webbrowser.open(url)


def search_doc(document, msg):
    def split_text(text, chunk_size):
        # Split the text into chunks of specified size
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        return chunks

    # Set the desired chunk size
    chunk_size = 2000  # You can adjust this to your preferred chunk size

    # Split the long text into chunks
    text_chunks = split_text(document, chunk_size)

    chunks = []

    # Print the chunks
    for i, chunk in enumerate(text_chunks, start=1):
        doc = {
            'title': i,
            'snippet': chunk,
        }
        chunks.append(doc)

    co = cohere.Client('rGjz0KNIMSReCgEyzpEUDQpYzxSoXb85RjjdyAel')
    
    response = co.chat(
    message= usermsg,
    documents=documents,
        prompt_truncation= "AUTO"
    )

    print(response.text)
    index = getdocindex(response)
    percentLocation = index / len(chunks)

    print(percentLocation)
    return percentLocation


    
   


url = input("Enter Youtube playlist URL: ")

# YoutubeParse(url)
storeURLS(url)
# Transcript(url)

print("Output added to documents.json")

print("Generating Response...")

vidurl = openVid()


# print(getLengthVideo(url))

finalOpenTime(0, 0, 0, vidurl)

# search_doc(transcripts[transcriptSelected], usermsg)





