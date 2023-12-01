from pytube import YouTube
from langchain.tools import YouTubeSearchTool
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.tools import Tool

def video_get(video_url):
    yt = YouTube(video_url,use_oauth=False, allow_oauth_cache=True)
    print(f"youtube to be downloaded - {yt}")
    stream = yt.streams.filter(progressive=True, file_extension='mp4').get_lowest_resolution()
    print(stream)
    vpath = stream.download()
    print(f"Downloaded video {vpath}")
    return vpath

def youtube_search(query: str):
    query_string = query#+",1"
    url_list = YouTubeSearchTool().run(query_string).strip('][').split(', ')
    video_id = url_list[0].strip("'").split("?v=")[1]
    # print(video_id)
    try:
        video_transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        video_transcript = ','.join(map(str, video_transcript_list))
    except:
        video_transcript = "Could not find transcript for this video."

    # print(video_transcript)
    video_url = 'https://www.youtube.com' + url_list[0].strip("'")

    result = {
        "tool": "youtube",
        "tldr" : "Found a related video.",
        "article" : video_url+"\n\n"+video_transcript
    }
    try:
        video_path = video_get(video_url)
        result["video"] = video_path
    except:
        print('failed to download video')

    return result
    #return "The youtube video on "+query+" is available at URL "+result_url+" ."+"The transcript is as follows: "+video_transcript

# define youtube search tool
youtube_tool = Tool.from_function(
        func=youtube_search,
        name="YouTube",
        description="Use this tool only to search for videos, songs and youtube. Prefer this over normal search when searching for videos.",
        return_direct=True
        #args_schema=CalculatorInput
        # coroutine= ... <- you can specify an async method if desired as well
    )