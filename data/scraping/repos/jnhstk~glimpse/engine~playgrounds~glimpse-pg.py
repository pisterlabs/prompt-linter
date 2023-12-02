'''
Returns a blog for a youtube video
Return Codes:


'''
import openai
from youtube_transcript_api import YouTubeTranscriptApi, YouTubeRequestFailed


def glimpse(video):
    '''
    Returns a glimpse of a youtube video

    Args:
        (str) video: either a YT video link or id

    Returns:
        
    '''

    def transcript_request(video):
        '''
        Retrieves/returns the transcript for a prompted video

        Args:
            (str) video: video link or id

        Returns: signal[]
            if error = [0] = (int) error_code

            else = [(int) 398, (str) transcript]

        Return Codes:
            398 - Success
            400 - Transcript not found
        '''
        if "youtube.com" in video:
            video_id = video.split("v=")[1].split("&")[0]
        elif "youtu.be" in video:
            video_id = video.split("/")[-1]
        else:
            video_id = video
        try: 
            transcript = ""
            for item in YouTubeTranscriptApi.get_transcript(video_id):
                transcript += item["text"] + " "

            if len(transcript) > 12000:
                transcript = transcript[:12000]

            return [398, transcript] # SUCCESS

        except Exception:
            return [400, video_id] #TRANSCRIPT NOT FOUND 


    def blog_request(transcript):
        try:
            openai.api_key = "key"
            blog = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"Write a long-form blog that discusses the main points in the following video transcript: {transcript}\
                    \nEnsure your response has a title and section headers formatted in markdown (.md) file format",
                temperature=0.5,
                max_tokens=1000
            ).choices[0].text
            return [399, blog]

        except Exception as e:
            return [401, e]

    transcript = transcript_request(video)
    if transcript[0] == 398:
        blog = blog_request(transcript[1])
        if blog[0] == 399:
            return [399, blog[1]]
        else:
            return [401]
    else:
        return [400]

glimpse = glimpse("https://www.youtube.com/watch?v=G6uwkc11NZ8")
if glimpse[0] == 399:
    with open("blog.md", "w") as f:
        f.write(glimpse[1])
else:
    print(glimpse[0])

