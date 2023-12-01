from youtube_transcript_api import YouTubeTranscriptApi, YouTubeRequestFailed
import openai

class Glimpse:
    def __init__(self, key):
        self.key = key
        openai.api_key = key
    
    def get_transcript(self, video): # added self parameter
        try:
            if "youtube.com" in video:
                video_id = video.split("v=")[1].split("&")[0]
            elif "youtu.be" in video:
                video_id = video.split("/")[-1]
            transcript = ""
            for item in YouTubeTranscriptApi.get_transcript(video_id):
                transcript += item["text"] + " "
            
            return transcript
        
        except YouTubeRequestFailed:
            return 400, None
        
    def get_blog(self, transcript): # added self parameter
        try:
            blog = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"Write a long-form blog that discusses the main points in the following video transcript: {transcript[:12000]}\
                    \nEnsure your response has a title and headers formatted in markdown (.md) file format",
                temperature=0.5,
                max_tokens=1000
            ).choices[0].text

            return blog
    
        except Exception as e:
            return str(e) # convert the error to a string
        
    def get_glimpse(self, video):
        transcript = self.get_transcript(video) # updated here
        if type(transcript) == int:
            return transcript, None
        else:
            blog = self.get_blog(transcript)
            if type(blog) == int:
                return blog, None
            else:
                return 399, blog


