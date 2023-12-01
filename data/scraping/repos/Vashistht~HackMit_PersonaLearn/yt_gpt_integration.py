# imports gpt
import os
import openai
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv

#general
load_dotenv()

class YtRec:
    def __init__(self, transcript):
        self.trascript = transcript
        self.ytapi = os.getenv('YOUTUBE_API_KEY')
        openai.api_key = os.getenv('OPENAI_API_KEY')
        print(self.ytapi, openai.api_key )
    
    def gpt_keywords(self):
        words = self.trascript
        query = "Extract keywords from this text: "
        prompt_ = query + words
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt_,
            temperature=0.3,
            max_tokens=400,
            top_p=.98,
            frequency_penalty=0.7,
            presence_penalty=0.0
        )
        x = response
        text = x['choices'][0]['text']
        text_new = text.replace('\n-', ',')
        return text_new

    def gpt_topic(self):
        words = self.trascript
        query_new = "Find the topic:"
        prompt_ = query_new + words

        response_new = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt_,
            temperature=0.3,
            max_tokens=400,
            top_p=.98,
            frequency_penalty=0.7,
            presence_penalty=0.0
        )
        x_new = response_new
        text = x_new['choices'][0]['text']
        return text
    
    
    def youtube_get_videos(self):
        #Initializing the API
        api_key = self.ytapi
        youtube = build( 'youtube', 'V3', developerKey= api_key)
        
        text = self.gpt_topic()
        #Searching for query with decreasing order as viewcount and result as video type
        results = youtube. search().list(q=text, part="snippet", type="video", 
                                         order="viewCount").execute( )
        
        return results
    
    # this next few fns can def be better
    def get_title(self, item):
        title = item['snippet']['title']
        return title

    def get_thumbnail(self, item):
        video_id = item['id']['videoId']
        video_url = "https://img.youtube.com/vi/" + video_id + "/default.jpg"
        return video_url

    def get_video_url(self, item):
        video_id = item['id']['videoId']
        video_thumbnail = "https://www.youtube.com/watch?v=" + video_id
        return video_thumbnail

    def get_viewcount(self, item, youtube):
        video_id = item['id']['videoId']
        video_statistics = youtube.videos().list(id=video_id,
                                            part='statistics').execute()
        viewcount = int(video_statistics['items'][0]['statistics']['viewCount'])
        return viewcount

    def get_likecount(self, item, youtube):
        video_id = item['id']['videoId']
        video_statistics = youtube.videos().list(id=video_id,
                                            part='statistics').execute()
        likecount = int(video_statistics['items'][0]['statistics']['likeCount'])
        return likecount
    
    # refine this based on the feedback from the team 
    def custom_score(self, likecount, viewcount) :
        score = ( (likecount **2 + viewcount) / (viewcount))
        return score

    def new_order(self, results, df, youtube):
        i = 1
        topics, keywords = self.gpt_topic(), self.gpt_keywords()
        for item in results['items']:
            title = self.get_title(item)
            thumbnail = self.get_thumbnail(item)
            viewcount = self.get_viewcount(item, youtube)
            video_url = self.get_video_url(item)
            likecount = self.get_likecount(item, youtube)
            # dislikecount = get_dislikecount(item, youtube)
            score = self.custom_score(likecount, viewcount)
            df.loc[i] = [title, thumbnail,video_url, topics, keywords, score, viewcount, ]
            i += 1
        df = df.sort_values(['custom_score'], ascending=[0])
        df = df.reset_index(drop=True)
        return df
    ##### back to okay code
    
    def final_df(self):
        # Initialise results dataframe
        df = pd.DataFrame(columns=('title', 'thumbnail', 'url', 'topics', 
                                   'keywords', 'custom_score', 'views'))
        api_key = self.ytapi
        print(api_key)
        youtube = build( 'youtube', 'V3', developerKey= api_key)
        
        text = self.gpt_topic()
        #Searching for query with decreasing order as viewcount and result as video type
        results = youtube. search().list(q=text, part="snippet", type="video", 
                                         order="viewCount").execute( )
        
        df_final = self.new_order(results, df, youtube)
        return df_final
    
    def final_dict(self):
        df_final = self.final_df()
        df_final = df_final.drop(columns = ['custom_score','views'])
        dict_final = df_final.to_dict('records')
        return dict_final