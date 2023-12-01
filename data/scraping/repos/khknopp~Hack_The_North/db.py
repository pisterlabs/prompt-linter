from dotenv import load_dotenv
import pickle as pk
from question_generator import get_questions, split_execution
from summarize import create_fragments, update_summary

from video_utils import run_transcript
import os
import cohere

load_dotenv()
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
co = cohere.Client(COHERE_API_KEY)

class get_params: 
    def __init__(self, link, title):
        self.link = link
        self.title = title
        
        self.transcription_timestamps, self.transcription = run_transcript(self.link)
        
        # self.transcription = self.transcription[:15000]
        open, closed, self.summary = split_execution(co, self.transcription)
        
        print(str(len(self.summary)) + "Length of summary after splitexec")
        print(self.summary)
        self.summary_array = self.summary
        self.summary = " ".join(self.summary)
        
        self.open_questions, self.closed_questions, self.closed_answers, self.open_answers =  get_questions(co, open, closed)
        
        print(self.open_questions)
        print(self.open_answers)
        print(self.closed_questions)
        print(self.closed_answers)

        self.transcription_timestamps = pk.dumps(self.transcription_timestamps)
        
        self.boundaries = pk.dumps([[70, 100], [400, 500], [700, 750], [800, 1200]])
        self.personalized_summary = []

        
    def add_video(self, conn):
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS videos (link TEXT PRIMARY KEY, summary TEXT, closed_questions BLOB, closed_answers BLOB, open_questions BLOB, open_answers BLOB, title TEXT, transcript TEXT, boundaries BLOB, transcription_timestamps BLOB, personalized_summary TEXT)"
            )
            conn.commit()
            cur.execute(
                "INSERT INTO videos (link, summary, closed_questions, closed_answers, open_questions, open_answers, title, transcript, boundaries, transcription_timestamps, personalized_summary) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s, %s, %s)", (self.link, self.summary, pk.dumps(self.closed_questions), pk.dumps(self.closed_answers), pk.dumps(self.open_questions), pk.dumps(self.open_answers), self.title, self.transcription, self.boundaries, self.transcription_timestamps, self.personalized_summary))
        conn.commit()
        print("Added succesfully")
        
        
    def update_db(self, conn, link):
        ## GET FROM DB
        with conn.cursor() as cur:
            cur.execute("SELECT boundaries, transcription_timestamps, summary FROM videos WHERE link = %s", (link,))
            data = cur.fetchall()
            
            print(data[0].transcription_timestamps)
            self.transcription_timestamps = pk.loads(data[0].transcription_timestamps)
        
            print(data[0].boundaries)
            self.boundaries = pk.loads(data[0].boundaries)
            
            print(data[0].summary)
            self.summary = data[0].summary
            
            print("all db reading was successful")
        
            fragments = create_fragments(self.transcription_timestamps, self.boundaries)
            print(fragments)
            
            self.personalized_summary, _ = update_summary(co, self.summary_array, fragments)
            print(self.personalized_summary)
            self.personalized_summary = " ".join(self.personalized_summary)
            
            cur.execute("UPDATE videos SET personalized_summary = '%s' WHERE link = '%s'", (self.personalized_summary, link,))
            conn.commit()
            
    def QandA(self, conn, link):
        with conn.cursor() as cur:
            cur.execute("SELECT open_questions, open_answers FROM videos WHERE link = %s", (link,))
            data = cur.fetchall()
            
            print(data[0].open_questions)
            self.open_questions = pk.loads(data[0].open_questions)
        
            print(data[0].open_answers)
            self.open_answers = pk.loads(data[0].open_answers)
            
        return self.open_questions, self.open_answers
    
    def get_summaries(self, conn, link):
        with conn.cursor() as cur:
            cur.execute("SELECT summary, personalized_summary FROM videos WHERE link = %s", (link,))
            data = cur.fetchall()
            
            print(data[0].summary)
            self.summary = data[0].summary
        
            print(data[0].personalized_summary)
            self.personalized_summary = data[0].personalized_summary
            
        return self.summary, self.personalized_summary
    
                    
        
        
    
        
        
        
    


#add_video(conn, "9syvZr-9xwk", "This is a summary of the video", ["These are the closed questions"], ["These are the closed answers"], ["These are the open questions"], ["These are the open answers"], "This is the title", "This is the transcript")

