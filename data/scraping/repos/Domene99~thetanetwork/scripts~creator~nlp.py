import openai
import mindsdb_sdk as mdb
import moviepy.editor as mp

class NLPWrapper:
    def __init__(self, openai_key, mdb_email, mdb_pass):
        self.mdb_email = mdb_email
        self.mdb_pass = mdb_pass
        openai.api_key = openai_key
    
    # def transcribeFake(self, file_name):
    #     audio = mp.VideoFileClip(file_name).audio
    #     audio.write_audiofile("audio.mp3")
    #     audio_file= open("audio.mp3", "rb")
    #     res = {"text": "Here is a test in which I am clearly talking a lot. I am talking about various topics. I am talking about vaping. I am talking about iPhones, Androids. I am talking about technology. I am talking about pen and paper. I am talking about programming, Python, JavaScript. I fucking hate marvel. Aunque tambien hablo espanol."}
        
    #     self.transcribed_text = res['text']

    def transcribe(self, file_name):
        audio = mp.VideoFileClip(file_name).audio
        audio.write_audiofile("audio.mp3")
        audio_file= open("audio.mp3", "rb")
        res = openai.Audio.transcribe("whisper-1", audio_file)
        self.transcribed_text = res['text']
    
    def extractFeaturesFromText(self):
        db = mdb.connect(login="jadomene99@gmail.com",password="KaMasQ9xMfkGBRU").list_databases()[0]
        query = db.query(f'SELECT * FROM mindsdb.count_swears_system WHERE script = \'{self.transcribed_text}\'')
        swear_count = query.fetch()
        query = db.query(f'SELECT * FROM mindsdb.sentiment_topics_updated_1 WHERE script = \'{self.transcribed_text}\'')
        topics = query.fetch()
        query = db.query(f'SELECT * FROM mindsdb.languages_updated WHERE script = \'{self.transcribed_text}\'')
        langs = query.fetch()
        query = db.query(f'SELECT * FROM mindsdb.brand_safety_updated WHERE script = \'{self.transcribed_text}\'')
        score = query.fetch()
        return swear_count, topics, langs, score
