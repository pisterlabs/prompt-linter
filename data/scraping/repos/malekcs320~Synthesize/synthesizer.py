import openai
from dotenv import load_dotenv
import os ,json
from db import Database
import pika
from bson import ObjectId
from utils_recom import distances_from_embeddings, indices_of_nearest_neighbors_from_distances

class Synthesizer:
    def __init__(self):
        load_dotenv()
        db_uri = os.getenv("DB_URI")
        db = Database(db_uri, 'myDataBase').db
        self.name = "synthesis"
        self.transcript_collection = db["transcripts"]
        self.notes_collection=db["notes"]
        self.synthesis_collection=db["synthesis"]
        self.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.api_key
        self.model_engine = "text-davinci-003"
        # GPT-3 prompt
        self.prompt = " based on some notes I have taken in class and the this transcription of my professor lecture, \
        Please arrange the synthesized document with subtitles to help me follow the different parts: \
        the transcription: \n {input_transcript} \
        the notes : \n {input_notes} . \
        recommend 5 internet resources linked to this synthesis,please generate it  a full stylized html format  differenciate titles  paragraphs."
        self.quizz_prompt = " based on this synthesis, \
        give me a 3 question quizz with answers \
        please generate it in a full stylized html format and differenciate answers from question. my fianl goal is to make the answers invesible at first and only show them when the user clicks on a button and make them visible when the user reclicks on it\
        the synthesis: \n {input_document} \n "
        self.tags_prompt = " Extract keywords from this text:\n\n {input_document} \n "
        self.quizz=""
        self.tags=""
        self.embedding=[]
        self.recommendations=""
        self.receive()
       
        
    def generate_summary(self, transcript, notes):
        # Call the OpenAI API to generate a summary of the input text
        response = openai.Completion.create(
            engine=self.model_engine,
            prompt=self.prompt.format(input_transcript=transcript, input_notes=notes),
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0,
        )
        return response.choices[0].text.strip()
    

    def generate_quizz(self, document):
        # Call the OpenAI API to generate a summary of the input text
        response = openai.Completion.create(
            engine=self.model_engine,
            prompt=self.quizz_prompt.format(input_document=document),
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0,
        )
        return response.choices[0].text.strip()
    
    def generate_tags(self, document):
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=self.tags_prompt.format(input_document=document),
            max_tokens=50,
            top_p=1.0,
            frequency_penalty=0.8,
            presence_penalty=0.0,
            temperature=0.5
        )
        return response.choices[0].text.strip()

    def generate_tags_embeds(self, document):
        tags = document.replace("\n", " ")
        return openai.Embedding.create(input = [tags], model="text-embedding-ada-002")['data'][0]['embedding']

    def generate_recommendations(self, tags):
        recommendations = []
        # retrieve embeddings with id from db
        try:
            embeddings = self.synthesis_collection.find({}, {"_id": 1, "embedding": 1})
            embeddings = list(embeddings)
        except StopIteration:
            print("No synthesis found in db")
            return []

        ids_dic = dict()
        for index in range(len(embeddings)):
            ids_dic[index] = embeddings[index]["_id"]
        # convert embeddings to list of lists
        embeddings = [embedding["embedding"] for embedding in embeddings]
        # generate embeddings for tags
        self.embedding = self.generate_tags_embeds(tags)

        # compute distances between tags and embeddings and keep only distances < 0.1
        distances = distances_from_embeddings(self.embedding, embeddings)
        distances = [distance for distance in distances if distance < 0.2]
        print (f"length of recommendations {len(distances)}")
        indices = []

        if len(distances) >= 3:
            indices = indices_of_nearest_neighbors_from_distances(distances)[:3]
            _ids = [ids_dic[index] for index in indices]
            # retrieve synthesis names with indices
            recommendations = self.synthesis_collection.find({"_id": {"$in": _ids}}, {"_id": 1, "name": 1})
            recommendations = list(recommendations)
        return recommendations


    def synthesize(self,transcript_id,notes_id):
        
        # Retrieve transcript from db with transcript_id
        # transcript_id=ObjectId(transcript_id)

        transcript=""
        notes=""
        try:
            transcript = self.transcript_collection.find({"_id":ObjectId(transcript_id) }).limit(1).next()  
            notes = self.notes_collection.find({"_id":ObjectId(notes_id) }).limit(1).next()  
        except StopIteration:
            print("Iteration Error")
        if not transcript:
            return "Transcript not found."
        if not notes:
            return "Notes not found."
        
        #convert ObjectId to string
        transcript['_id'] = str(transcript['_id'])
        transcript_dict = dict(transcript)
        notes['_id'] = str(notes['_id'])
        notes_dict = dict(notes)
        # Synthesize document using Synthesizer
        document = self.generate_summary(transcript_dict['text'], notes_dict['text'])
        print("************************************\n")
        f = open("static/synthesis.html", "w")
        f.write(document)
        f.close()
        #generate tags
        self.tags=self.generate_tags(document)
        print(self.tags)
        #generate recommendations
        self.recommendations=self.generate_recommendations(self.tags)
        f = open("static/recommendations.html", "w")
        #convert recommendations to html
        html = "<ul>"
        for recommendation in self.recommendations:
            html += f"<li>{recommendation['_id']}: {recommendation['name']}</li>"
        html += "</ul>"
        f.write(html)
        f.close()
        
        print(self.recommendations)
        # generate quizz
        self.quizz=self.generate_quizz(document)
        f = open("static/quizz.html", "w")
        f.write(self.quizz)
        f.close()
        # print(self.quizz)
        # Return synthesized document
        return document
    
    
    
    def receive(self):
        connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
        channel = connection.channel()
        channel.queue_declare(queue='synthesize')

        def callback(ch, method, properties, body):
            jsonbody=json.loads(body)
            transcript_id=jsonbody["transcript_id"]
            notes_id=jsonbody["notes_id"]
            result=self.synthesize(transcript_id,notes_id)
            self.synthesis_collection.insert_one({'name': self.name,'text':result , 'transcript_id':transcript_id , 'notes_id':notes_id , 'quizz': self.quizz, 'tags': self.tags, 'embedding': self.embedding, 'recommendations': self.recommendations})

        channel.basic_consume(queue='synthesize', on_message_callback=callback, auto_ack=True)

        print(' [*] Waiting for messages. To exit press CTRL+C')
        channel.start_consuming()
    
    
if __name__ == "__main__":
    synthesizer=Synthesizer()