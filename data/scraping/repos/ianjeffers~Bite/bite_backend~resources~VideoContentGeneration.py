import json

from flask import current_app
from flask_restful import Resource, reqparse
from services.DBService import DBService
from services.OpenAIService import OpenAIService
from services.WikipediaService import WikipediaService
from services.PineconeService import PineconeService
from services.HuggingFaceService import HuggingFaceService
import random

class VideoContentGeneration(Resource):
    def __init__(self):
        self.wikipedia_service = WikipediaService()
        self.db_service = DBService()
        self.openai_service = OpenAIService()  
        self.pinecone_service = PineconeService(index_name='bite')
        self.hugging_face_service = current_app.hugging_face_service

        self.moods = ['humorous', 'nostalgic', 'sad', 'ragebait'] 

    def post(self):
        parser = reqparse.RequestParser()
        parser.add_argument('topic', type=str, required=True, help='Education topic is required')
        data = parser.parse_args()

        # dummy_content = {
        #     'Tweet': "Learn about Pythagoras! He was a smart dude who discovered a neat property of triangles. I'm sure he'd be a blast at parties! #geometry #mathfacts"
        # }
                
        # Fetch Wikipedia page 
        summary = self.wikipedia_service.get_summary(data['topic'])
        if summary is None:
            return {'message': 'Page does not exist'}, 404

        # Randomly select a mood
        mood = random.choice(self.moods)

        # Create the prompt for OpenAI
        prompt = f'Given the following summary, please create an educational Tweet in a {mood} mood. Please include a specific fact from the summary. Your response should be in JSON format like this: {{"Tweet": "Your tweet here"}}.\n\n' + summary
        gpt_response = self.openai_service.generate_json(prompt, 350)

        # Extract the Tweet content
        content = json.loads(gpt_response.choices[0].text.strip())
        # content = content_json["Tweet"]


        # content = dummy_content
        content_string = content['Tweet']
        try:
            vector = self.hugging_face_service.generate_vector(content_string)
            content_db = self.db_service.save_content(content, 'video')
            self.pinecone_service.upsert(content_db.id, vector)
        except Exception as e:
            print("VIDEO", e)
            return {'message': 'Error in generating video content', 'error': str(e)}, 500

        return {'message': 'Video content generated successfully', 'content': content}, 201
