import openai
import json
from datetime import datetime
import os
from google.cloud import vision_v1
from dotenv import load_dotenv
import uuid
from google.oauth2 import service_account
import os
from dotenv import load_dotenv
from google.cloud import vision_v1
from google.auth import _cloud_sdk
import base64
import requests

# Get the API key from the environment variable
api_key = os.getenv('GOOGLE_API_KEY')

class EssayService:
    def __init__(self, essay_repository, essay_additional_text_repository, essay_theme_repository):
        self.essay_repository = essay_repository
        self.essay_additional_text_repository = essay_additional_text_repository
        self.essay_theme_repository = essay_theme_repository
    
    def get_essay_history(self, user_id):
        response = self.essay_repository.get_by_user_id(user_id)
        self.essay_repository.commit()
        return response
    
    def get_newest_created_essay(self, user_id):
        response = self.essay_repository.get_created_essay_by_user_id(user_id)
        self.essay_repository.commit()
        return response
    
    def get_unfinished_essays(self, user_id):
        response = self.essay_repository.get_unfinished_by_user_id(user_id)
        self.essay_repository.commit()
        return response
    
    def create_new_essay(self, theme_id, user_id, title):
        result = self.essay_repository.insert_new_essay(str(uuid.uuid4()), theme_id, user_id, title, "", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "False")
        self.essay_repository.commit()
        return result
    
    def save_essay(self, essay):
        result = self.essay_repository.insert_essay(essay)
        self.essay_repository.commit()
        return result
    
    def update_essay(self, essay):
        result = self.essay_repository.update_essay(essay)
        self.essay_repository.commit()
        return result
    
    def delete_essay(self, essay_id):
        response = self.essay_repository.delete_essay(essay_id)
        self.essay_repository.commit()
        return response

    def get_essay_analysis(self, id, user_id, theme_id, essay_title, essay_content):
        prompt = (
            "Você é um corretor experiente do Enem. Um aluno está solicitando que você corrija a redação que ele está te mandando."
            "Usando os moldes de correção de redação do Enem, analise e dê nota (nota de 0 a 200 para cada competência) com base no tema proposto o título e a redação do seu aluno.\n"
            f"Tema: {essay_title}\n"
            "Título da redação: O titulo da redação é a primeira linha do texto da redação\n"
            f"Texto da redação: {essay_content}\n\n"
            "Por favor, retorne a análise da redação seguindo estritamente essa estrutura:\n"
            "C1 Nota: \n"
            "C2 Nota: \n"
            "C3 Nota: \n"
            "C4 Nota: \n"
            "C5 Nota: \n"
            "Nota Total: \n"
            "C1 Explicação: \n"
            "C2 Explicação: \n"
            "C3 Explicação: \n"
            "C4 Explicação: \n"
            "C5 Explicação: \n"
            "Explicação Geral: \n"
        )
        
        # Send a request to the ChatGPT API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages = [
                {'role': 'user', 'content': prompt}
            ],
            # n=1,
            # stop=None,
            temperature=0.7,
        )
        
        # Extract and parse the API response
        analysis_text = response['choices'][0]['message']['content'].strip()
        analysis_lines = analysis_text.split('\n')
        analysis_dict = {line.split(': ')[0]: line.split(': ')[1] for line in analysis_lines if line.strip()}
        
        # Construct the essay analysis entity
        essay_analysis = {
            "id": id,
            "user_id": user_id,
            "theme_id": theme_id,
            "title": essay_title,
            "contents": essay_content,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_finished": True,
            "c1_grade": analysis_dict["C1 Nota"],
            "c2_grade": analysis_dict["C2 Nota"],
            "c3_grade": analysis_dict["C3 Nota"],
            "c4_grade": analysis_dict["C4 Nota"],
            "c5_grade": analysis_dict["C5 Nota"],
            "total_grade": analysis_dict["Nota Total"],
            "c1_analysis": analysis_dict["C1 Explicação"],
            "c2_analysis": analysis_dict["C2 Explicação"],
            "c3_analysis": analysis_dict["C3 Explicação"],
            "c4_analysis": analysis_dict["C4 Explicação"],
            "c5_analysis": analysis_dict["C5 Explicação"],
            "general_analysis": analysis_dict["Explicação Geral"]
        }
        self.update_essay(essay_analysis)
        return essay_analysis

    def perform_ocr(self, content):
        # Encode the image content to base64
        image_base64 = base64.b64encode(content).decode('utf-8')
        
        # Create the JSON request payload
        request_payload = {
            "requests": [
                {
                    "image": {
                        "content": image_base64
                    },
                    "features": [
                        {
                            "type": "TEXT_DETECTION"
                        }
                    ]
                }
            ]
        }
        
        # Send a request to the Google Cloud Vision API
        response = requests.post(
            url=f'https://vision.googleapis.com/v1/images:annotate?key={api_key}',
            headers={'Content-Type': 'application/json'},
            json=request_payload
        )
        
        # Check for a valid response
        if response.status_code == 200:
            response_data = response.json()
            texts = response_data['responses'][0].get('textAnnotations', [])
            if texts:
                return texts[0]['description']  # Return the entire text from the image
        else:
            return response.status_code, 500
        return ""
