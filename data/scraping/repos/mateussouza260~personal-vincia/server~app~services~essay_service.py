import openai
import json
from datetime import datetime
import os
from google.cloud import vision_v1
from dotenv import load_dotenv


load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')


# Instantiate the Google Cloud Vision client
client = vision_v1.ImageAnnotatorClient.from_service_account_file(credentials_path)

class EssayService:
    def __init__(self, essay_repository, essay_additional_text_repository, essay_theme_repository):
        self.essay_repository = essay_repository
        self.essay_additional_text_repository = essay_additional_text_repository
        self.essay_theme_repository = essay_theme_repository
    
    def get_essay_history(self, user_id):
        """Retrieve a user's essay history."""
        return self.essay_repository.get_by_user_id(user_id)
    
    def get_unfinished_essays(self, user_id):
        """Retrieve a user's unfinished essays."""
        return self.essay_repository.get_unfinished_by_user_id(user_id)
    
    def save_essay(self, essay):
        """Save an essay, either inserting a new one or updating an existing one."""
        existing_essay = self.essay_repository.get_by_id(essay.essay_id)
        if existing_essay:
            return self.essay_repository.update_essay(essay)
        else:
            return self.essay_repository.insert_essay(essay)
    
    def delete_essay(self, essay_id):
        """Delete an essay by its ID."""
        return self.essay_repository.delete_essay(essay_id)

    def get_essay_analysis(essay_id, user_id, theme_id, theme_title, essay_title, essay_content):
        # Construct the prompt
        prompt = (
            "Você é um corretor experiente do Enem. Um aluno está solicitando que você corrija a redação que ele está te mandando."
            "Usando os moldes de correção de redação do Enem, analise e dê nota com base no tema proposto o título e a redação do seu aluno.\n"
            f"Tema: {theme_title}\n"
            f"Título da redação: {essay_title}\n"
            f"Texto da redação: {essay_content}\n\n"
            "Please provide an analysis of the essay with the following structure:\n"
            "Por favor, retorne a analise da redação seguindo extritamente essa estrutura:"
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
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0.5,
        )
        
        # Extract and parse the API response
        analysis_text = response.choices[0].text.strip()
        analysis_lines = analysis_text.split('\n')
        analysis_dict = {line.split(': ')[0]: line.split(': ')[1] for line in analysis_lines}
        
        # Construct the essay analysis entity
        essay_analysis = {
            "essay_id": essay_id,
            "user_id": user_id,
            "theme_id": theme_id,
            "title": essay_title,
            "content": essay_content,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # current datetime in a string format
            "is_finished": True,  # assuming the analysis indicates a finished essay
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
        print(essay_analysis)
        return essay_analysis

    def perform_ocr(content):
        image = vision_v1.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        if texts:
            return texts[0].description  # Return the entire text from the image
        return ""
