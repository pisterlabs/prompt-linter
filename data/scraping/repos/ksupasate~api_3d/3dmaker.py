import os
import requests
from openai import OpenAI
import subprocess
from fastapi import FastAPI, HTTPException

app = FastAPI()

class Text3D:
    def __init__(self, api_key):
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key

    def translate_word(self, word):
        try:
            response = requests.get(f"https://r6gqmtszgrkj7d27r43rtas37u0joaqc.lambda-url.ap-southeast-1.on.aws/translate/sample/{word}")

            if response.status_code == 200:
                data = response.json()
                translated_word = data.get('message')
                return translated_word
            else:
                return f"Request failed with status code {response.status_code}"
        except requests.exceptions.RequestException as e:
            return f"Request exception: {e}"
        except Exception as e:
            return f"An error occurred: {e}"

    def generate_dalle_image(self, objective):
        client = OpenAI()
        prompt = f"Generate a realistic image of a {objective} against a pure white background, ensuring that the object is fully visible without any part being cut off."

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )

        if response and response.data and len(response.data) > 0:
            image_url = response.data[0].url
            return image_url
        else:
            return None

    def download_and_save_image(self, image_url, objective_value, folder_path):
        response = requests.get(image_url)

        if response.status_code == 200:
            objective_value_cleaned = objective_value.replace(" ", "_")

            new_file_name = f"{objective_value_cleaned}.jpg"
            new_file_path = os.path.join(folder_path, new_file_name)

            with open(new_file_path, 'wb') as file:
                file.write(response.content)
            print(f"Image saved as {new_file_path}")
            return new_file_name
        else:
            print(f"Failed to download image. Status code: {response.status_code}")
            return None

    def run_process_script(self, IMAGE):
        script_path = "process.py"
        command = f"python3 {script_path} data/{IMAGE}"
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running the process.py script: {e}")

    def run_main_script(self, NAME, IMAGE_PROCESSED, Elevation):
        script_path = "main.py"
        config_file = "configs/image.yaml"
        command = f"python3 {script_path} --config {config_file} input=data/{IMAGE_PROCESSED} save_path={NAME} mesh_format=glb elevation={Elevation} force_cuda_rast=True"
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running the main.py script: {e}")

    def run_main2_script(self, NAME, IMAGE_PROCESSED, Elevation):
        script_path = "main2.py"
        config_file = "configs/image.yaml"
        command = f"python3 {script_path} --config {config_file} input=data/{IMAGE_PROCESSED} save_path={NAME} mesh_format=glb elevation={Elevation} force_cuda_rast=True"
        try:
            subprocess.run(command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running the main2.py script: {e}")

    def gen_3d(self, prompt):
        # Translate to English
        objective = self.translate_word(prompt)
        image_url = self.generate_dalle_image(objective)
        print(objective)
        print(image_url)

        folder_path = "data"
        new_file_name = self.download_and_save_image(image_url, objective, folder_path)

        if new_file_name:
            IMAGE = new_file_name
            # Preprocess
            self.run_process_script(IMAGE)
            NAME = os.path.splitext(IMAGE)[0]
            IMAGE_PROCESSED = NAME + '_rgba.png'
            Elevation = 0
            # Stage 1
            self.run_main_script(NAME, IMAGE_PROCESSED, Elevation)
            # Stage 2
            self.run_main2_script(NAME, IMAGE_PROCESSED, Elevation)
            return f"Output file: {NAME}.glb"
        else:
            return "Failed to generate 3D text."

@app.get("/predict/{prompt}")
async def root(prompt):
    model = Text3D(api_key="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")
    result = model.gen_3d(prompt)
    return {"output_path": result}
