import os
import requests
from openai import OpenAI
import subprocess
from PIL import Image

def translate_word(word):
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

def generate_dalle_image(objective):
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

def download_and_save_image(image_url, objective_value, folder_path):
    response = requests.get(image_url)

    if response.status_code == 200:
        new_file_name = f"{objective_value}.jpg"
        new_file_path = os.path.join(folder_path, new_file_name)

        with open(new_file_path, 'wb') as file:
            file.write(response.content)
        print(f"Image saved as {new_file_path}")
        return new_file_name
    else:
        print(f"Failed to download image. Status code: {response.status_code}")
        return None

def run_process_script(IMAGE):
    script_path = "process.py"
    command = f"python {script_path} data/{IMAGE}"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the process.py script: {e}")

def run_main_script(NAME, IMAGE_PROCESSED, Elevation):
    script_path = "main.py"
    config_file = "configs/image.yaml"
    command = f"python {script_path} --config {config_file} input=data/{IMAGE_PROCESSED} save_path={NAME} elevation={Elevation} force_cuda_rast=True"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the main.py script: {e}")

def run_main2_script(NAME, IMAGE_PROCESSED, Elevation):
    script_path = "main2.py"
    config_file = "configs/image.yaml"
    command = f"python {script_path} --config {config_file} input=data/{IMAGE_PROCESSED} save_path={NAME} elevation={Elevation} force_cuda_rast=True"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the main2.py script: {e}")

def optimize_stage_1(image_block: Image.Image, preprocess_chk: bool, elevation_slider: float):
    if not os.path.exists('tmp_data'):
        os.makedirs('tmp_data')
    if preprocess_chk:
        # save image to a designated path
        image_block.save(os.path.join('tmp_data', 'tmp.png'))

        # preprocess image
        print(f'python process.py {os.path.join("tmp_data", "tmp.png")}')
        subprocess.run(f'python process.py {os.path.join("tmp_data", "tmp.png")}', shell=True)
    else:
        image_block.save(os.path.join('tmp_data', 'tmp_rgba.png'))

    # stage 1
    subprocess.run(f'python main.py --config {os.path.join("configs", "image.yaml")} input={os.path.join("tmp_data", "tmp_rgba.png")} save_path=tmp mesh_format=glb elevation={elevation_slider} force_cuda_rast=True', shell=True)

    return os.path.join('logs', 'tmp_mesh.glb')


def optimize_stage_2(elevation_slider: float):
    # stage 2
    subprocess.run(f'python main2.py --config {os.path.join("configs", "image.yaml")} input={os.path.join("tmp_data", "tmp_rgba.png")} save_path=tmp mesh_format=glb elevation={elevation_slider} force_cuda_rast=True', shell=True)

    return os.path.join('logs', 'tmp.glb')

def main():
    # Input Thai word
    input_word = "ตึก"
    # Translate to English
    objective = translate_word(input_word)
    api_key = "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"
    os.environ["OPENAI_API_KEY"] = api_key

    image_url = generate_dalle_image(objective)

    print(objective)
    print(image_url)

    folder_path = "data"
    new_file_name = download_and_save_image(image_url, objective, folder_path)

    if new_file_name:
        IMAGE = new_file_name
        # Preprocess
        run_process_script(IMAGE)
        
        NAME = os.path.splitext(IMAGE)[0]
        IMAGE_PROCESSED = NAME + '_rgba.png'

        Elevation = 0
        # Stage 1
        optimized_stage_1_output = optimize_stage_1(Image.open(os.path.join(folder_path, IMAGE)), True, Elevation)

        # Stage 2
        optimized_stage_2_output = optimize_stage_2(Elevation)

        # # Stage 1
        # run_main_script(NAME, IMAGE_PROCESSED, Elevation)

        # # Stage 2
        # run_main2_script(NAME, IMAGE_PROCESSED, Elevation)

if __name__ == "__main__": 
    main()