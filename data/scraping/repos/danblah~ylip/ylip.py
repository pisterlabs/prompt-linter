# Import required libraries
import os
import uuid
import inspect
import requests
import base64
import time
from itertools import cycle
import threading
import sys
import glob
import json
import shutil
from io import BytesIO
from threading import Thread

from dotenv import load_dotenv
import openai
from elevenlabs import generate, save

from tinydb import TinyDB, Query
from flask import Flask, render_template_string, request, jsonify, render_template
from flask_socketio import SocketIO, emit
import simple_websocket
import gevent

from PIL import Image
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
import moviepy.video.fx.all as vfx

############################
# Import .env variables
load_dotenv()

# Define API keys
openai.api_key = os.getenv('OPENAI_KEY')
elevenlabs_api_key = os.getenv('ELEVENLABS_KEY')

# Define OpenAI settings
openai_gpt_model = os.getenv('OPENAI_MODEL')
openai_gpt_max_tokens = os.getenv('OPENAI_MAX_TOKENS')
openai_gpt_temp = os.getenv('OPENAI_TEMP')
openai_gpt_top_p = os.getenv('OPENAI_TOP_P')

openai_dalle_size = os.getenv('OPENAI_DALLE_SIZE')

# Define Eleven Labs settings
elevenlabs_model = os.getenv('ELEVENLABS_MODEL')
elevenlabs_voice = os.getenv('ELEVENLABS_VOICE')

# Define default prompts
gpt_child_prompt = os.getenv('PROMPT_THEME')
gpt_pre_prompt = os.getenv('PROMPT_PRE')
gpt_image_prompt = os.getenv('PROMPT_IMAGE')
with open(os.getenv('PROMPT_OUTPUT_FORMAT'), 'r') as file:
    gpt_format_prompt = file.read()


############################

# Define databases
db_content = TinyDB('data/'+ os.getenv('DB_CONTENT'))
db_config = TinyDB('data/'+ os.getenv('DB_CONFIG'))

# Generate a unique 7 character id
unique_id = str(uuid.uuid4())[:7]

start_time = time.time()

# setup variables for the current story
stories = db_content.all()
stories.sort(key=lambda x: x['timestamp'], reverse=True)
story_data = stories[0]['story_data']
story_id = stories[0]['unique_id']
story_folder = 'static/stories/' + story_id

############################


app = Flask(__name__, static_folder='./static')
socketio = SocketIO(app, async_mode='threading')
app.secret_key = os.getenv('SECRET_KEY')

@app.route("/settings", methods=['GET', 'POST'])
def settings():
    routes = {
        str(rule): rule.endpoint.replace('_', ' ').title() 
        for rule in app.url_map.iter_rules() if not 'static' in rule.rule
    }
    if request.method == 'POST':
        for key, value in request.form.items():
            if key in os.environ:
                os.environ[key] = value
                # set_key('.env', key, value) # Commented out as set_key is not defined
        return "Settings updated successfully!"
    else:
        env_vars = {
            key: ('*****' if 'key' in key.lower() else os.getenv(key), 'prompt' in key.lower())
            for key in sorted(os.environ.keys())
            if key in open('.env').read() and key != '_'
        }
        
        return render_template('settings.html', env_vars=env_vars, routes=routes)

@app.route("/", methods=['GET', 'POST'])
def home():
    global db, gpt_child_prompt, gpt_pre_prompt, gpt_image_prompt

    global stories, story_data, story_id, story_folder

    routes = {
        str(rule): rule.endpoint.replace('_', ' ').title() 
        for rule in app.url_map.iter_rules() if not 'static' in rule.rule
    }

    initial_movie_status = "Try generating a new movie."

    if stories:
        stories.sort(key=lambda x: x['timestamp'], reverse=True)
        story = stories[0]['story_data']
        story_id = stories[0]['unique_id']
        # Check if 'prompt' exists in story_data and set it as gpt_child_prompt
        if 'prompt_child' in story:
            gpt_child_prompt = story['prompt_child']
            gpt_pre_prompt = story['prompt_pre']
            gpt_image_prompt = story['prompt_image']
    # Define HTML inline
    if story:
        story_paragraphs = story['story']
        story_text = "".join(f"<p>{para}</p>" for para in story_paragraphs.values())
    else:
        story_text = "<p>No story available.</p>"

    movie_file = f"static/stories/{story_id}/{story_id}_movie.mp4" if story_id else None
    if movie_file and os.path.isfile(movie_file):
        movie_element = f"<video width='320' height='240' controls><source src='/{movie_file}' type='video/mp4'></video>"
        direct_link = f"<a href='/{movie_file}'>Video link</a>"
        initial_movie_status = "To generate a new movie, first generate a new story."
    else:
        movie_element = ""
        direct_link = ""
        initial_movie_status = ""
        
    return render_template('index.html', 
                           gpt_child_prompt=gpt_child_prompt, 
                           gpt_pre_prompt=gpt_pre_prompt, 
                           gpt_image_prompt=gpt_image_prompt, 
                           story_text=story_text, 
                           movie_element=movie_element, 
                           direct_link=direct_link, 
                           initial_movie_status=initial_movie_status,
                           routes=routes)

@socketio.on('save_prompt_theme')
def handle_save_prompt(prompt_theme):
    global gpt_child_prompt
    gpt_child_prompt = prompt_theme
    emit('prompt_status', {'status': 'Theme prompt saved successfully!'})

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - Saved new prompt_theme: {gpt_child_prompt}")

@socketio.on('save_prompt_pre')
def handle_save_prompt(prompt_pre):
    global gpt_pre_prompt
    gpt_pre_prompt = prompt_pre
    emit('prompt_status', {'status': 'Pre prompt saved successfully!'})

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - Saved new prompt_pre: {gpt_pre_prompt}")

@socketio.on('save_prompt_image')
def handle_save_prompt(prompt_image):
    global gpt_image_prompt
    gpt_image_prompt = prompt_image
    emit('prompt_status', {'status': 'Image prompt saved successfully!'})

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - Saved new prompt_image: {gpt_image_prompt}")

@socketio.on('generate_story')
def generate_story():
    global unique_id, gpt_child_prompt, gpt_pre_prompt, gpt_format_prompt, db
    global stories, story_data, story_id, story_folder


    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: Starting")
    try:
        gpt_full_prompt = gpt_pre_prompt + " The story's main theme is: " + gpt_child_prompt + "\n" + gpt_image_prompt + "\n" + gpt_format_prompt

        # Generate new story text
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: About to request a new story, here's the prompt:" + "\n" + gpt_full_prompt)
        emit('story_status', {'status': 'Generating a new story'})
        
        def generate_completion():
            global completion
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                max_tokens=4096,
                temperature=0.15,
                top_p=1,
                messages=[
                    {"role": "user", "content": gpt_full_prompt}
                ]
            )

        def emit_status():
            messages = cycle([
                'Okay, let me think...',
                'Shhhh... Loud noise distracts me!',
                'Here we go, got an idea...',
                'Now, where did I put my pencil...',
                'Oh no! Do you have any paper???',
                'Getting warmed up now...',
                'Creative juices are really flowing!!!',
                'Just putting on the finishing touches...',
                'Almost done...'
            ])
            while thread.is_alive():
                socketio.emit('story_status', {'status': next(messages)}, namespace='/')
                time.sleep(5)

        thread = threading.Thread(target=generate_completion)
        thread.start()

        emit_thread = threading.Thread(target=emit_status)
        emit_thread.start()

        thread.join()
        emit_thread.join()

       
        # Remove leading and trailing quotes from story_text
        story_text = completion.choices[0].message.content.strip('"')
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: Got a new story: " + "\n" + story_text)
        emit('story_status', {'status': 'Here comes a brand new story!'})
        # Load the JSON data from story_text
        # Added json.loads() inside a try-except block to handle JSONDecodeError
        try:
            # Replace escaped double quotes with actual double quotes
            story_text = story_text.replace('\\"', '"')
            story_data = json.loads(story_text)

            # Insert the prompts as a separate objects in the story_data dictionary
            story_data.update({"prompt_child": gpt_child_prompt})
            story_data.update({"prompt_pre": gpt_pre_prompt})
            story_data.update({"prompt_image": gpt_image_prompt})

        except json.JSONDecodeError as json_err:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: Error in parsing JSON: {json_err}")
            return

        # Save the contents of story_data to db.json with a timestamp
        db.insert({'timestamp': time.time(), 'unique_id': unique_id, 'story_data': story_data})

        # Emit the new story to the client
        socketio.emit('new_story', {'story': story_data['story']})

    except Exception as e:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: Error in function: {e}")

@socketio.on('create_movie')
def create_movie():
    global stories, story_data, story_id, story_folder

    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - Running create_movie")

    # Check if a directory named static/stories/story_id exists. if not create it.
    if not os.path.exists(story_folder):
        os.makedirs(story_folder, exist_ok=True)


    # In the directory named story_id, check if any a video already exist.
    video_clip = glob.glob(f"{story_folder}/*.mp4")

    # In the directory named story_id, check if images already exist.
    image_clips = glob.glob(f"{story_folder}/*.png")

    # In the directory named story_id, check if audio files already exist.
    audio_clips = glob.glob(f"{story_folder}/*.wav")

    #Check if everything exists to make a movie

    try:
        # Check if audio_clips exist. if not, generate audio_clips
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: Checking for existing audio clips")
        try:
            emit('movie_status', {'status': 'Checking for existing audio files'})
        except RuntimeError as e:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: Error emitting status: {e}")
            return

        if not audio_clips:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: No audio files, generating")
            emit('movie_status', {'status': 'Creating text-to-speech audio files'})
            generate_audio()
            # Wait for the audio files to be generated before proceeding
            # Check for each story_paragraph in story_data
            for i in range(1, len(story_data['story']) + 1):
                while not os.path.exists(f"{story_folder}/{story_id}_story_paragraph_{i}.wav"):
                    time.sleep(1)
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: Audio file generation complete")

        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: Existing audio files, skipping generation")

    except Exception as e:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: Audio regeneration error: {e}")

    try:
        # Check if images exist. if not, generate them
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: Checking for existing image files")
        emit('movie_status', {'status': 'Checking for existing image files'})

        if not image_clips:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: No image files")
            # Check if there is an image_urls object within story_data
            if "image_urls" not in story_data:
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: No image files, generating")
                emit('movie_status', {'status': 'Generating new images to go with the story'})
                # If not, run the generate_images() function and wait until the function completes
                generate_images()
                while "image_urls" not in story_data:
                    time.sleep(1)

            # Check if the image_urls are responding 403, if so regenerate images
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: Checking if images are expired") 
            emit('movie_status', {'status': 'Checking if image URLs have expired'})       
            image_url = story_data["image_urls"][f"image_url_1"]
            response = requests.get(image_url)
            if response.status_code == 403:
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: Image urls expired, regenerating images")
                emit('movie_status', {'status': 'Image urls expired, regenerating images'})
                generate_images()

                print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: Image regeneration success")
                emit('movie_status', {'status': 'Image regeneration success'})

    except Exception as e:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: Image regeneration error: {e}")
                        
    finally:
        # Download images
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: downloading images")
        emit('movie_status', {'status': 'Downloading images to use in the video'})
        i = 1
        while True:
            image_filename = f"{story_folder}/{story_id}_image_url_{i}.png"
            
            if not os.path.exists(image_filename):
                if f"image_url_{i}" in story_data["image_urls"]:
                    image_url = story_data["image_urls"][f"image_url_{i}"]
                    response = requests.get(image_url)
                    if response.status_code == 200:
                        with open(image_filename, 'wb') as f:
                            f.write(response.content)
                            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: downloaded new image: {image_filename}")
                    else:
                        break
                else:
                    break
            i += 1
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {__name__}: images downloaded")
        emit('movie_status', {'status': 'Images downloaded'})

        # Generate video clips from the audio_clips and image_clips
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {__name__}: creating movie clips")
        emit('movie_status', {'status': 'Combining the audio and image files into video clips'})
        i = 1
        video_clips = []  # Initialize video_clips as an empty list
        while True:
            audio_filename = f"{story_folder}/{story_id}_story_paragraph_{i}.wav"
            image_filename = f"{story_folder}/{story_id}_image_url_{i}.png"
            if os.path.exists(audio_filename) and os.path.exists(image_filename):
                audio = AudioFileClip(audio_filename)
                image = ImageClip(image_filename, duration=audio.duration)
                video_clip = image.set_audio(audio)
                video_clips.append(video_clip.fx(vfx.speedx, 0.90))  # Slow down the video by 5%
            else:
                break
            i += 1

        # Create a single video from all of the video_clips
        if video_clips:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {__name__}: creating the movie")
            emit('movie_status', {'status': 'Combining video clips into a single movie'})

            final_video = concatenate_videoclips(video_clips)
            final_video.write_videofile(f"{story_folder}/{story_id}_movie.mp4", fps=24, codec='libx264', audio_codec='aac')
                
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {__name__}: movie created success")
            emit('movie_status', {'status': 'Movie creation success'})
            emit('new_movie', {'status': f"{story_folder}/{story_id}_movie.mp4"})

# Generate images for each image_prompt in story_data, save the image URLs, and update the database
def generate_images():
    global unique_id, db
    global stories, story_data, story_id, story_folder

    try:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - Running generate_images")

        # Initialize an empty dictionary for image_urls
        story_data["image_urls"] = {}

        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - print story_data: {story_data}")

        i = 1
        while f"image_prompt_{i}" in story_data["image_prompts"]:
            prompt = story_data["image_prompts"][f"image_prompt_{i}"]
            
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - Generating new image, prompt: {prompt}")

            response = openai.Image.create(
                prompt=prompt,
                n=1,
                size="1024x1024",
            )
            image_url = response["data"][0]["url"]

            # Save the image URL as image_url_{i} in the story_data object
            story_data["image_urls"][f"image_url_{i}"] = image_url

            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - Request sent:\n{prompt}\n\nResponse received:\n{image_url}\n\n")
            i += 1
        
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - Image generation complete, new story_data: {story_data}")

        # Update the story_data in the database with the new image_urls
        db.update({'story_data': story_data}, Query().unique_id == story_id)

        create_movie()

    except Exception as e:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - Error in generate_images function: {e}\n")

def generate_audio():
    global stories, story_data, story_id, story_folder

    try:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: Running generate_audio")

        # Initialize an empty dictionary for audio_urls
        story_data["audio_urls"] = {}

        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: print story_data: {story_data}")

        # Check if a directory named story_id exists. if not create it.
        if not os.path.exists(story_folder):
            os.makedirs(story_folder, exist_ok=True)

        i = 1
        while f"story_paragraph_{i}" in story_data["story"]:
            text = story_data["story"][f"story_paragraph_{i}"]
            
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: Generating new audio, text: {text}")

            audio_bytes = generate(
                api_key=elevenlabs_api_key,
                text=text,
                voice="Bella",
                model="eleven_monolingual_v1"
            )

            # Save the audio file as story_paragraph_{i}.wav in the unique_id directory
            filename = f"{story_folder}/{story_id}_story_paragraph_{i}.wav"
            save(
                audio=audio_bytes,               # Audio bytes (returned by generate)
                filename=filename,               # Filename to save audio to
            )
            i += 1

    except Exception as e:
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} - {inspect.stack()[0][3]}: Error in generate_audio function: {e}")

if __name__ == '__main__':
    socketio.run(app,port=5000,debug=True)
    serve(app)