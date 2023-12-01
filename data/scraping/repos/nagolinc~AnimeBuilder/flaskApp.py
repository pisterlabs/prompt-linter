import queue
import concurrent.futures
import os
import openai
import argparse
from flask import Flask, render_template, request, jsonify
import json
import animeCreator
from animeCreator import AnimeBuilder, getFilename
import uuid
from flask_ngrok2 import run_with_ngrok
import dataset
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = Flask(__name__)

def handleExtraTemplates(extraTemplates):
    templates=extraTemplates.split("===")
    for template in templates:
        #find the first line that starts with > and use it to get template name
        lines=template.split("\n")
        templateName=None
        for line in lines:
            if line.startswith(">"):
                templateName=line[1:].strip() #remove the > and any whitespace
                break
        #make sure we got a template name
        if templateName is None:
            print("Template name not found")
            continue
        #add the template to animeBuilder.templates
        animeBuilder.templates[templateName]=template


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/editor')
def editor():
    return render_template('movie_editor.html')


@app.route('/create_summary', methods=['POST'])
def create_summary():
    data = request.get_json()
    #handle extra templates
    extra_templates = data.get("extraTemplates", [])
    handleExtraTemplates(extra_templates)
    story_objects = data.get("storyObjects", {})
    novel_summary = animeBuilder.create_novel_summary(story_objects)
    return jsonify(novel_summary)


@app.route('/create_characters', methods=['POST'])
def create_characters():
    data = request.get_json()
    #handle extra templates
    extra_templates = data.get("extraTemplates", [])
    handleExtraTemplates(extra_templates)
    story_objects = data.get("storyObjects", {})
    novel_summary = data.get("novelSummary", {})
    characters = animeBuilder.create_characters(story_objects, novel_summary)
    return jsonify(characters)


@app.route('/create_chapters', methods=['POST'])
def create_chapters():
    data = request.get_json()
    #handle extra templates
    extra_templates = data.get("extraTemplates", [])
    handleExtraTemplates(extra_templates)
    story_objects = data.get("storyObjects", {})
    novel_summary = data.get("novelSummary", {})
    characters = data.get("characters", {})
    num_chapters = int(data.get("numChapters", 3))
    chapters = animeBuilder.create_chapters(
        story_objects, novel_summary, characters, num_chapters, nTrials=nTrials)
    return jsonify(chapters)


@app.route('/create_scenes', methods=['POST'])
def create_scenes():
    data = request.get_json()
    #handle extra templates
    extra_templates = data.get("extraTemplates", [])
    handleExtraTemplates(extra_templates)
    story_objects = data.get("storyObjects", {})
    novel_summary = data.get("novelSummary", {})
    characters = data.get("characters", {})
    chapters = data.get("chapters", [])
    num_chapters = int(data.get("numChapters", 3))
    num_scenes = int(data.get("numScenes", 3))
    all_scenes = animeBuilder.create_scenes(
        story_objects, novel_summary, characters, chapters, num_chapters, num_scenes, nTrials=nTrials)
    return jsonify(all_scenes)


movies = {}


@app.route('/create_movie', methods=['POST'])
def create_movie():
    data = request.get_json()
    #handle extra templates
    extra_templates = data.get("extraTemplates", [])
    handleExtraTemplates(extra_templates)
    story_objects = data.get("storyObjects", {})
    novel_summary = data.get('novelSummary')
    characters = data.get('characters')
    chapters = data.get('chapters')
    all_scenes = data.get('scenes')
    num_chapters = int(data.get("numChapters", 3))
    num_scenes = int(data.get("numScenes", 3))

    movie_id = getFilename("", "mov")
    # movie_generator = animeBuilder.generate_movie_data(novel_summary, characters, chapters, scenes)
    movie_generator = animeBuilder.generate_movie_data(
        story_objects, novel_summary, characters, chapters, all_scenes, num_chapters, num_scenes,
        aggressive_merging=aggressive_merging,
        portrait_size=portrait_size)
    movies[movie_id] = MovieGeneratorWrapper(movie_generator, movie_id,args.queueSize)
    movies

    # return jsonify({"movie_id": movie_id})
    return jsonify(movie_id)


class MovieGeneratorWrapper:
    def __init__(self, generator, movie_id,queue_size=5):
        self.generator = generator
        self.movie_id = movie_id
        self.current_count = 0
        self.available_count = 0
        self.queue_size = queue_size

        self.active_tasks = 0
        self.futures=[]
        self.queue_index=0 # the index of the last element placed in the queue

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        self._queue = queue.Queue(self.queue_size)
        self._fetch_next_element()

        # Create a table for movie elements
        self.movie_elements_table = db['movie_elements']

        

    def _get_next_element(self):
        try:
            element = next(self.generator)

            # Add movie_id and count to the element
            element["movie_id"] = self.movie_id
            element["count"] = self.available_count

            # Increment the available count
            self.available_count += 1

            # Insert the element as a new record in the database
            self.movie_elements_table.insert(element)

            return element
        except StopIteration:
            print("RETURNING NONE")
            return None

    def _fetch_next_element(self):
        if self.active_tasks >= 1:
            print("ACTIVE TASKS",self.active_tasks)
            return
        
        thisFuture = self._executor.submit(self._fetch_and_enqueue_next_element)
        self.active_tasks += 1
        self.futures.append(thisFuture)
        print("ACTIVE TASKS",self.active_tasks,[f.done() for f in self.futures])

    def _fetch_and_enqueue_next_element(self):
        while self.available_count - self.current_count < self.queue_size:
            print("DOING THE WORK",self.available_count,self.current_count,self.queue_size)
            element = self._get_next_element()
            print("DID THE WORK",element)
            if element is None:
                self._queue.put(None)
                break
            self._queue.put(element)
            self.queue_index+=1

        self.active_tasks -= 1
        print("GOT HERE SOMEHOW",self.available_count,self.current_count,self.queue_size,"active tasks",self.active_tasks)
        

    def get_next_element(self, count=None):


        print("WE ARE HERE",count,self.current_count,self.available_count,self.queue_size)

        if count is not None:
            self.current_count = count

        current_element = self.movie_elements_table.find_one(
            movie_id=self.movie_id, count=self.current_count)
        
        #also pop from the queue if we're in the right zone to do so
        if self.available_count - self.current_count < self.queue_size:
            self._queue.get()


        if current_element is None:
            found_count = -1
            while found_count < self.current_count:
                current_element = self._queue.get()
                if current_element is None:
                    break
                found_count = current_element["count"]

        if current_element is not None:
            if self.available_count - self.current_count < self.queue_size:
                print("FETCHING NEXT ELEMENT")
                self._fetch_next_element()
                

            current_element = {k: v for k,
                               v in current_element.items() if v is not None}

        self.current_count += 1

        if current_element is not None:
            # print("Foo",count,current_element["count"])
            pass

        return current_element


# DatabaseMovieGenerator class
class DatabaseMovieGenerator:
    def __init__(self, movie_id):
        self.movie_id = movie_id
        self.movie_elements_table = db['movie_elements']
        self.current_count = 0

    def get_next_element(self, count=None):
        if count is not None:
            self.current_count = count

        current_element = self.movie_elements_table.find_one(
            movie_id=self.movie_id, count=self.current_count)

        if current_element is None:
            return None

        current_element = {k: v for k,
                           v in current_element.items() if v is not None}

        self.current_count += 1

        return current_element


@app.route('/get_next_element/<string:movie_id>', methods=['GET'])
def get_next_element(movie_id):
    movie_generator = movies.get(movie_id)

    if movie_generator is None:
        # Check if there's at least one element in the movie_elements_table with movie_id
        element_count = db['movie_elements'].count(movie_id=movie_id)

        if element_count == 0:
            return jsonify({"error": "Movie not found"}), 404

        # Create an instance of the DatabaseMovieGenerator class and use it as the movie_generator
        movie_generator = DatabaseMovieGenerator(movie_id)

    count = request.args.get('count', None)
    if count is not None:
        count = int(count)
        print("count", count)

    element = movie_generator.get_next_element(count)
    if element is None:
        return jsonify({"done": "No more elements"}), 200

    return jsonify(element)


@app.route('/get_all_movies', methods=['GET'])
def get_all_movies():
    # Find all movie elements with "debug": "new movie"
    movie_elements = db['movie_elements'].find(debug="new movie")

    # Extract the movie information (title, summary, etc.) from the movie elements
    movies_list = []
    for element in movie_elements:
        movie_info = {
            "movie_id": element["movie_id"],
            "title": element["title"],
            "summary": element["summary"],
        }

        # if we can't find an element where dialgue is "THE END", then don't add to this list (only want complted movies)
        end = db['movie_elements'].find_one(
            movie_id=element["movie_id"],
            debug="movie completed successfully"
        )
        if end is None:
            continue

        print(end)

        # Find the first element with movie_id where the image field is not null
        image_elements = db['movie_elements'].find(
            movie_id=element["movie_id"], image={'notlike': ''})
        image_element = next(iter(image_elements), None)

        # Add the image field to movie_info if an element is found
        if image_element:
            movie_info["image"] = image_element["image"]

        movies_list.append(movie_info)

    return jsonify(movies_list)


@app.route('/movies')
def movie_list():
    return render_template('movie_list.html')


@app.route('/movie/<string:movie_id>', methods=['GET'])
def movie_page(movie_id):
    #get movie title from db
    movie_elements = db['movie_elements'].find_one(movie_id=movie_id, debug="new movie")
    movie_title = movie_elements["title"]
    # Replace 'movie_template.html' with the name of your movie template file
    return render_template('movie_template.html', movie_id=movie_id,movie_title=movie_title)




openai.api_key = os.environ['OPENAI_API_KEY']

if __name__ == '__main__':

    savePath = "./static/samples/"

    parser = argparse.ArgumentParser(
        description="Flask App with model name parameter")
    parser.add_argument('--modelName', type=str,
                        default="andite/anything-v4.0", help="Name of the model")
    parser.add_argument('--promptSuffix', type=str,
                        default=", anime drawing", help="add to image prompt")

    parser.add_argument('--negativePrompt', type=str,
                        default="collage, grayscale, text, watermark, lowres, bad anatomy, bad hands, text, error, missing fingers, cropped, worst quality, low quality, normal quality, jpeg artifacts, watermark, blurry, grayscale, deformed weapons, deformed face, deformed human body",
                        help="negative prompt")

    #this argument is use to specify (more than one) file with template overrides, to it should be a list[string]
    parser.add_argument('--extraTemplatesFile', type=str,
                        nargs='+', help="extra templates file")
                        

    parser.add_argument('--ntrials', type=int, default=5,
                        help='Number of trials (default: 5)')

    parser.add_argument('--numInferenceSteps', type=int, default=15,
                        help='Number of inference steps (default: 15)')

    parser.add_argument('--disable-aggressive-merging',
                        action='store_true', help='Disable aggressive merging')

    parser.add_argument('--img2img', action='store_true',
                        help='upscale with img2img')

    parser.add_argument('--ngrok', action='store_true',
                        help='use ngrok tunnel')

    parser.add_argument('--musicDuration', type=int, default=30,
                        help='Duration of background music loop (default: 30)')
    
    #portraitprompt with a default value of ', anime, face, portrait, headshot, white background'
    parser.add_argument('--portraitPrompt', type=str, default=', anime, face, portrait, headshot, white background',
                        help='portrait prompt')
    
    #language model (with a default value of 'llama')
    parser.add_argument('--languageModel', type=str, default='llama',
                        help='language model')

    # Add the argument for the list of 4 integers with default values
    parser.add_argument(
        "-s",
        "--imageSizes",
        nargs=4,
        type=int,
        default=[512, 512, 1024, 1024],
        help="Four integers representing image sizes (default: 512 512 1024 1024)",
    )


    #add queue size argument
    parser.add_argument('--queueSize', type=int, default=10,
                        help='Queue size (default: 10)')
    
    #add useGPTForChatCompletion argument
    parser.add_argument('--useGPTForChatCompletion', action='store_true',
                        help='Use GPT for chat completion')
    
    #controlnet_diffusion_model
    parser.add_argument('--controlnet_diffusion_model', type=str, default="D:/img/auto1113/stable-diffusion-webui/models/Stable-diffusion/abyssorangemix3AOM3_aom3a1b.safetensors",
                        help='controlnet diffusion model')

    #if --doNotSaveMemory is passed, then saveMemory is set to False
    #otherwise, saveMemory is set to True
    parser.add_argument('--doNotSaveMemory', dest='saveMemory', action='store_false')

    #how much to decimate talking head video
    parser.add_argument('--decimate_talking_head', type=int, default=1,
                        help='Decimate talking head video (default: 1)')
    

    #how many steps to take when generating faces
    parser.add_argument('--numFaceSteps', type=int, default=20,
                        help='Number of face steps (default: 50)')
    

    #audio model
    parser.add_argument('--audioModel', type=str, default="cvssp/audioldm-s-full-v2",
                        help='audio model')
    

    #use_GPT4
    parser.add_argument('--use_GPT4', action='store_true',
                        help='Use GPT4')


    args = parser.parse_args()

    nTrials = args.ntrials

    if args.disable_aggressive_merging:
        aggressive_merging = False
    else:
        aggressive_merging = True

    if args.img2img:
        portrait_size = 256
    else:
        portrait_size = 128

    # database
    db = dataset.connect('sqlite:///movie_elements.db')

    print("USING GPT",args.useGPTForChatCompletion)

    animeBuilder = AnimeBuilder(num_inference_steps=args.numInferenceSteps,
                                #textModel="GPT3",
                                #textModel='EleutherAI/gpt-neo-2.7B',
                                textModel=args.languageModel,
                                use_gpt_for_chat_completion=args.useGPTForChatCompletion,
                                diffusionModel=args.modelName,
                                doImg2Img=args.img2img,
                                negativePrompt=args.negativePrompt,
                                suffix=args.promptSuffix,
                                musicDuration=args.musicDuration,
                                imageSizes=args.imageSizes,
                                portraitPrompt=args.portraitPrompt,
                                controlnet_diffusion_model=args.controlnet_diffusion_model,
                                saveMemory=args.saveMemory,
                                talking_head_decimate=args.decimate_talking_head,
                                face_steps=args.numFaceSteps,
                                audioLDM=args.audioModel,
                                use_GPT4=args.use_GPT4
                                )
    

    print("DOING GEN")
    animeBuilder.doGen("a girl sitting in her bedroom",num_inference_steps=args.numInferenceSteps)
    print("DONE GEN")

    if args.extraTemplatesFile is  not None:

        for extraTemplatesFile in args.extraTemplatesFile:
            print("reading templates from",extraTemplatesFile)
            #does it end in .py or .txt
            if extraTemplatesFile.endswith(".py"):
                with open(extraTemplatesFile, "r") as file:
                    code = file.read()
                    templateOverrides = eval(code)
                    for k, v in templateOverrides.items():
                        print("setting template",k)
                        animeBuilder.templates[k] = v
            elif extraTemplatesFile.endswith(".txt"):
                #split on ===
                #the key is the first line (minus ininitail >)
                #the value is the rest of the file
                with open(extraTemplatesFile, "r") as file:
                    text = file.read()
                    templates = text.split("===")
                    for template in templates:
                        #remove leading and trailing whitespace
                        template = template.strip()
                        lines = template.split("\n")                    
                        key = lines[0][1:].strip()
                        if key=="":
                            print("ERROR\n>>>"+template+"<<<\n")
                            continue
                        value = "\n".join(lines[1:])
                        print("template:",key)
                        animeBuilder.templates[key] = value


    if args.ngrok:
        run_with_ngrok(app, auth_token=os.environ["NGROK_TOKEN"])
        app.run()
    else:
        app.run(debug=True, use_reloader=False)

        # app.run()
