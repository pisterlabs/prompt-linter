# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import json
import os
from glob import glob
from re import L

import openai
from cog import BasePredictor, Input, Path
from dotenv import load_dotenv
from googletrans import Translator

load_dotenv()
from glob import glob
from time import sleep

from pypollsdk import run_model

openai.api_key = os.getenv("OPENAI_API_KEY")



gpt_prompt = """Prompt Design

You can borrow some photographic prompt terminology (especially for framing) to apply to illustrations: e.g: 'close-up.' If you are generating mockups of 3D art, you can also define how that piece is photographed!

Adjectives can easily influence multiple factors, e.g: 'art deco' will influence the illustration style, but also the clothing and materials of the subject, unless otherwise defined. Years, decades and eras, like '1924' or 'late-90s', can also have this effect.

Even superficially specific prompts have more 'general' effects. For instance, defining a camera or lens ('Sigma 75mm') doesn't just 'create that specific look', it more broadly alludes to 'the kind of photo where the lens/camera appears in the description', which tend to be professional and hence higher-quality.

Detailed prompts are great if you know exactly what you're looking for and are trying to get a specific effect. But there is nothing wrong with being vague, and seeing what happens!


Examples of how to make prompts

prompt: universe in a jar
pimped: Intricate illustration of a universe in a jar. intricately exquisitely detailed. holographic. beautiful. colourful. 3 d vray render, artstation, deviantart, pinterest

prompt: Fox with a cloak
pimped: A fox wearing a cloak. angled view, cinematic, mid-day, professional photography,8k, photo realistic, 50mm lens , Pixar, Dreamworks, Alex Ross, Tim Burton, Nickelodeon, Alex Ross, Character design, breath of wild, 3d render
 
prompt: Jellfish phoenix goddess being
pimped: Goddess portrait. jellyfish phoenix head. intricate artwork by tooth wu and wlop and beeple. octane render, trending on artstation, greg rutkowski very coherent symmetrical artwork. cinematic, hyper realism, high detail, octane render, 8k

prompt: humanoid robot head
pimped: Cute humanoid robot, crystal material, portrait bust, symmetry, faded colors, aztec theme, cypherpunk background, tim hildebrandt, wayne barlowe, bruce pennington, donato giancola, larry elmore, masterpiece, trending on artstation, featured on pixiv, cinematic composition, beautiful lighting, hyper detailed, 8 k, unreal engine 5

prompt: Portrait of Hrry Potter
pimped: A close up portrait of harry potter as a young man, art station, highly detailed, focused gaze, concept art, sharp focus, illustration in pen and ink, wide angle, by kentaro miura

prompt: Artichoke head monster
pimped: Humanoid figure with an artichoke head, highly detailed, digital art, sharp focus, trending on art station, monster, glowing eyes, anime art style

prompt: A boy and  with black short hair in a rowing boat
pimped: A girl and a boy with long flowing auburn hair sitting together on the rowboat. boy has black short hair, boy has black short hair. atmospheric lighting, long shot, romantic, boy and girl are the focus, trees, river. details, sharp focus, illustration, by jordan grimmer and greg rutkowski, trending artstation, pixiv, digital art

prompt: Diagram of abird
pimped: A patent drawing of mechanical bird robots, birds of all kinds, infographic, intricate drawing, 1960s advertising, watercolour, ink drawing, patent drawing, wireframe, technical and mechanical details, descriptions, explosion drawing, cad
    
prompt: Coffe grinder
pimped: drawn coffee grinder in the style of thomas edison, patent filing, detailed, hd

prompt: Colorful digital indigenous toy
pimped: A colorfull portrait of a selk'nam with a capirote, plastic toy, art toy, by hikari shimoda, al feldstein, mark ryden, yayoi kusama, lisa frank, garbage pail kids, award winning photo, iridiscense, houdini algorithmic generative render, dramatic lighting, volumetric light, accurate and detailed, sharp focus, octane render 8k, zoom out

prompt: Camera still life
pimped: Slr camera advertisment, still life, 1 9 7 0 s japan shouwa advertisement, print, nostalgic

prompt: digital fantasy world
pimped: epic cinematic digital wallpaper of a fantasy world, isometric 3 d, artwork by james gilleard, 

prompt: {}
pimped:""".format


def report_status(**kwargs):
    status = json.dumps(kwargs)
    print(f"pollen_status: {status}")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.translator= Translator()

    def predict(
        self,
        prompt: str,
    ) -> Path:
        """Run a single prediction on the model"""

        # JSON encode {title: "Pimping your prompt", payload: prompt }
        report_status(title="Translating", payload=prompt)
        prompt = self.translator.translate(prompt.strip()).text 
        report_status(title="Pimping prompt", payload=prompt)
        response = openai.Completion.create(
            model="text-curie-001",
            prompt=gpt_prompt(prompt),
            max_tokens=200,
            temperature=0.82,
            n=3,
            stop=["prompt:", "\n"]
        ).choices
        prompts_list = [i.text.strip().replace("pimped:", "") for i in response]
        prompts_list = [prompt for prompt in prompts_list if len(prompt) > 0]
        print("got prompts", len(prompts_list))
        prompts = "\n".join(prompts_list)
        report_status(title="Generating images", payload=prompts)

        print("prompts:", prompts)
        run_model("614871946825.dkr.ecr.us-east-1.amazonaws.com/pollinations/stable-diffusion-private", {"prompts": prompts, "num_frames_per_prompt": 1, "diffusion_steps": -50, "prompt_scale": 15}, "/outputs/stable-diffusion")
        report_status(title="Display", payload=prompts)

        for i, image in enumerate(glob("/outputs/stable-diffusion/*.png")):
            # move image to /outputs
            os.system(f"cp -v {image} /outputs")
            
            # create .txt file with same name as image
            prompt_filename =  os.path.splitext(os.path.basename(image))[0]+".txt"
            with open("/outputs/"+prompt_filename, "w") as prompt_file:
                prompt_file.write(prompts_list[i])
                print("wrote",prompts_list[i],"to file", prompt_filename)
        
        sleep(5)
        return

