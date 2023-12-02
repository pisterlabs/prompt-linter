"""Prompt character descriptions for attribute types

Input
  - character descriptions json file
      - path = mica-character-attribute-extraction/attribute-types/character_descriptions.json
      - contains <imdb id, character name, character description> tuples
      - created by sample_character_descriptions.py
#
Output
  - character attribute types json file
      - path = mica-character-attribute-extraction/attribute-types/character_attribute_types.json
      - contains list of GPT completions

Parameters
  - n
       number of character descriptions from the character descriptions json file to prompt
"""

import os
import openai
import json
import tenacity
import tqdm
import random

from absl import flags
from absl import app

FLAGS = flags.FLAGS
data_dir = os.path.join(os.getenv("DATA_DIR"), "mica-character-attribute-extraction")
input_file = os.path.join(data_dir, "attribute-types/character_descriptions.json")
output_file = os.path.join(data_dir, "attribute-types/character_attribute_types.json")

flags.DEFINE_integer("n", default=50, help="number of character descriptions to prompt")

template = """List the attribute types of <CHARACTER> described in <PASSAGE> as a comma-separated list. If the <PASSAGE> does not describe any attribute of <CHARACTER>, answer as NONE.

PASSAGE: The bell has just rung, and Mike Damone comes out of Youth and Law class. He has an absorbed, driven look on his face. He walks past the rows of lockers, and doesn't even notice as he passes Stacy Hamilton standing by her locker. She smiles, grabs his arm affectionately.
CHARACTER: Stacy Hamilton
ATTRIBUTE-TYPES: NONE

PASSAGE: Sharon Pogue is driving the car. It is two years since we saw her at the accident site. Her partner, ROBBY LEWIS, sips coffee and keeps one eye on the CAD monitor which lists all area police calls. She slows behind a car that is crawling along, an old 60s car, driven by a young man and woman who sit very close on the bench seat. The car's ENGINE is MISSING and smoking.
CHARACTER: Sharon Pogue
ATTRIBUTE-TYPES: NONE

PASSAGE: DETECTIVE SERGEANT ALONZO HARRIS, in black shirt, black leather jacket. And just enough platinum and diamonds to look like somebody. He reads the paper in a booth. The gun leather-tough LAPD vet is a hands-on, blue-collar cop who can kick your ass with a look.
CHARACTER: Alonzo Harris
ATTRIBUTE-TYPES: Profession, Attire, Attitude

PASSAGE: The SPORTS COMMENTATOR is at the airport and about to interview the heavyweight champion of the world, APOLLO CREED. Creed is twenty-eight years old. He is a tall, smooth-muscled black man with barely a scar on his light coffee-colored face...
CHARACTER: Apollo Creed
ATTRIBUTE-TYPES: Age, Race, Appearance, Profession, Accomplishment

PASSAGE: A woman's face BANKS INTO SHOT, her head resting against grimy wallpaper. She is tense, sweaty, wide-eyed with concentration. This is CLARICE STARLING, mid-20's, trim, very pretty. She wears Kevlar body armor over a navy windbreaker, khaki pants. Her thick hair is piled under a navy baseball cap. A revolver, clutched in her right hand, hovers by her ear.
CHARACTER: Clarice Starling
ATTRIBUTE-TYPES: Age, Appearance, Hair Type, Attire, Possession, Emotion, Posture

PASSAGE: The tremendous heat of two huge twin suns settle on a lone figure, Luke Skywalker, a farm boy with heroic aspirations who looks much younger than his eighteen years. His shaggy hair and baggy tunic give him the air of a simple but lovable lad with a prize-winning smile. The lovely young girl huddles in a small alcove as the stormtroopers search through the ship. She is Princess Leia Organa, a member of the Alderaan Senate. Han is a tough, roguish starpilot about thirty years old. A mercenary on a starship, he is simple, sentimental, and cocksure.
CHARACTER: Princess Leia Organa
ATTRIBUTE-TYPES: Age, Profession

PASSAGE: MICHAEL CORLEONE, dressed in the uniform of a Marine Captain, leads KAY ADAMS through the wedding crowd, occasionally stopped and greeted by friends of the family.
CHARACTER: Michael Corleone
ATTRIBUTE-TYPES: Attire

PASSAGE: HE IS MELVIN PURVIS (30) cleancut and handsome. A square jaw. Chester Gould modelled Dick Tracy's profile after Purvis. He's not big, but he's tenacious. He's an incarnation of the social elite of his time: white, Southern patrician and a Christian gentleman. With Purvis are Special Agents WARREN BARTON (31) and Purvis's friend CARTER DAUM (29) They have a harder time in the steep woods. They're chasing someone. They are guided by East Liverpool police chief
CHARACTER: Melvin Purvis
ATTRIBUTE-TYPES: Age, Appearance, Ethnicity, Social Status, Religion, Qualities

PASSAGE: A TRANSPORT PLANE has just landed. The cargo doors are open and a tricky unloading operation is under way. Robin Cavendish is being hoisted out of the plane, strapped to a stretcher. A big battery powered respirator is being unloaded alongside him. An ambulance waits to receive him. Medics, RAF personnel and ground crew are gathered round.
CHARACTER: Robin Cavendish
ATTRIBUTE-TYPES: Health Status

"""

@tenacity.retry(wait=tenacity.wait_random_exponential(min=1, max=60), stop=tenacity.stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

def prompt_sample(passage, character):
    prompt = f"{template}PASSAGE: {passage}\nCHARACTER: {character}\nATTRIBUTE-TYPES:"
    try:
        response = completion_with_backoff(
            model="text-davinci-003", 
            prompt=prompt,
            temperature=0.7,
            max_tokens=1024,
            )
        return response.to_dict()
    except Exception:
        return

def prompt_attribute_types(_):
    n = FLAGS.n
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organization = "org-xPjDKPQ58le6x8A7CE13e8O6"

    with open(input_file) as fr:
        descs = json.load(fr)
    sampled_descs = random.sample(descs, n) if n < len(descs) else descs

    completions = []
    for desc in tqdm.tqdm(sampled_descs, unit="desc"):
        output = prompt_sample(desc["desc"], desc["character"])
        if output is not None:
            output.update({"desc": desc["desc"], "character": desc["character"]})
            completions.append(output)
    with open(output_file, "w") as fw:
        json.dump(completions, fw, indent=2)

if __name__ == '__main__':
    app.run(prompt_attribute_types)