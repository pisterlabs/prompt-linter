import cohere
import numpy as np
import re
import pandas as pd
from cohere.classify import Example
import csv
import json

# getting cohere key

with open("cohere-key.json") as json_file:
  data = json.load(json_file)
  co = cohere.Client(data["cohere-key"])

# examples to train genre classifier

examples=[   
  Example("Something thrilling with fights, chases, and explosions. I like stunts. Violence and gore.", "action"), 
  Example("I'm in the mood for a treasure hunt or quest, maybe with pirates and epic heros and villains.", "adventure"), 
  Example("i watch cartoons and anime. My children love wholesome educational artwork", "animation"), 
  Example("Something inspiring. About life. Biopic. True story and real life.", "biography"), 
  Example("Something hilarious and crude. So so funny and lighthearted. Filled with utter joy. Satirical and witty", "comedy"), 
  Example("Mysterious, sherlock holmes. Murder mystery. Tragedy, dark. A whodunnit, similar to Agatha Christie", "crime"), 
  Example("A true story or similar to real life. Sad and heavy. About love, life, and loss.", "drama"), 
  Example("Something I can watch with my kids, age appropriate. Funny, animated. Disney perhaps.", "family"), 
  Example("I like dragons and ogres and folklore. The supernatural like ghosts and zombies and witches. Imagination, unreal.", "fantasy"), 
  Example("Something that has happened, low key, low budget, artistic", "history"), 
  Example("Supernatural, psychological, thrilling, on the edge of my seat, terrifying. Apocalyptic", "horror"), 
  Example("Fun. Dance numbers, jazzy. singing", "music"), 
  Example("Dancing and singing", "musical"), 
  Example("Agatha Christie, Sherlock Holmes. Piecing together clues. Plot twist. Murder", "mystery"), 
  Example("Predictable hallmark, sentimental. Emotional tragedy, personal journey. Care Hope.  Queer, gay.", "romance"),
  Example("Time travel teleportation, telepathy and aliens. Star wars. Fantastic, dystopian.", "sci-fi"), 
  Example("Baseball, football, olympics, competition. Underdog protagonist.", "sport"), 
  Example("I want to be sweating at the edge of my seat. Anxious and uncertain and surprise.", "thriller"), 
  Example("Violence destruction mortality. Life and death and the moments in between. Uncertainty, legality and ethics.", "war"), 
  Example("America, small towns, saloons. Outlaws and bandits. Horse, cowboy.", "western"),
  Example("Heart pounding and charged. Like a superhero movie.", "action"), 
  Example("An exciting story about new experiences and discoveries. Voyages and travelling.", "adventure"), 
  Example("Disney movies and princesses and adventures. Talking animals and children and innocence.", "animation"), 
  Example("Something true and inspiring. About real people making a difference. History and comtemporary.", "biography"), 
  Example("Amusing and light. Lovable characters getting into shenanigans.", "comedy"), 
  Example("Something dark and violent, about a criminal mastermind or drug cartel or gang ousting the police", "crime"), 
  Example("The opeak of human experience and emotion. Nuanced, moving, beautiful with big themes", "drama"), 
  Example("Something imaginative with a sense of humour and charm. Cartoon, simplistic, sentimental and nostalgic.", "family"), 
  Example("I'm thinking escapism. Epic myth and wonder.", "fantasy"), 
  Example("Serious, intimate, real, epic, war, science", "history"), 
  Example("I want to scream, a plot twist, element of surprise, doom, fear, death, pain", "horror"), 
  Example("Score, soundtrack", "music"), 
  Example("Theatre like. Cheesy, feel good", "musical"), 
  Example("Investigation and deduction. Suspenseful, insightful. Thinking", "mystery"), 
  Example("Love cheesy, sexy. Banter, wit, hot beautiful leads. Chemistry", "romance"),
  Example("Technology in a fictional world. Insightful and allegorical. Robot, alien life form and interstellar.", "sci-fi"), 
  Example("Loss, victory, epic highs and lows. struggle and triumph", "sport"), 
  Example("Melodramatic and violent. Unexpected twists and high stakes", "thriller"), 
  Example("Societal commentary, dark but hopeful. The human condition at it's lowest. Thrilling and action", "war"), 
  Example("Individualism, justice, freedom, struggle, poverty. Community town, courage and pride.", "western")
]

# genre classifier

def get_genre_prediction(input):
  r = [input]
  try:
    response = co.classify(
    model='large',
    inputs=r,
    examples=examples,
    )
    return(response.classifications[0].prediction)
  except:
    return("no")

# get cosine similarity between input text and tag

def get_similarity(input, movie):
  phrases = [input,movie]
  try:
    [r,x] = co.embed(phrases).embeddings
    def calculate_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return calculate_similarity(r,x)
  except:
    return(0)

# clean text by removing filler words and punctuation

def clean_text(
  string: str, 
  punctuations=r'''!()-[]{};:'"\,<>./?@#$%^&*_~''',
  stop_words=['the', 'a', 'and', 'is', 'be', 'will', 'should','my']) -> str:
 
  for x in string.lower(): 
      if x in punctuations: 
          string = string.replace(x, "") 

  string = string.lower()
  string = ' '.join([word for word in string.split() if word not in stop_words])
  string = re.sub(r'\s+', ' ', string).strip()
  return string

# get tags using embedder

def get_tags(prompt):
      taglist = ['absurd', 'action', 'adult comedy', 'allegory', 'alternate history', 'alternate reality', 'anti war', 'atmospheric', 'autobiographical', 'avant garde', 'blaxploitation', 'bleak', 'boring', 'brainwashing', 'claustrophobic', 'clever', 'comedy', 'comic', 'cruelty', 'cult', 'cute', 'dark', 'depressing', 'dramatic', 'entertaining', 'fantasy', 'feel-good', 'flashback', 'good versus evil', 'gothic', 'haunting', 'historical', 'historical fiction', 'home movie', 'horror', 'humor', 'insanity', 'inspiring', 'intrigue', 'magical realism', 'melodrama', 'murder', 'mystery', 'neo noir', 'non fiction', 'paranormal', 'philosophical', 'plot twist', 'pornographic', 'prank', 'psychedelic', 'psychological', 'queer', 'realism', 'revenge', 'romantic', 'sadist', 'satire', 'sci-fi', 'sentimental', 'storytelling', 'stupid', 'suicidal', 'suspenseful', 'thought-provoking', 'tragedy', 'violence', 'western', 'whimsical']
      prompt = clean_text(prompt)
      sims = []
      for tag in taglist:
            sims.append((get_similarity(tag,prompt),tag))

      sims.sort(key = lambda x: x[0], reverse=True)
      output = ""
      
      return([sims[0],sims[1],sims[2],sims[3]])