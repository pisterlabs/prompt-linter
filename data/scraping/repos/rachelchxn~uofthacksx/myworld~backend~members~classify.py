import cohere
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

# have to put api key in .env
co = cohere.Client(os.getenv('API_KEY_COHERE'))
from cohere.classify import Example

a = 'Core Concept'
b = 'Point'
c = 'Example'
d = 'Definition'
e = 'Remove'

examples=[
  Example("Our body is made up of trillions of cells that all require energy made in the mitochondria to function. ", a),
  Example("Mitochondria use the oxygen you breathe to make energy for you.", a),
  Example("Taking a look at the next topic", a),
  Example("We will first discuss how to design an interface", a),
  Example("The human body needs energy to move, eat, breathe; chemical energy which is produced in the mitochondria and is called ATP.", a),

  Example("One of the features of an interface is the navigation bar", b),
  Example("Scientists think that mitochondria evolved from bacteria and used to be a separate single celledorganism", b),
  Example("ATP is released by mitochondria, so cells can use it. Mitochondria consists of two membranes an outer membrane separating it from the cytosol and an inner membrane surrounding the so called matrix", b),
  Example("Cells that require more energy contain more mitochondria. Muscle cells use a lot of ATP for muscle contraction to produce the amount of ATP that's required. There are more mitochondria found in muscle cells.", b),
  
  Example("For example, the mitochodrion uses energy from ATP", c),
  Example("An example of a graph of a non-function is shown here", c),
  Example("John Smith is an example of an important theorist", c),
  Example("Take this cube for example. The middle square on the top appears to be a shade of brown, while the one on the side looks much more orange. But in actuality, they are both the exact same colour.", c),
  
  Example("Electrons are stable subatomic particle with a charge of negative electricity, found in all atoms and acting as the primary carrier of electricity in solids.",d),
  Example("The mitochondria is an organelle found in large numbers in most cells.",d),
  Example("DNA is a self-replicating material that is present in nearly all living organisms as the main constituent of chromosomes. It is the carrier of genetic information",d),
  Example("Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",d),
  Example("Science is the study of the universe around us and within us, natural phenomena, and solutions to problems. ",d),
  Example("The scientific method is a process for gathering data and processing information",d),
  Example("ATP is the energy currency of the human body.",d),

  Example("Want to learn more? Subscribe to our channel",e),
  Example("Visit our website to learn more",e),
  Example("Please subscribe and like this video",e),

]


def extract(self, example):
    extraction = co.generate(
        model='large',
        prompt=self.make_prompt(example),
        max_tokens=10,
        temperature=0.1,
        stop_sequences=["\n"])
    return(extraction.generations[0].text[:-1])


def classifyNotes(input):
    response = co.classify(
        model='large',  
        inputs=input,  
        examples=examples
        )
    preds = []
    for i in response.classifications:
        preds.append(i.prediction)
    return preds



def summarize(input):
    prompt = f"""Passage: Alright, so the world has seemingly become utterly divided on this dress. What colours do you see? On one side we have team Black and Blue - on the other, team White and Gold
    
    TLDR: People disagree on the colour of this dress; some see Black and Blue, others see White and Gold. 
    --
    Passage: Our body is made up of trillions of cells. They all require energy to function. This energy is created within our cells in the mitochondria
    
    TLDR: Our body is made up of trillions of cells that all require energy made in the mitochondria to function. 
    --
    Passage: This continuous pumping creates a proton gradient where the positively charged protons are attracted to the more negative matrix. When the protons reenter the matrix through the ATP synthase protein complex they catalyze the production of ATP
    
    TLDR: Continuous pumping creates a proton gradient where protons are attracted to the more negative matrix. Protons catalyze ATP production when they reenter the matrix. 
    --
    Passage: Research suggests that the protein essential for the formation of chicken eggs, called ov 17, is only found in chicken ovaries. Without it, the chicken eggshell could not be formed. So without a chicken, you technically can't get a chicken egg
    
    TLDR: Research shows that ov 17, only found in chicken ovaries, is essential to the production of the chicken eggshell. 
    --
    Passage: Is Wordle getting tougher to solve? Players seem to be convinced that the game has gotten harder in recent weeks ever since The New York Times bought it from developer Josh Wardle in late January. The Times has come forward and shared that this likely isn't the case. That said, the NYT did mess with the back end code a bit, removing some offensive and sexual language, as well as some obscure words There is a viral thread claiming that a confirmation bias was at play. One Twitter user went so far as to claim the game has gone to "the dusty section of the dictionary" to find its latest words
    
    TLDR: Wordle has not gotten more difficult to solve. 
    --
    Passage: The human body needs energy to move, eat, breathe. Chemical energy which is produced in the mitochondria of the cell and is called ATP.

    TLDR: The human body needs energy; chemical energy which is produced in mitochondria and is called ATP.
    --
    Passage: {input}
    
    TLDR: """

    response = co.generate( 
        model='xlarge', 
        prompt = prompt,
        max_tokens=80, 
        temperature=1,
        stop_sequences=['--'])

    return response.generations[0].text
