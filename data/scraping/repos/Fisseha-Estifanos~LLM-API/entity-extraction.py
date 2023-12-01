"""
An experiment script to understand the basics of working with the co:here API
Using the extraction functionality
"""

import os
import sys
import cohere
sys.path.insert(1, 'scripts/')
from dotenv import load_dotenv

# load your environment
load_dotenv()
# use your own api key here
cohere_api_key = os.getenv('cohere_api_key')
co = cohere.Client(cohere_api_key)

response = co.generate(
  model='large',
  prompt='Passage: Is Wordle getting tougher to solve? Players seem to be convinced that the game has gotten harder in recent weeks ever since The New York Times bought it from developer Josh Wardle in late January. The Times has come forward and shared that this likely isn’t the case. That said, the NYT did mess with the back end code a bit, removing some offensive and sexual language, as well as some obscure words There is a viral thread claiming that a confirmation bias was at play. One Twitter user went so far as to claim the game has gone to “the dusty section of the dictionary” to find its latest words.\n\nTLDR: Wordle has not gotten more difficult to solve.\n--\nPassage: ArtificialIvan, a seven-year-old, London-based payment and expense management software company, has raised $190 million in Series C funding led by ARG Global, with participation from D9 Capital Group and Boulder Capital. Earlier backers also joined the round, including Hilton Group, Roxanne Capital, Paved Roads Ventures, Brook Partners, and Plato Capital.\n\nTLDR: ArtificialIvan has raised $190 million in Series C funding.\n--\nPassage: The National Weather Service announced Tuesday that a freeze warning is in effect for the Bay Area, with freezing temperatures expected in these areas overnight. Temperatures could fall into the mid-20s to low 30s in some areas. In anticipation of the hard freeze, the weather service warns people to take action now.\n\nTLDR:',
  max_tokens=50,
  temperature=0.8,
  k=0,
  p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop_sequences=["--"],
  return_likelihoods='NONE')
print('Prediction: {}'.format(response.generations[0].text))
