###### Importation des librairies
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from detoxify import Detoxify
import string
from tqdm import tqdm
import nltk
import keras
import openai
from eldonback.secret import SECRET_KEY_OPEN_AI
nltk.download("punkt")
from nltk.stem import PorterStemmer
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


### Prédiction de la nature du message
def algorithm_gpt3(x_to_classify):
    openai.api_key = SECRET_KEY_OPEN_AI
    
    response = openai.Moderation.create(input=x_to_classify)
    response = response['results'][0]['flagged']
    try:
      y_predicted = 1 if response==True else 0
    except Exception as e:
      print(str(e))
    
    return y_predicted
