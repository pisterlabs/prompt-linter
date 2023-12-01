import json
import os
import re
import subprocess

import pandas as pd
import numpy as np
import spacy
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

from Processing import Processor

subprocess.call(['./pdf_converter'])

path = 'txt_files/'

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    files = [os.path.join(r, file) for file in f if '.txt' in file]

processor = Processor(spacy.load('en_core_web_lg'))


storage = {}

for f in files:
    file = f[len(path):-4]
    storage[file] = processor.process(f, export=False)


with open('documents' + '.json', 'w') as outfile:
    json.dump(storage, outfile)
