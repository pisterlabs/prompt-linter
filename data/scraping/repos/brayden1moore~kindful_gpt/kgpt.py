import time
import json
import torch
import openai
import pickle
import numpy as np
import sentence_transformers
torch.set_num_threads(1)

from pathlib import Path
THIS_FOLDER = Path(__file__).parent.resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2').to(device)

with open(f"{THIS_FOLDER}/pkl/texts.pkl", "rb") as file:
    texts = pickle.load(file)

with open(f"{THIS_FOLDER}/pkl/embedTensors.pkl", "rb") as file:
    embedTensors = pickle.load(file)


def getApiKey(service):
    '''Reads API key from apikey.json file'''

    with open('apikey.json', 'r') as myfile:
        data=myfile.read()

    obj = json.loads(data)
    return obj[service]


def embedTXT(txt_path):
    '''Split large txt and convert to embeddings using HuggingFace's MiniLM sentence transformer'''

    with open(txt_path, encoding="utf8") as file:
        document = file.readlines()

    strpDocs = [i.strip() for i in document]
    filtDocs = list(filter(lambda v: v!= '' and ' ' in v or '[[split]]' in v, strpDocs))
    cleanDocs = [i.replace('\xa0',' ') + ':' if (i[-1] != '.' and i[-1] != '?' and i[-1] != '!' and i[-1] != ':' and 'For more information, visit:' not in i and '[[split]]' not in i) else i.replace('\xa0',' ') for i in filtDocs] 
    splitDocs = '\n'.join(cleanDocs).split('[[split]]')
    texts = [i[1:] if i[:1]=='\n' else i for i in splitDocs]
    model = sentence_transformers.SentenceTransformer('all-MiniLM-L6-v2').to(device)
    embeddings = model.encode(texts)
    embedTensors = [torch.tensor(i) for i in embeddings]
    
    with open(f"{THIS_FOLDER}/pkl/texts.pkl", "wb") as file:
        pickle.dump(texts, file)

    with open(f"{THIS_FOLDER}/pkl/embedTensors.pkl", "wb") as file:
        pickle.dump(embedTensors, file)


def summarizeGPT(query,document,key):
    '''Summarize with gpt-3.5-turbo.'''

    start = time.time()
    print('Summarizing...')
    if len(key)>0:
        openai.api_key = key
    else:
        openai.api_key = getApiKey('openai')
    promptString = f'Answer this question: "{query}" by summarizing this document: "{document}". Use numbered instructions. Include a link to more information. If you do not know the answer, say "I am sorry, I do not have specific information about this question. Try searching the Kindful Help Center here: https://support.kindful.com/hc/en-us"'
    resp = openai.ChatCompletion.create(model='gpt-3.5-turbo', temperature=0, messages=[{'role': 'system', 'content': promptString}])
    print(time.time() - start)
    return resp['choices'][0]['message']['content']


def summarizeBMO(query,document):
    '''Summarize with my own method. Work in progress.'''

    queryEmbed = model.encode([query])
    queryTensor = torch.tensor(queryEmbed)

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    sims = []

    for i in embedTensors:
        sims.append(cos(i,queryTensor).item())

    simArray = np.array(sims)

    lines = document.split('\n')[:-2]
    forMore = document.split('\n')[-2]

    bestArticle = model.encode(lines)
    bestTensors = [torch.tensor(i) for i in bestArticle]
    lineSims = []

    for i in bestTensors:
        lineSims.append(cos(i,queryTensor).item())

    lineSimArray = np.array(lineSims)
    meanSim = (lineSimArray.mean() + simArray.max()) / 2

    resp = f'Great question! Here is what I found for "{query}."'
    if meanSim < 0.1:
        resp = (f"""I'm sorry, it doesn't look like there is anything related to '{query}' in the Kindful documentation that I was trained on.
    Try visiting the help center: https://support.kindful.com/hc/en-us""")

    else:
        goodLines = []
        for i in lineSimArray:
            if i >= meanSim:
                goodLines.append(np.where(lineSimArray==i)[0][0])

        for i in goodLines:
            line = lines[i]

            if i==goodLines[-1] and ':' in line[-2:]:
                line = line.replace(':','.')

            if line[-1] != ':':
                resp += ('\n- '+ line)
            
            else:
                resp +=  ('\n\n' + line)

        resp +=  ('\n\n' + forMore)
    return resp


def askTXT(query,key=''):
    '''Use KNN to match query with text chunk then summarize.'''

    queryEmbed = model.encode([query])
    queryTensor = torch.tensor(queryEmbed)

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    sims = []

    for i in embedTensors:
        sims.append(cos(i,queryTensor).item())

    simArray = np.array(sims)
    document = texts[simArray.argmax()]


    if len(key)>0:
        resp = summarizeGPT(query,document,key)
    else:
        resp = summarizeBMO(query,document)
    return resp