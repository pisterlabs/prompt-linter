from cleantext import clean
import string
from nltk.tokenize import SpaceTokenizer
import nltk
import cohere
from cohere import CohereError
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from docx import Document
import pandas as pd
import numpy as np
from numpy.linalg import norm
import ssl
from dotenv import load_dotenv
import plotly_express as px
from scrape_onet import get_onet_code

# SSL CERTIFICATE FIX
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# DOWNLOAD NLTK DATA IF NOT ALREADY DOWNLOADED
if os.path.isdir('nltk_data')==False:
    nltk.download('stopwords', quiet=True)

# LOAD ENVIRONMENT VARIABLES
load_dotenv()

# LOAD COHERE EMBEDDINGS:
simdat = pd.read_csv('static/embeddings/cohere_embeddings.csv')
coheredat = pd.read_csv('static/cohere_tSNE_dat.csv')

# LOAD FINE-TUNED MODEL 
# (see https://huggingface.co/celise88/distilbert-base-uncased-finetuned-binary-classifier)
model = AutoModelForSequenceClassification.from_pretrained('static/model_shards', low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained('static/tokenizer_shards', low_cpu_mem_usage=True)
classifier = pipeline('text-classification', model = model, tokenizer = tokenizer)

# UTILITY FUNCTIONS
async def neighborhoods(jobtitle=None):
    def format_title(logo, title, subtitle, title_font_size = 28, subtitle_font_size=14):
        logo = f'<a href="/" target="_self">{logo}</a>'
        subtitle = f'<span style="font-size: {subtitle_font_size}px;">{subtitle}</span>'
        title = f'<span style="font-size: {title_font_size}px;">{title}</span>'
        return f'{logo}{title}<br>{subtitle}'
    fig = px.scatter(coheredat, x = 'longitude', y = 'latitude', color = 'Category', hover_data = ['Category', 'Title'], 
        title=format_title("Pathfinder", "     Job Neighborhoods: Explore the Map!", "(Generated using Co-here AI's LLM & ONET's Task Statements)"))
    fig['layout'].update(height=1000, width=1500, font=dict(family='Courier New, monospace', color='black'))
    fig.write_html('templates/job_neighborhoods.html')

def get_resume(resume):
    path = f"static/{resume.filename}"
    with open(path, 'wb') as buffer:
        buffer.write(resume.file.read())
    file = Document(path)
    text = []
    for para in file.paragraphs:
        text.append(para.text)
    resume = "\n".join(text)
    return resume

def coSkillEmbed(text):
    try:
        co = cohere.Client(os.getenv("COHERE_TOKEN"))
        response = co.embed(
            model='large',
            texts=[text])
        return response.embeddings
    except CohereError as e:
        return e

async def sim_result_loop(skilltext):
    if type(skilltext) == str:
        skills = skilltext
    if type(skilltext) == dict:
        skills = [key for key, value in skilltext.items() if value == "Skill"]
        skills = str(skills).replace("'", "").replace(",", "")
    embeds = coSkillEmbed(skills)
    def cosine(A, B):
        return np.dot(A,B)/(norm(A)*norm(B))
    def format_sim(sim):
        return "{:0.2f}".format(sim)
    simResults = []
    [simResults.append(cosine(np.array(embeds), np.array(simdat.iloc[i,1:]))) for i in range(len(simdat))]
    simResults = pd.DataFrame(simResults)
    simResults['JobTitle'] = simdat['Title']
    simResults = simResults.iloc[:,[1,0]]
    simResults.columns = ['JobTitle', 'Similarity']
    simResults = simResults.sort_values(by = "Similarity", ascending = False)
    simResults = simResults.iloc[:13,:]
    simResults = simResults.iloc[1:,:]
    simResults.reset_index(drop=True, inplace=True)
    if simResults['Similarity'].min() < 0.5:
        simResults['Similarity'] = simResults['Similarity'] + (0.5 - simResults['Similarity'].min())
        if simResults['Similarity'].max() > 1.0:
            simResults['Similarity'] = simResults['Similarity'] - (simResults['Similarity'].max() - 1.0)
    for x in range(len(simResults)):
        simResults.iloc[x,1] = format_sim(simResults.iloc[x,1])
    return simResults, embeds

async def skillNER(resume):
    def clean_my_text(text):
        clean_text = ' '.join(text.splitlines())
        clean_text = clean_text.replace('-', " ").replace("/"," ")
        clean_text = clean(clean_text.translate(str.maketrans('', '', string.punctuation)))
        return clean_text
    resume = clean_my_text(resume)
    stops = set(nltk.corpus.stopwords.words('english'))
    stops = stops.union({'eg', 'ie', 'etc', 'experience', 'experiences', 'experienced', 'experiencing', 'knowledge', 
    'ability', 'abilities', 'skill', 'skills', 'skilled', 'including', 'includes', 'included', 'include'
    'education', 'follow', 'following', 'follows', 'followed', 'make', 'made', 'makes', 'making', 'maker',
    'available', 'large', 'larger', 'largescale', 'client', 'clients', 'responsible', 'x', 'many', 'team', 'teams', 
    'concern', 'concerned', 'concerning', 'concerns', 'space', 'spaces', 'spaced'})
    resume = [word for word in SpaceTokenizer().tokenize(resume) if word not in stops]
    resume = [word for word in resume if ")" not in word]
    resume = [word for word in resume if "(" not in word]
    skills = {}
    [skills.update({word : "Skill"}) if classifier(word)[0]['label'] == 'LABEL_1' else skills.update({word: "Not Skill"}) for word in resume]
    return skills

def get_links(simResults):
    links = []
    titles = simResults["JobTitle"]
    [links.append("https://www.onetonline.org/link/summary/" + get_onet_code(title)) for title in titles]
    return links

def sim_result_loop_jobFinder(resume):
    embeds = coSkillEmbed(resume)
    def cosine(A, B):
        return np.dot(A,B)/(norm(A)*norm(B))
    def format_sim(sim):
        return "{:0.2f}".format(sim)
    jobdat = pd.read_csv('static/jd_embeddings.csv')
    jobembeds = jobdat.iloc[:,5:].dropna()
    simResults = []
    [simResults.append(cosine(np.array(embeds), np.array(jobembeds.iloc[i,:]))) for i in range(len(jobembeds))]
    simResults = pd.DataFrame(simResults)
    simResults['job_id'] = jobdat['id']
    simResults['emp_email'] = jobdat['email']
    simResults = simResults.iloc[:,[1,2,0]]
    simResults.columns = ['job_id', 'employer_email', 'similarity']
    simResults = simResults.sort_values(by = "similarity", ascending = False)
    simResults.reset_index(drop=True, inplace=True)
    for x in range(len(simResults)):
        simResults.iloc[x,2] = format_sim(simResults.iloc[x,2])
    return simResults

def sim_result_loop_candFinder(jobdesc):
    embeds = coSkillEmbed(jobdesc)
    def cosine(A, B):
        return np.dot(A,B)/(norm(A)*norm(B))
    def format_sim(sim):
        return "{:0.2f}".format(sim)
    canddat = pd.read_csv('static/res_embeddings.csv')
    candembeds = canddat.iloc[:,5:].dropna()
    simResults = []
    [simResults.append(cosine(np.array(embeds), np.array(candembeds.iloc[i,:]))) for i in range(len(candembeds))]
    simResults = pd.DataFrame(simResults)
    simResults['cand_id'] = canddat['id']
    simResults['cand_email'] = canddat['email']
    simResults = simResults.iloc[:,[1,2,0]]
    simResults.columns = ['candidate_id', 'candidate_email', 'similarity']
    simResults = simResults.sort_values(by = "similarity", ascending = False)
    simResults.reset_index(drop=True, inplace=True)
    for x in range(len(simResults)):
        simResults.iloc[x,2] = format_sim(simResults.iloc[x,2])
    return simResults