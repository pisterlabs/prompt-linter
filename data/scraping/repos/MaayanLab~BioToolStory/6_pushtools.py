# Tables: resources <program,uuid> --> libraries <journal,ISSN> --> signatures <tools,pmid>

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import requests
import urllib.request
import json
import datetime
import http.client
import sys
import shutil
import os
import time
from datetime import datetime
import re
import pandas as pd
from Bio import Entrez
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
import uuid
from crossref.restful import Works
from crossref.restful import Journals
import numpy as np
import requests as req
import itertools
from bs4 import BeautifulSoup
import lxml
import collections
import ast
from datetime import datetime
from itertools import chain
import math
from pandas.io.json import json_normalize
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from os import listdir
from os.path import isfile, join

nlp = spacy.load('en', disable=['parser', 'ner'])

load_dotenv(verbose=True)
PTH = os.environ.get('PTH_A')

API_url = os.environ.get('API_URL')
username = os.getenv("USERNAME")
password = os.getenv("PASSWORD")
credentials = HTTPBasicAuth(username, password)

Entrez.email = os.environ.get('EMAIL')
API_KEY = os.environ.get('API_KEY')

# middleman credentials
username_middle = os.getenv("USERNAME_middle")
password_middle = os.getenv("PASSWORD_middle")
auth_middle = HTTPBasicAuth(username_middle, password_middle)

start = str(sys.argv[1])
end = str(sys.argv[2])
s = start.replace("/","")
en = end.replace("/","")

all_tools = []
dry_run = int(sys.argv[3])
#==================================================  Database ===================================================================

# delete a single journal
# res = requests.delete(API_url%('libraries','b788c70e-79af-4acc-8ccf-0816c7bb59e3', auth=credentials)

# delete a single item
def delete_data(data,schema):
  res = requests.delete(API_url%(schema,data["id"]), auth=credentials)
  if not res.ok:
    raise Exception(res.text)


# delete all * from Database
def del_all_tools(schema):
  res = requests.get(API_url%(schema,""))
  tools_DB = res.json()
  for tool in tools_DB:
    delete_data(tool,schema)


# dump json from BioToolStory to file
def write_to_file(schema):
  res = requests.get(API_url%(schema,""))
  tools_DB = res.json()
  with open(os.path.join(PTH,schema + '.json'), 'w') as outfile:
    json.dump(tools_DB, outfile)


# update tool in middleman
def update_middleman(tool):
  time.sleep(1)
  res = requests.patch('https://maayanlab.cloud/biotoolstory/middleman/api/signatures/' + tool['id'], json=tool, auth=auth_middle)
  if (not res.ok):
    print(res.text)
    time.sleep(60)
    return ("error")


# check if the tool was already pushed to middleman website
def is_pushed(pmid):
  res = requests.get('https://maayanlab.cloud/biotoolstory/middleman/api/signatures?filter={"limit": 10000}', auth=auth_middle)
  tools_DB = res.json()
  for tool in tools_DB:
    if int(pmid) == tool['meta']['PMID'][0]:
      print('PMID', pmid, 'exists in middleman')
      return(False)
  return(True)


# push daily data to the middleman service for manual approval
def post_data_middleman(data):
  if is_pushed(data['meta']['PMID'][0]):
    API  = "https://maayanlab.cloud/biotoolstory/middleman/api"
    data['meta']['Year']=int(data['meta']['Year'])
    data['meta']['Citations']=int(data['meta']['Citations'])
    res = requests.post(API+"/signatures/" + data["id"], json=data, auth=auth_middle)
    try:
      if not res.ok:
        raise Exception(res.text)
    except Exception as e:
      print(e)


# push data (tools or journals) directly to the biotoolstory server
def post_data(data,model):
  time.sleep(0.5)
  res = requests.post(API_url%(model,""), auth=credentials, json=data)
  try:
    if not res.ok:
      raise Exception(res.text)
  except Exception as e:
    print(e)
    if model == "signatures":
      f = open(os.path.join(PTH,"data/fail_to_load.txt"), "a")
      f.write(','.join(map(str, data['meta']['PMID'])) + "\n")
      f.close()


# update the website after petching data
def refresh():
  res = requests.get("https://maayanlab.cloud/biotoolstory/metadata-api/optimize/refresh", auth=credentials)
  print(res.ok)
  time.sleep(2)
  res = requests.get("https://maayanlab.cloud/biotoolstory/metadata-api/"+"optimize/status", auth=credentials)
  time.sleep(2)
  while not res.text == "Ready":
    time.sleep(2)
    res = requests.get("https://maayanlab.cloud/biotoolstory/metadata-api"+"/optimize/status", auth=credentials)
  time.sleep(2)
  res = requests.get("https://maayanlab.cloud/biotoolstory/metadata-api/"+"summary/refresh", auth=credentials)
  print(res.ok)

#==================================================  HELP FUNCTIONS ===================================================================
def is_key(data,key):
  if key in data.keys():
    return(data[key])
  else:
    return('')


def isnan(x):
  if type(x) == list:
    return(x)
  if type(x) == str:
    if x=='nan':
      return('')
    return(x)
  if math.isnan(x):
    return('')
  else:
    return(x)


def restructure_author_info(data):
  data = fix_dirty_json(data)
  if data == '':
    return('')
  res = []
  for x in data:
    res.append({
        "Name": isnan(is_key(x,'ForeName')) + " " + isnan(is_key(x,'LastName')),
        "ForeName": isnan(is_key(x,'ForeName')),
        "Initials": isnan(is_key(x,"Initials")),
        "LastName": isnan(is_key(x,'LastName')),
        "AffiliationInfo": [ isnan(is_key(y,'Affiliation')) for y in x['AffiliationInfo'] ]
    }
    )
  return(res)


# fix pubmed json 
def fix_dirty_json(text,flg=False):
  if isinstance(text, pd.Series):
    text = text.tolist()[0]
  if isinstance(text, list):
    return(text)
  try:
    x = ast.literal_eval(text)
  except:
    if(flg):
      x = text
    else:
      x = []
  return(x)


# find the most recent tool in case of duplicare tools
def find_max(duplicates):
  mx = duplicates[0]
  mn_date = 'None'
  if 'Article_Date' in mx['meta']:
    mx_date = datetime.strptime(mx['meta']['Article_Date'], '%Y-%m-%d')
    for tool in duplicates:
      try:
        dt = datetime.strptime(tool['meta']['Article_Date'], '%Y-%m-%d')
        if mx_date > dt:
          mx = tool
          mx_date = dt
        if mn_date < dt:
          mn_date = dt
      except:
        pass
  return([mx,mn_date])


def find_duplicates(tools_DB):
  urls = []
  for i in range(len(tools_DB)):
      urls.append(tools_DB[i]['meta']['tool_homepage_url'])
  # a list of unique duplicated urls
  dup_links = [item for item, count in collections.Counter(urls).items() if count > 1]
  return(dup_links)


def pmid_tolist(tools_DB):
  pmids = []
  for i in range(len(tools_DB)):
    if len(tools_DB[i]['meta']['PMID']) >1:
      for x in tools_DB[i]['meta']['PMID']:
        pmids.append(x)
    else:
      pmids.append(tools_DB[i]['meta']['PMID'][0])
  return(pmids)


def empty_cleaner(obj):
  if type(obj) == str:
    obj = obj.strip()
    if obj == "":
      return None
    else:
      return obj
  elif type(obj) == list:
    new_list = []
    for i in obj:
      v = empty_cleaner(i)
      if v or v==0:
        new_list.append(v)
    if len(new_list) > 0:
      return new_list
    else:
      return None
  elif type(obj) == dict:
    new_dict = {}
    for k,v in obj.items():
      val = empty_cleaner(v)
      if val or val == 0:
        new_dict[k] = val
    if len(new_dict) > 0:
      return new_dict
    else:
      return None
  else:
    return obj

#=================================================== Detect the topic of a tool ==================================================================
def sent_to_words(text):
  return(gensim.utils.simple_preprocess(str(text), deacc=True))  # deacc=True removes punctuations


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out


def predict_topic(text, nlp=nlp):
    # load the LDA model and vectorier
    lda_model = pickle.load(open(os.path.join(PTH,'LDA/LDA_model.pk'), 'rb'))
    vectorizer = pickle.load(open(os.path.join(PTH,'LDA/vectorizer.pk'), 'rb'))
    # Clean with simple_preprocess
    mytext_2 = [sent_to_words(text)]
    # Lemmatize
    mytext_3 = lemmatization(mytext_2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    # Vectorize transform
    mytext_4 = vectorizer.transform(mytext_3)
    # Step 4: LDA Transform
    topic_probability_scores = lda_model.transform(mytext_4)
    dt = pd.DataFrame({ 'Topic':[
                                  'Genome sequence databases',
                                  'Alignment algorithms and methods',
                                  'Tools to perform sequence analysis',
                                  'Sequence-based prediction of DNA and RNA',
                                  'Disease study using gene expression',
                                  'Protein structure',
                                  'Biological pathways and interactions',
                                  'Drugs and chemical studies',
                                  'Brain studies using images'
                                  ],
                      'LDA_probability': topic_probability_scores[0],
                      'Topic_number': ['1','2','3','4','5','6','7','8','9']
                    })
    dt = dt.sort_values('LDA_probability',ascending=False)
    dt.reset_index().to_json(orient='records')
    js=dt.to_json(orient='records')
    return(ast.literal_eval(js))


def final_test(data):
  try:
    if 'Last_Author' in data['meta'].keys():
      if type(data['meta']['Last_Author'])==list:
        data['meta']['Last_Author'] = data['meta']['Last_Author'][0]
    if 'Author_Information' in data['meta'].keys():
      for x in range(len(data['meta']['Author_Information'] ) ):
        if 'AffiliationInfo' in data['meta']['Author_Information'][x]:
          if type(data['meta']['Author_Information'][x]['AffiliationInfo'][0]) == str:
            data['meta']['Author_Information'][x]['AffiliationInfo'] = [{ 'Affiliation' : y} for y in data['meta']['Author_Information'][x]['AffiliationInfo']]
  except Exception as e:
    data['meta']['Author_Information'] = ""
    print(e)
  return(data)


def testURL(data):
  url = data['meta']['tool_homepage_url']
  try:
    request = requests.head(url,allow_redirects=False, timeout=5)
    status = request.status_code
  except Exception as e:
    status = e
  return(status)
  
  
def get_inst(id_):
  handleS = Entrez.efetch(db="pubmed", id=id_,rettype="xml", api_key=API_KEY)
  records = Entrez.read(handleS)
  try:
    x = records['PubmedArticle'][0]['MedlineCitation']['Article']['AuthorList'][-1]['AffiliationInfo'][0]['Affiliation'].split(",")
    if len(x) >0:
      x = x[0].strip()
      x = re.sub("university", "Uni", x, flags=re.I)
      x = re.sub("european", "EU", x, flags=re.I)
      x = re.sub("institute", "Inst", x, flags=re.I)
      x = re.sub("technology", "Tech", x, flags=re.I)
      x = re.sub("technical", "Tech", x, flags=re.I)
      x = re.sub("singapore", "SG", x, flags=re.I)
      x = re.sub("california", "CA", x, flags=re.I)
      x = re.sub("science", "Sci", x, flags=re.I)
      x = re.sub("national", "Nat'l", x, flags=re.I)
      x = re.sub("electronic", "Elec", x, flags=re.I)
      x = re.sub("Denmark", "DK", x, flags=re.I)
    else:
      x = 'err'
  except Exception as e:
    x = 'err'
  return(x)
  
  
  
  
  
#================================================ Push data ============================================================================================

def push_new_journal(ISSN):
  try:
    time.sleep(1)
    url = 'http://api.crossref.org/journals/' + urllib.parse.quote(ISSN)
    resp = req.get(url)
    text = resp.text
    resp = json.loads(text)
    jour = resp['message']['title'] #journal name
    pub = resp['message']['publisher']
  except Exception as e:
    print("error in push_new_journal() --> ", e)
    jour = 'NA'
    pub = 'NA'
  new_journal = {'$validator': '/dcic/signature-commons-schema/v5/core/library.json', 
  'id': str(uuid.uuid4()),
  'dataset': 'journal',
  'dataset_type': 'rank_matrix', 
  'meta': {
    'Journal_Title': jour,
    'ISSN': ISSN,
    'publisher': pub,
    'icon': '',
    # replace validator with raw.github
    '$validator': 'https://raw.githubusercontent.com/MaayanLab/BioToolStory/master/validators/btools_journal.json'
      #'/dcic/signature-commons-schema/v5/core/unknown.json'  
    }
    }
  new_journal = empty_cleaner(new_journal)
  post_data(new_journal,"libraries")
  return(new_journal['id'])


def push_tools(df):
  k = len(df)
  i = 1
  res = requests.get(API_url%("signatures",""))
  tools_DB = res.json()
  tools_pmids = []
  for x in tools_DB:
    for y in x['meta']['PMID']:
      tools_pmids.append(y)
  tools_pmids = list(set(tools_pmids))
  if 'Author_Information' in df.columns:
    keep = df.columns.drop(['Author_Information'])
  else:
    keep = df.columns
  for tool in df.to_dict(orient='records')[0:]:
    print('Uploaded',i,'tools out of',k)
    i = i + 1
    data = {}
    data["$validator"] = '/dcic/signature-commons-schema/v5/core/signature.json'
    data["id"] = str(uuid.uuid4()) # create random id
    ISSN = isnan(tool['ISSN'])
    # get journals from DB
    res = requests.get(API_url%("libraries",""))
    journal_list = res.json()
    key = [x['id'] for x in journal_list if x['meta']['ISSN']==ISSN ]
    if len(key)>0:
      data["library"] = key[0] # uuid from libraries TABLE
    else:
      data["library"]  = push_new_journal(ISSN)
    data["meta"] = { key: tool[key] for key in keep }
    data["meta"]["PMID"] = [int(tool["PMID"])]
    data["meta"]["Abstract"] =  fix_dirty_json(tool['Abstract'],flg=True)
    if data["meta"]["Abstract"] == '': # this is a mandatory field
      print("missing abstract")
      continue
    data["meta"]["Article_Language"] =  fix_dirty_json(tool['Article_Language'])
    data["meta"]["Author_Information"] = restructure_author_info(tool['Author_Information'])
    data['meta']['Last_Author'] = {
      'Name': isnan(data["meta"]["Author_Information"][-1]['ForeName']) + " " + isnan(data["meta"]["Author_Information"][-1]['LastName']),
      'ForeName': isnan(data["meta"]["Author_Information"][-1]['ForeName']),
      'Initials': isnan(data["meta"]["Author_Information"][-1]['Initials']),
      'LastName': isnan(data["meta"]["Author_Information"][-1]['LastName'])
      }
    inst = get_inst(data['meta']['PMID'][0])
    if inst != 'err':
      data['meta']['Institution'] = inst
    data['meta']['Topic'] = predict_topic(data["meta"]["Abstract"])
    data["meta"]["Electronic_Location_Identifier"] =  str(fix_dirty_json(tool['DOI']))
    data["meta"]["Publication_Type"] =  fix_dirty_json(tool['Publication_Type'])
    data["meta"]["Grant_List"] =  fix_dirty_json(tool['Grant_List'])
    data["meta"]["Chemical_List"] =  fix_dirty_json(tool['Chemical_List'])
    data["meta"]["KeywordList"] =  fix_dirty_json(tool['KeywordList'])
    if len(data["meta"]["KeywordList"]) > 0:
      if (isinstance(data["meta"]["KeywordList"], list)) and (data["meta"]["KeywordList"]):
        data["meta"]["KeywordList"] = isnan(data["meta"]["KeywordList"][0])
        # https://raw.githubusercontent.com/MaayanLab/btools-ui/toolstory/validators/btools_tools.json
        #'/dcic/signature-commons-schema/v5/core/unknown.json'
    data["meta"]["$validator"] = 'https://raw.githubusercontent.com/MaayanLab/BioToolStory/master/validators/btools_tools.json'
    data['meta']['Published_On'] =''
    data['meta']['Added_On']=''
    data['meta']['Last_Updated']=''
    code= testURL(data)
    if len(str(code) )> 4:
      data['meta']['url_status']= str(code)
    else:
      data['meta']['url_status']= str(code) +":"+str(http.client.responses[code])
    data["meta"] = empty_cleaner(data['meta']) # delete empty fields
    data = final_test(data)
    # check that the pmid does not exist in the dataset
    if data['meta']['PMID'][0] in tools_pmids:
      print("pmid",data['meta']['PMID'], 'exist')
      pass
    else:
      tools_pmids.append(data['meta']['PMID'])
      post_data_middleman(data) # post tools to be manually validated before pushing them to biotoolstory
      #post_data(data,"signatures") # enable to push each tool directly to the server
      #refresh()  # enable to push each tool directly to the server


def read_data(fpath):  
  try:
    return(pd.read_csv(fpath))
  except:
    print("No tools were detected for",start)
    sys.exit()


def deletefeiles():
  try:
    if os.path.exists(os.path.join(PTH,'data/tools_'+s+'_'+en+'.csv')):
      os.remove(os.path.join(PTH,'data/tools_'+s+'_'+en+'.csv'))
    # remove folders
    if os.path.exists(os.path.join(PTH,'data/tools_'+s+'_'+en)):
      shutil.rmtree(os.path.join(PTH,'data/tools_'+s+'_'+en))
    if os.path.exists(os.path.join(PTH,'data/jsons_'+s+'_'+en)):
      shutil.rmtree(os.path.join(PTH,'data/jsons_'+s+'_'+en))
  except Exception as e:
    print(e)
  
#================================================== Main ===========================================================================================

if __name__ == "__main__":
  if os.path.exists(os.path.join(PTH,"data/fail_to_load.txt")):
    os.remove(os.path.join(PTH,"data/fail_to_load.txt")) # delete failure log file from last time
  mypath = os.path.join(PTH,'data')
  new_tools = [f for f in listdir(mypath) if ( isfile(join(mypath, f)) ) and (f.startswith('classified_tools') ) ]
  for file in new_tools:
    try:
      df = read_data(os.path.join(PTH,'data/'+file))
      df = df.replace(np.nan, '', regex=True)
      deletefeiles()
      push_tools(df)
    except Exception as er:
      print(er,'line 481')
    try:
      if os.path.exists(os.path.join(PTH,'data/'+file)):
        os.remove(os.path.join(PTH,'data/'+file))
      if dry_run:
        with open(os.path.join(PTH,schema + '.json'), 'w') as outfile:
          json.dump(all_tools, outfile)
    except Exception as e:
      print(e, 'line 533')
    print("Done!",file)
 
#-------------------------------------------------------------------------------------------------------------------------------------------------------------
# Example of how to update data on middlemane
# for i in range(0,len(tools_DB)):
#   print(i)
#   tools_DB[i]['meta']['Topic'] = predict_topic(tools_DB[i]["meta"]["Abstract"])
#   print(tools_DB[i]['meta']['Topic'])
#   update_middleman(tools_DB[i])
