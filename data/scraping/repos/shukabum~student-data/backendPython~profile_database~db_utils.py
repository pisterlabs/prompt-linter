from chromadb.api.types import Documents, Embeddings
from typing import List, Tuple
from langchain.docstore.document import Document
from newspaper import Article 
import requests
from langchain.vectorstores.chroma import Chroma
import sys
sys.path.append('backendPython/')
from utils import Embedding
import pandas as pd
from chains import *
import ast
import pickle 

path = 'backendPython/Placement data.xlsx'
df = pd.read_excel(path, sheet_name='2020-21')

values = {'Selected': df.iloc[:,2].mode() , 'CTC':df.iloc[:, 4].median() , 'CGPA':0}
df.fillna(value=values, inplace=True)

vector_db = Chroma(embedding_function = Embedding(), persist_directory= 'backendPython/profile_database/info_db')
cols = list(df.columns)

def get_about_company(url):
  title, article =  web_scraping(url)
  if article is None:
    return 'No article found'
  try:
    return summarize_chain.run(article)
  except:
    return article[:1000]
  

skill_template = """You  will be provided with a job role ,you need to follow the below instructions.
 

Instructions : 
1) get top 3 tech_stacks that they need to know for the given job role.
2) Be cautious that you put the skills in the order of their importance.
4) Return a dictionary of skills with tech_stacks needed as keys and their 1-2 line explanation as values. 
3) Don't put backticks(`) in the output. Keep check that there are no syntax errors in the output.

Job Profile : {job_profile}
"""
skill_prompt = PromptTemplate(
    input_variables=["job_profile"],
    template=skill_template,
)
skill_chain = LLMChain(llm=llm, prompt=skill_prompt,)

syntax_template = '''You will be provided with a code snippet for python dictionary , you need to find error in code and return 
 the correct code snippet. Return a string of correct code snippet. Don't put backticks(`) in the output. 
 Keep check that there are no syntax errors in the output.
Do not make any changes in the code snippet if there is no error part.


Code Snippet : 
{code}
'''

syntax_prompt = PromptTemplate(
    input_variables=["code"],
    template=syntax_template,
)
syntax_chain = LLMChain(llm=llm, prompt=syntax_prompt,)

syntax_err_ct = 0 

def get_context(job_profile):
  skills = skill_chain.run(job_profile)   # return a string format dict
  skills = syntax_chain.run(skills)   
  try :
    skills = ast.literal_eval(skills)    # 1.) removing backticks , 2.) convert string to dict
  except : 
    syntax_err_ct += 1
    skills = {'Hardwork':'Hardwork is great' , 'GrateFullness' : 'Gratefullness is great, it would keep you motivated' }  
    
  return skills


def web_scraping(url):
    headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36'
            }
    session = requests.Session()

    try:
      response = session.get(url, headers=headers, timeout=10)
    
      if response.status_code == 200:     # the request was successful
        article = Article(url)
        article.download()
        article.parse()
        
        print('All fetching successful!', end= '\n\n\n')
        return article.title, article.text
      else:
        print(f"Failed to fetch article at {url}")
    except Exception as e:
      print(f"Error occurred while fetching article at {url}: {e}") 
    
    return  None, None
  

docs = []

page_prompt = '''

The company is looking for {JobProfile}. The number of students that got selected for this profile are {Selected} and the average CTC is {CTC} LPA. The CGPA cutoff is {CGPA}.
You can find more about the company in its metadata. The company is looking for the following skills : {Skills}
'''
skill_set = set()
for index, row in df.iterrows():
  print(index , end = ' ')
  metadata = {}
  for col in cols:
    if col == 'About company':
      metadata['About Company'] = get_about_company(row[col])
    else :   
      metadata[col] = row.loc[col]

  skill_dict = get_context(metadata['JobProfile'])
  
  
  page_content = page_prompt.format(JobProfile = metadata['JobProfile'], Selected = metadata['Selected'], 
                                    CTC = metadata['CTC'], CGPA = metadata['CGPA'] , Skills = ', '.join(skill_dict.keys()))
  skill_set.update(set(skill_dict.keys()))
  metadata['Skills'] = ', '.join(skill_dict.keys())
  docs.append(Document(page_content=page_content, metadata=metadata))


vector_db.from_documents(docs, Embedding(), persist_directory= 'backendPython/profile_database/info_db')
print('syntax_err_ct : ', syntax_err_ct)

pickle.dump(list(skill_set), open('backendPython/profile_database/skill_set.pkl', 'wb'))
print(vector_db.similarity_search('python , nodejs'))