import openai 
import itertools
import threading
import time
import sys
import pandas as pd
import numpy as np
from pandasql import sqldf
import argparse
import duckdb

parser = argparse.ArgumentParser(description='BI Assitant')
parser.add_argument('-r','--max_rows', type=int, help='Maximum rows to read from the IMDB tables - if set, the answers will not be accurate but the loading will be faster', required=False, default=sys.maxsize)
parser.add_argument('-k','--api_key', type=str, help='Open AI key. You can get it from https://beta.openai.com/account/api-keys', required=True)
parser.add_argument('-p','--path', type=str, help='IMDB non commercial dataset path', required=False, default='data')
args = vars(parser.parse_args())

key = args['api_key']
    
openai.api_key = key

pysqldf = lambda q: sqldf(q, globals())

def animate(msg):
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write(f'\r{msg} ' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\n')
       

print("Loading Data - It Will Take Some Time...")

imdb_titles = pd.read_csv(f'{args["path"]}/imdb_titles.csv', nrows=args['max_rows'], dtype={
    'titleId':           'string',
    'language':          'category',
    'titleType':         'string',
    'title':             'string',
    'isAdult':            bool,
    'startYear':          float,
    'endYear':            float,
    'runtimeMinutes':     float,
    'genres':             'category',
    'directors':          'category',
    'writers':            'category',
    'averageRating':       float,
    'numVotes':            float,
    'NumberSeasons':       float,
    'NumberEpisodes':      float,

    })

print("title_akas_df loaded (1/3)")
imdb_actors_titles = pd.read_csv(f'{args["path"]}/imdb_actors_titles.csv', nrows=args['max_rows'], dtype={

    'tconst':              'string',
    'nconst':              'string',
    'characters':          'category',

    }) 

print("title_basics_df loaded (2/3)")
imdb_actors_details = pd.read_csv(f'{args["path"]}/imdb_actors_details.csv', nrows=args['max_rows'], dtype={

    'primaryName':         'string',
    'birthYear':           float,
    'deathYear':           float,
    'knownForTitles':      'category',
    'nconst':              'string',
    }) 

print("title_basics_df loaded (3/3)")

answer_refine_template = "use the following query result for your answer: "

question_query_template = """
CURRENT_YEAR = 2023

Here's the description of IMDB Dataset:

imdb_titles: 

titleType (string) - the type of the title (movie, tvseries, tvepisode, tvMovie, tvMiniSeries)
titleId (string) - unique identifier of the title
title (string) - the title
NumberSeasons (float64) - number of seasons of the title (if titleType is tvSeries or tvMiniSeries) else NaN
NumberEpisodes (float64) - number of episodes of the title (if titleType is tvSeries or tvMiniSeries) else NaN
startYear (float64) - year of the title can be NaN
endYear (float64) - end year of the title can be NaN
language (category) - the different languages of the title
isAdult (bool) - 0: non-adult title; 1: adult title
runtimeMinutes (float64) - primary runtime of the title, in minutes if applicable, else NaN 
directors (category) - director(s) of the given title
writers (category) - writer(s) of the given title
averageRating  (float64) - if applicable, else NaN 
numVotes (float64) - if applicable, else NaN 
genres - (category) - genres associated with the title
       
imdb_actors_titles:

nconst (string) - identifier of the actor
tconst (string) - identifier of the title
characters (category) - characters the actor played in the title

imdb_actors_details:

nconst (string)  - identifier of the actor
primaryName - actor's name
birthYear (float64) - if applicable, else NaN 
deathYear (float64) - if applicable, else NaN 
knownForTitles (category) - titles the actor is known for

Please provide an SQL query to answer the following question:

**Additional Instructions:**
- Don't use ARRAY_CONTAINS in your query.
- Check for NaN where neccessary.
- Add ';' at the end of your query.
- Be accurate with the column names (see above).
- Be accurate with the queries.
- Change titles and names to CammelCase.
"""

CURRENT_YEAR = 2023

while True: 
     
    question = ''
    while question == '':
        question = input("Enter Your Question: ").replace('\n', '')

    done = False
    t = threading.Thread(target=animate, args=['Waiting For The Answer...'])
    t.start() 
    
    chat = openai.ChatCompletion.create( 
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": question_query_template},
                                         {"role": "user", "content": question}], 
        stop=[";"], temperature = 0.2
    ) 
    query_answer = 'no data available'
    for choice in chat.choices:
    
        reply = choice.message.content 
        reply = reply.replace("\\", "")
        reply = reply.replace("//", "")
        reply = reply.replace("CURRENT_YEAR", str(CURRENT_YEAR))
        
        if (reply.find("SELECT") == -1):
            continue
        
        query = reply[reply.find("SELECT"):].strip()

        try:
            res = duckdb.sql(query).df()
            query_answer = str(res)
            break
        except Exception as e:
            continue

    gpt_answer = f"{question} {answer_refine_template} {query_answer}"

    chat = openai.ChatCompletion.create( 
        model="gpt-3.5-turbo", messages=[{"role": "user", "content": gpt_answer}],
        temperature = 0.5, frequency_penalty = 2, presence_penalty = 2
    ) 
    done = True
    
    answer = chat.choices[0].message.content

    print(f"\n{answer}") 
