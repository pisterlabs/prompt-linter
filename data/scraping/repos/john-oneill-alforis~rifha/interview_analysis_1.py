import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel

from pprint import pprint
from dotenv import load_dotenv
import pyLDAvis
import pyLDAvis.gensim

import matplotlib.pyplot as plt
import pandas as pd
import mysql.connector
import os
import re


def interview_analysis_question_1():

    from nltk.corpus import stopwords
    stop = stopwords.words('english')

    load_dotenv()
    # Database Connection to Pull Existing Meta Values

    mydb = mysql.connector.connect(
        user=(os.getenv("db_user")),
        password=(os.getenv("db_password")),
        host="localhost",
        database="thesis_vert",
    )

    responses = pd.read_sql_query("""SELECT secondary_answer_text FROM thesis_vert.polls_transcriptcapture where question_id IN (1,2);""", mydb) #Change this with the name of your downloaded file
    responses['secondary_answer_text'] = responses['secondary_answer_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    responses['secondary_answer_text'] = responses['secondary_answer_text'].apply(str.lower)
    responses['secondary_answer_text'] = responses['secondary_answer_text'].str.replace('\n', '')
    responses['secondary_answer_text'] = responses['secondary_answer_text'].str.replace('.', '')
    responses['secondary_answer_text'] = responses['secondary_answer_text'].str.replace(',', '')
    responses['secondary_answer_text'] = responses['secondary_answer_text'].str.replace("'", '')
    #responses['secondary_answer_text'] = responses['secondary_answer_text'].str.replace('"', '')

    responses = responses.secondary_answer_text.values.tolist()

    print(responses)

    # Remove unused characters
    #responses = [t.replace('\n', '') for t in responses]
    #responses = [t.replace('\n\n', '') for t in responses]
    #responses = [t.replace('"', '') for t in responses]
    #responses = [t.replace('.', '') for t in responses]
    #responses = [t.lower() for t in responses]
    responses = [t.split(' ') for t in responses]

    #print(responses)


    id2word = Dictionary(responses)


    

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in responses]
    #print(corpus[:1])


    # Build LDA model
    lda_model = LdaModel(corpus=corpus,
                    id2word=id2word,
                    num_topics=10,
                    random_state=0,
                    chunksize=100,
                    alpha='auto',
                    per_word_topics=True)

    #pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    #Creating Topic Distance Visualization 

    visualisation = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(visualisation, 'LDA_Visualization_1.html')


    
   


interview_analysis_question_1()