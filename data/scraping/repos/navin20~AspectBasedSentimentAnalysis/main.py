import pandas as pd
import numpy as np
import joblib
import timeit
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils.preprocess import preprocess
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import *
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import time
app = Flask(__name__)
import pyrebase
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
# from utils.connectdatabase import connectingwith
import pandas as pd
import os
from utils import sentiment_classifiers
from utils import stats
from utils import data_to_csv
from utils.database import connects
from liba import inject, delete_collection, delete_in_batch
# app = Flask(__name__)
import firebase_admin
from firebase_admin import credentials, firestore

'''
Main file. 
1. Starts by pre-loading all models.
2. Enter loop
3. -------------
4. API should check if user put new data (file or typed)
5. If data is file, retrieve file(csv) and convert to df
6. Else if data is user input retrieve user input and convert to df 
7. Preprocess df
8. Classify aspects
9. Classify sentiments
10. New predicted data should be appended to previous predicted data (if any), for website to use.
11. (optional) save file to disk
12. Functions are provided to retrieve stats for visualization and details page of web.
13. If no user data, then it loops back to step 3.

* See individual files for their details *
'''
db = connects()


def get_check_box_a(checkbox_a):
    box = []
    if checkbox_a[0] != False:
        box.append("a1")
    if checkbox_a[1] != False:
        box.append("a2")
    if checkbox_a[2] != False:
        box.append("a3")
    if checkbox_a[2] != False:
        box.append("a4")
    if checkbox_a[2] != False:
        box.append("no")
    return box

def get_check_box_s(checkbox_s):
    box = []
    if checkbox_s[0] != False:
        box.append("positive")
    if checkbox_s[1] != False:
        box.append("negative")
    if checkbox_s[2] != False:
        box.append("neutral")
    return box

def check_sentiment(checkbox_s, sentiment):
    for i in checkbox_s:
        if sentiment == i:
            return True
    return False

def check_aspect(checkbox_a,a1,a2,a3,a4):
    for i in checkbox_a:
        if (a1 != "" and i =='a1' ) or (a2 != "" and i =='a2' ) or (a3 != "" and i =='a3' ) or (a4 != "" and i =='a4' ):
            return True
    return False

def contains_aspect(checkbox_a):
    for i in checkbox_a:
        if i == 'no':
            return True
    return False

def aspect_IS_null(a1,a2,a3,a4):
    if a1 == "" and a2 == "" and a3 == "" and a4 == "":
        return True
    else:
        return False

app.jinja_env.globals.update(check_sentiment=check_sentiment,check_aspect=check_aspect,contains_aspect=contains_aspect,aspect_IS_null=aspect_IS_null)


def read_from_file(file):
    """
    Reads from file.
    """

    col = ['review']
    df = pd.read_csv(file, header=None, names=col)
    print('Loaded data')

    return df


        
def read_from_sentences(sen1,sen2,sen3,sen4,sen5):
    """
    Reads from individual sentences.
    """
    str1 = ''+sen1
    str2 = ''+sen2
    str3 = ''+sen3
    str4 = ''+sen4
    str5 = ''+sen5

    reviews = [str1+str2+str3+str4+str5]
    col = ['review']
    df = pd.DataFrame(columns=col)
    df['review'] = reviews
    print('Loaded data')

    return df

@app.route('/visualization')
def visualization():
    return render_template("pie_chart.html")

@app.route('/detail',methods = ['GET','POST'])

def detail():
    database = firestore.client()
    col_ref = database.collection('details') # col_ref is CollectionReference
    results = col_ref.order_by('sentence',direction='DESCENDING').get()
    data = []
    for item in results:
        data.append(item.to_dict())


    if request.method == 'POST':
        if request.form['submit'] == 'submit':
            pos = 'pos' in request.form
            neg = 'neg' in request.form
            neu = 'neu' in request.form
            a1 = 'a1' in request.form
            a2 = 'a2' in request.form
            a3 = 'a3' in request.form
            a4 = 'a4' in request.form
            no = 'no' in request.form
            print(pos, neg, neu, a1,a2,a3,a4,no)

            sent = [pos,neg,neu]
            aspect = [a1,a2,a3,a4,no]
                
            return render_template('detail.html', details_array=data, checkbox_s=get_check_box_s(sent), checkbox_a=get_check_box_a(aspect))

    sent = [True,True,True]
    aspect = [True,True,True,True,False]
        
    return render_template('detail.html', details_array=data, checkbox_s=get_check_box_s(sent), checkbox_a=get_check_box_a(aspect))

@app.route('/testing',methods = ['GET','POST'])
def testing():
     
    start = timeit.default_timer()
    # Load models
    classifiers = sentiment_classifiers.Asp_Sentiment_classifiers()
    stop = timeit.default_timer()
    print('Model load times: ', stop - start)

    if request.method == 'POST':
        if request.form['submit'] == 'submit':
            # print('test')
            
            

            sentence1 = request.form['sentence1']+""
            sentence2=request.form['sentence2']+""
            sentence3=request.form['sentence3']+""
            sentence4=request.form['sentence4']+""
            sentence5 = request.form['sentence5']+""

            # direc1 = dal+"/"+name
            # for root,dirs,files in os.walk('D:/William/Workspace/test'):
            #     for name in files:
            #         if file_ in name:
            #             direc1 = os.path.abspath(os.path.join(root, name))
            # direc2 = direc1.replace("\\","/")
            # # print(direc2)

            df = read_from_sentences(sentence1,sentence2,sentence3,sentence4,sentence5)
                
            collection_name = 'details'
            print('Preprocessing...')
            reviews_df = preprocess(df)
            print('Classifying aspects...')
            reviews_df = classifiers.aspect_classifier(reviews_df, 0.06)
            print('Classifying sentiments...')
            reviews_df = classifiers.sentiments(reviews_df)
            print('Done!')
            
            # Detail page
            details_data = stats.get_details(reviews_df)

            data_to_csv.save_file(details_data)

            inject('data/data_pred.csv',collection_name)




            # Update (testing page)
            details_data
            s1 = ["","","","",""]

            for i in range(len(details_data)):
                s1[i] = (details_data['sentence'][i])

            det = {"userinput": {"sentence1": s1[0], "sentence2": s1[1], "sentence3": s1[2], "sentence4": s1[3], "sentence5": s1[4]}}

            db.child("testing").update(det)


            # Get new data (update testing page only)
            overall, aspect1, aspect2, aspect3, aspect4, _ = stats.get_chart_data(reviews_df)

            detailer1 = {"overall":{"positive": overall[0],"neutral": overall[1],"negative": overall[2]},
                "aspect1":{"positive": aspect1[0],"neutral": aspect1[1],"negative": aspect1[2]},
                "aspect2":{"positive": aspect2[0],"neutral": aspect2[1],"negative": aspect2[2]},
                "aspect3":{"positive": aspect3[0],"neutral": aspect3[1],"negative": aspect3[2]},
                "aspect4":{"positive": aspect4[0],"neutral": aspect4[1],"negative": aspect4[2]},}
            db.child("testing").update(detailer1)





            # Update all pie chart data
            # Get existing data
            ref = db.child('visualization').get()
            a1 = ref.each()[0].val()
            a2 = ref.each()[1].val()
            a3 = ref.each()[2].val()
            a4 = ref.each()[3].val()
            ov = ref.each()[4].val()

            # Combine
            overall = [overall[0] + ov['positive'], overall[1] + ov['neutral'], overall[2] +ov['negative']]
            aspect1 = [aspect1[0] + a1['positive'], aspect1[1] + a1['neutral'], aspect1[2] +a1['negative']]
            aspect2 = [aspect2[0] + a2['positive'], aspect2[1] + a2['neutral'], aspect2[2] +a2['negative']]
            aspect3 = [aspect3[0] + a3['positive'], aspect3[1] + a3['neutral'], aspect3[2] +a3['negative']]
            aspect4 = [aspect3[0] + a4['positive'], aspect4[1] + a4['neutral'], aspect4[2] +a4['negative']]

            # Update
            detailer = {"overall":{"positive": overall[0],"neutral": overall[1],"negative": overall[2]},
                "aspect1":{"positive": aspect1[0],"neutral": aspect1[1],"negative": aspect1[2]},
                "aspect2":{"positive": aspect2[0],"neutral": aspect2[1],"negative": aspect2[2]},
                "aspect3":{"positive": aspect3[0],"neutral": aspect3[1],"negative": aspect3[2]},
                "aspect4":{"positive": aspect4[0],"neutral": aspect4[1],"negative": aspect4[2]},}
            db.child("visualization").update(detailer)

            


            return render_template("testing.html")
    return render_template("testing.html")


            #     # Optional
            # data_to_csv.save_file(reviews_df)

            # # Using stats functions, data can be retrieved from newly predicted data or all data.
            # # Gets info for pie charts. (Returns arrays of overall_sent_values, aspect1, aspect2, aspect3, aspect4, total_data)
            # # **Sorted by negative, neutral, positive.**
            # stats.get_chart_data(reviews_df)
            # # Contains all sentence details. (as 2d array)
            # stats.get_details(reviews_df)
       

# if __name__ == "__main__":
@app.route('/', methods=['GET', 'POST'])
def index():
    
    start = timeit.default_timer()
    # Load models
    classifiers = sentiment_classifiers.Asp_Sentiment_classifiers()
    stop = timeit.default_timer()
    print('Model load times: ', stop - start)

    if request.method == 'POST':
        # if request.form['submit'] == 'submit':
            # print('test')
            
            

            file_ = request.form['file']+""
            # direc1 = dal+"/"+name
            for root,dirs,files in os.walk('D:/William/Workspace/test2'):
                for name in files:
                    if file_ in name:
                        direc1 = os.path.abspath(os.path.join(root, name))
            direc2 = direc1.replace("\\","/")
            # print(direc2)

            df = read_from_file(direc2)
                
            collection_name = 'details'
            print('Preprocessing...')
            reviews_df = preprocess(df)
            print('Classifying aspects...')
            reviews_df = classifiers.aspect_classifier(reviews_df, 0.1)
            print('Classifying sentiments...')
            reviews_df = classifiers.sentiments(reviews_df)
            print('Done!')

            delete_in_batch()

            details = stats.get_details(reviews_df)

            data_to_csv.save_file(details)

            inject('data/data_pred.csv',collection_name)





            # Get new data
            overall, aspect1, aspect2, aspect3, aspect4, _ = stats.get_chart_data(reviews_df)

            # # Get existing data and update
            # ref = db.child('visualization').get()
            # a1 = ref.each()[0].val()
            # a2 = ref.each()[1].val()
            # a3 = ref.each()[2].val()
            # a4 = ref.each()[3].val()
            # ov = ref.each()[4].val()

            # # Combine
            # overall = [overall[0] + ov['positive'], overall[1] + ov['neutral'], overall[2] +ov['negative']]
            # aspect1 = [aspect1[0] + a1['positive'], aspect1[1] + a1['neutral'], aspect1[2] +a1['negative']]
            # aspect2 = [aspect2[0] + a2['positive'], aspect2[1] + a2['neutral'], aspect2[2] +a2['negative']]
            # aspect3 = [aspect3[0] + a3['positive'], aspect3[1] + a3['neutral'], aspect3[2] +a3['negative']]
            # aspect4 = [aspect3[0] + a4['positive'], aspect4[1] + a4['neutral'], aspect4[2] +a4['negative']]

            # Update
            detailer = {"overall":{"positive": overall[0],"neutral": overall[1],"negative": overall[2]},
                "aspect1":{"positive": aspect1[0],"neutral": aspect1[1],"negative": aspect1[2]},
                "aspect2":{"positive": aspect2[0],"neutral": aspect2[1],"negative": aspect2[2]},
                "aspect3":{"positive": aspect3[0],"neutral": aspect3[1],"negative": aspect3[2]},
                "aspect4":{"positive": aspect4[0],"neutral": aspect4[1],"negative": aspect4[2]},}
            db.child("visualization").update(detailer)

            
            

            

            

            


            return redirect(url_for('visualization'))
    return render_template("home_page.html")

        # timing for testing purposes
        # stop = timeit.default_timer()
        # print('1 Loop time: ', stop - start)
        # # Comment done to keep looping.
        # done=True
if __name__ == '__main__':
    app.run(debug=True)