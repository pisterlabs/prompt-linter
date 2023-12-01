import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import folium
from folium.plugins import *
from folium import plugins
from streamlit_folium import folium_static
import datetime 
from time import strptime
from datetime import datetime, timedelta
import datetime as dt
import plotly.graph_objects as go
import base64
from pycaret.nlp import *
from plotly.graph_objs import *
import re
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

import warnings
import tensorflow
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
# Set seeds to make the experiment more reproducible.
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(1)

import streamlit as st
import pandas as pd
import base64
import time
from pycaret.nlp import *
import en_core_web_sm
import plotly.figure_factory as ff
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
from plotly.graph_objs import *
import sys
import re, numpy as np, pandas as pd
from pprint import pprint

# Gensim
import gensim, spacy, logging, warnings
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

import subprocess
def run_and_display_stdout(*cmd_with_args):
    result = subprocess.Popen(cmd_with_args, stdout=subprocess.PIPE)
    for line in iter(lambda: result.stdout.readline(), b""):
        st.text(line.decode("utf-8"))


def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

###########################
# Static Map Daily Observer
###########################

def map_obs():


        map_bd = folium.Map(location= [23.6850, 90.3563], tiles="cartodbpositron", zoom_start = 6)

        # Create a list with lat and long values and add the list to a heat map, then show map
        heat_data = [[row['Latitude'],row['Longitude']] for index, row in df_new.iterrows()]
        HeatMap(heat_data).add_to(map_bd)

        # instantiate a feature group for the incidents in the dataframe
        incidents = folium.map.FeatureGroup()

        # loop through the 100 crimes and add each to the incidents feature group
        for lat, lng, in zip(df_new.Latitude, df_new.Longitude):
            incidents.add_child(
                folium.CircleMarker(
                    [lat, lng],
                    radius=5, # define how big you want the circle markers to be
                    color='darkred',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.6
                )
            )
        
        #map_van.add_child(incidents)
        folium.TileLayer('cartodbdark_matter').add_to(map_bd)

        # instantiate a mark cluster object for the incidents in the dataframe
        incident = plugins.MarkerCluster().add_to(map_bd)

        # loop through the dataframe and add each data point to the mark cluster
        for lat, lng, label, in zip(df_new.Latitude, df_new.Longitude, df_new.Location_appox):
            folium.Marker(
                location=[lat, lng],
                icon=None,
                popup=label,
            ).add_to(incident)

        # add incidents to map
        map_bd.add_child(incident)
                    
        return map_bd 

###########################
# Static Map Dhaka Tribune
###########################

def map_(df_new):


    map_bd = folium.Map(location= [23.6850, 90.3563], tiles="cartodbpositron", zoom_start = 6)

    # Create a list with lat and long values and add the list to a heat map, then show map
    heat_data = [[row['Latitude'],row['Longitude']] for index, row in df_new.iterrows()]
    HeatMap(heat_data).add_to(map_bd)

    # instantiate a feature group for the incidents in the dataframe
    incidents = folium.map.FeatureGroup()

    # loop through the 100 crimes and add each to the incidents feature group
    for lat, lng, in zip(df_new.Latitude, df_new.Longitude):
        incidents.add_child(
            folium.CircleMarker(
                [lat, lng],
                radius=5, # define how big you want the circle markers to be
                color='darkred',
                fill=True,
                fill_color='red',
                fill_opacity=0.6
            )
        )
    
    #map_van.add_child(incidents)
    folium.TileLayer('cartodbdark_matter').add_to(map_bd)

    # instantiate a mark cluster object for the incidents in the dataframe
    incident = plugins.MarkerCluster().add_to(map_bd)

    # loop through the dataframe and add each data point to the mark cluster
    for lat, lng, label, in zip(df_new.Latitude, df_new.Longitude, df_new.Location_appox):
        folium.Marker(
            location=[lat, lng],
            icon=None,
            popup=label,
        ).add_to(incident)

    # add incidents to map
    map_bd.add_child(incident)
    return map_bd

###########################
# TS Plots using Plotly
###########################

def plot_comparison(loss, val_loss, x, xx, z, zz, model_name):
    
               
        l = list(range(len(loss)))
        ll = list(range(len(val_loss)))
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=l,
            y=loss,
            name = 'Training Loss', line=dict(color='salmon', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=ll,
            y=val_loss,
            name='Validation Loss', line=dict(color='lightseagreen', width=4,
                                      dash='dot')
        ))

        fig.update_layout(title='Model Performance: Loss vs Epochs',
                           xaxis_title='Epochs',
                           yaxis_title='MAE', template="ggplot2")

        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)


        st.plotly_chart(fig, use_container_width=True)

        #-------------------------------------------------

        ll = list(x.reset_index().Date.astype(str).values)
        lll = list(xx.reset_index().Date.astype(str).values)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=ll,
            y=x,
            name = 'Actual Price', line=dict(color='salmon', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=lll,
            y=xx,
            name='Predicted Price', line=dict(color='lightseagreen', width=4,
                                      dash='dot')
        ))

        fig.update_layout(title='Comparison between Actual & Prediction (Training Set)',
                           xaxis_title='Dates',
                           yaxis_title='Number of Death', template="ggplot2")

        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)


        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------------------------      
        

        oo = list(z.reset_index().Date.astype(str).values)
        ooo = list(zz.reset_index().Date.astype(str).values)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=oo,
            y=z,
            name = 'Actual Price', line=dict(color='salmon', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=ooo,
            y=zz,
            name='Predicted Price', line=dict(color='lightseagreen', width=2,
                                      dash='dot')
        ))

        fig.update_layout(title='Comparison between Actual & Prediction (Forecast)',
                           xaxis_title='Dates',
                           yaxis_title='Number of Death', template="ggplot2")

        fig.update_layout(xaxis_showgrid=False, yaxis_showgrid=False)


        st.plotly_chart(fig, use_container_width=True)


##################
# Set up sidebar #
##################

#st.sidebar.markdown('Road safety is a major concern in Bangladesh. Some AI-based initiatives have been made to collect the right datasets, bring insights from the collected data, forecast the number of people killed due to road crashes, and building an object tracking model to compute the velocity of vehicles.')

st.sidebar.image("wc (2).png", use_column_width=True)

option = st.sidebar.selectbox('Select from below', ( "Summary of Statistics", "Geographic Heatmaps", 'Time-Series Solutions', "Text Analytics Results", "Explore Our Awesome Datasets"))


###################
# Set up main app #
###################

st.markdown("<h1 style='text-align: center; color: white;'>AI for Improving Road Safety in Bangladesh</h1>", unsafe_allow_html=True)



if option == "Summary of Statistics":



    #st.text('Road safety is a major concern in Bangladesh. An estimate states that 55 people are killed in road crashes every day, and that vulnerable road users including walkers, motorcyclists, and unsafe and informal public transportation users account for more than 80% of road traffic deaths [1] . As a direct consequence of rapid growth in population, motorization and urbanization, the situation is deteriorating rapidly. The potential of maturing Artificial Intelligence (AI) and Internet-of-Things (IoT) technologies to enable rapid improvements in road safety has been largely overlooked. There is a pressing need and opportunity to improve road safety by enacting effective and coordinated Artificial Intelligence-driven (AI) policies and actions, which will necessitate significant improvements in the relevant sectors, such as better enforcement, better roads including improving design to eliminate accident black spots, and improved public education programs.')
    #st.markdown('Road safety is a major concern in Bangladesh. Some AI-based initiatives have been made to collect the right datasets, bring insights from the collected data, forecast the number of people killed due to road crashes, and building an object tracking model to compute the velocity of vehicles.')
    st.markdown("<p style='text-align: center; color: white;'>Road safety is a major concern in Bangladesh. Some AI-based initiatives have been made to collect the right datasets, bring insights from the collected data, forecast the number of people killed due to road crashes, and building an object tracking model to compute the velocity of vehicles. </p>", unsafe_allow_html=True)
    # bootstrap 4 collapse example

    components.html(
        """
        <div class="flourish-embed flourish-bar-chart-race" data-src="visualisation/6687631"><script src="https://public.flourish.studio/resources/embed.js"></script></div>
        """,
        height=700,
    )
    components.html(
        """
        <div class="flourish-embed" data-src="story/921147"><script src="https://public.flourish.studio/resources/embed.js"></script></div>
        """, height=850,
    )


# --------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------ Summary of Statistics ends here
# --------------------------------------------------------------------------------------------------------------------------------------------------


if option == "Time-Series Solutions":
    sol_type = st.radio("Select from below", ("Prediction of Number of Death from Road Accidents", "Prediction of Chances of Death Caused by Road Accidents"))

    if sol_type == "Prediction of Number of Death from Road Accidents":

        st.subheader('Time-Series Forecast with LSTM')

        df = pd.read_csv('Dhaka_Tribune_TS.csv')
        df['Period'] = pd.to_datetime(df['Period'])
        df["Date"] = df['Period'].dt.date

        df = df[['Date', 'Deaths']]

        start_ = df.Date.max()
        end_ = df.Date.min()

        test_end = df.Date.max()
        test_start = df.Date.max() - timedelta(days = 60)

        train_end = test_start - timedelta(days = 1)
        train_start = train_end - timedelta(days = 365*5)
      
        st.text(' ')
        st.text('Training Period: `%s` to `%s`' % (end_, train_end))
        st.text('Forecast Period: `%s` to `%s`' % (test_start, test_end))

        mask = (df.Date > train_start) & (df.Date <= train_end)
        train = df.loc[mask]

        mask = (df.Date >= test_start) & (df.Date <= test_end)
        test = df.loc[mask]

        d = pd.concat([train, test])
        d = d.reset_index(drop = True)

        def series_to_supervised(data, window=1, lag=1, dropnan=True):
            cols, names = list(), list()
            # Input sequence (t-n, ... t-1)
            for i in range(window, 0, -1):
                cols.append(data.shift(i))
                names += [('%s(t-%d)' % (col, i)) for col in data.columns]
            # Current timestep (t=0)
            cols.append(data)
            names += [('%s(t)' % (col)) for col in data.columns]
            # Target timestep (t=lag)
            cols.append(data.shift(-lag))
            names += [('%s(t+%d)' % (col, lag)) for col in data.columns]
            # Put it all together
            agg = pd.concat(cols, axis=1)
            agg.columns = names
            # Drop rows with NaN values
            if dropnan:
                agg.dropna(inplace=True)
            return agg

        window = 2
        lag = 1
        series = series_to_supervised(d.drop('Date', axis=1), window=window, lag=lag)

        lag_size = 1 

        # Label
        labels_col = 'Deaths(t+%d)' % lag_size
        labels = series[[labels_col]]
        series = series.drop(labels_col, axis=1)

        X_train = series[:len(train)- window]
        Y_train = labels[:len(train)- window]
        X_valid = series[len(train)- window:]
        Y_valid = labels[len(train)- window:]

        Y_train = np.asarray(Y_train)
        Y_valid = np.asarray(Y_valid)

        X_train_series = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_valid_series = X_valid.values.reshape((X_valid.shape[0], X_valid.shape[1], 1))

        epochs = 500
        batch = 4
        lr = 0.0008
        adam = Adam(lr)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=70)
        mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
        rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=70, verbose=1)

        model_lstm = Sequential()
        model_lstm.add(LSTM(50, activation='relu', input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
        model_lstm.add(Dense(1))
        model_lstm.compile(loss='mae', optimizer=adam)

        lstm_history = model_lstm.fit(X_train_series, Y_train, validation_data=(X_valid_series, Y_valid), epochs=epochs, verbose=1, callbacks=[es, mcp, rlr])

        lstm_valid_pred = model_lstm.predict(X_valid_series)
        lstm_train_pred = model_lstm.predict(X_train_series)

        comparison = test.copy()
        comparison = comparison[:len(test)-lag]
        comparison['Prediction'] = lstm_valid_pred
        comparison['Prediction'] = round(comparison['Prediction'])
        comparison = comparison.set_index('Date')
        st.write(comparison)

        comparison_train = train.copy()
        comparison_train = comparison_train[window:]
        comparison_train['Prediction'] = lstm_train_pred
        comparison_train['Prediction'] = round(comparison_train['Prediction'])
        comparison_train = comparison_train.set_index('Date')

        plot_comparison(lstm_history.history['loss'], lstm_history.history['val_loss'], comparison_train['Deaths'], comparison_train['Prediction'], comparison['Deaths'], comparison['Prediction'], 'LSTM')

        # Define a function to calculate MAE and RMSE
        
        errors = comparison['Deaths'] - comparison['Prediction']
        mse = np.square(errors).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(errors).mean()
        mape = np.abs(errors/comparison['Deaths']).mean() * 100
        r2 = r2_score(comparison['Deaths'], comparison['Prediction'])

        st.subheader('Evaluation Metrics')
        st.text(' ')
        st.text('Mean Absolute Error (MAE): {:.4f}'.format(mae))
        st.text('Mean Squared Error (MSE): {:.4f}'.format(mse))
        st.text('Root Mean Square Error (RMSE): {:.4f}'.format(rmse))
        st.text('Mean Absolute Percentage Error (MAPE): {:.4f}'.format(mape))
        st.text('R2 Score: {:.4f}'.format(r2))


# --------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------- Time Series Forecast ends here
# --------------------------------------------------------------------------------------------------------------------------------------------------
    

    if sol_type == "Prediction of Chances of Death Caused by Road Accidents":
        st.text("here goes Mithilesh's PyCaret-Power-BI integration")

# --------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------- Time Series Classification ends here
# --------------------------------------------------------------------------------------------------------------------------------------------------


if option == "Geographic Heatmaps":
    news_type = st.radio("Newspaper Selection", ("Dhaka Tribune", "Daily Observer"))
    if news_type == "Dhaka Tribune":

        df = pd.read_csv('Dhaka_Tribune_hm.csv')
        df_new = df.copy()

        df_new = df_new.dropna()
        folium_static(map_(df_new))

        df_new["Weight"] = df_new['Accident date'].astype(str)
        df_new["Weight"] = df_new["Weight"].str[5:7]
        df_new["Weight"] = df_new["Weight"].astype(float)
        
        
        df_new['year'] = pd.DatetimeIndex(df_new['Accident date']).year
        df_new['month'] = pd.DatetimeIndex(df_new['Accident date']).month

        import datetime
        lista_tempo = [] 

        for x in df_new['month']: 
            monthinteger = x 
            lista_tempo.append(datetime.date(1900, monthinteger, 1).strftime('%B')) 
    
        df_new['months_in_full'] = lista_tempo 
        df_new['month_year'] = [d.split('-')[1] + " " + d.split('-')[0] for d in df_new['Accident date'].astype(str)]

        df_new['indexx'] = df_new['months_in_full'] + ' ' + df_new['year'].astype(str)
        lista_index = df_new['indexx'].unique().tolist()

        weight_list = []

        df_new['conta'] = 1 
        for x in df_new['month_year'].sort_values().unique(): 
            weight_list.append(df_new.loc[df_new['month_year'] == x, 
                                                ['Latitude',"Longitude",'conta']].groupby(['Latitude','Longitude']).sum().reset_index().values.tolist()) 

            
        base_map = folium.Map(location=[23.6850, 90.3563],tiles="stamen toner",zoom_start = 6) 

        #create the heatmap from our List 
        HeatMapWithTime(weight_list, radius=15,index= lista_index, gradient={0.1: 'blue',0.25: 'green', 0.25: 'yellow', 0.95: 'orange', 1: 'red'}, \
                                                                             
                                auto_play =True, min_opacity=0.5, max_opacity=1, use_local_extrema=True).add_to(base_map) 
                                                                        
                                                                             
        folium_static(base_map)

    if news_type == "Daily Observer":
        df = pd.read_csv('Daily_Observer_hm.csv')
        df_new = df.copy()
        df_new = df_new.dropna()
        folium_static(map_obs())

        df_new['month'] = df_new.date.str[3:6]
        month_num = []
        for i in range(len(df_new)):
            month_num.append(strptime(df_new['month'][i],'%b').tm_mon)
        df_new['month_num'] = month_num
        df_new["Weight"] = df_new['month_num'].astype(float)    

        import datetime
        lista_tempo = [] 

        for x in df_new['month_num']: 
            monthinteger = x 
            lista_tempo.append(datetime.date(1900, monthinteger, 1).strftime('%B')) 
    
        df_new['months_in_full'] = lista_tempo 
        df_new['month_year'] = df_new['month'] + ' ' + df_new['Year'].astype(str)   

        df_new['indexx'] = df_new['month'] + ' ' + df_new['Year'].astype(str)
        lista_index = df_new['indexx'].unique().tolist()

        weight_list = []

        df_new['conta'] = 1 

        for x in df_new['month_year'].sort_values().unique(): 
            weight_list.append(df_new.loc[df_new['month_year'] == x, 
                                        ['Latitude',"Longitude",'conta']].groupby(['Latitude','Longitude']).sum().reset_index().values.tolist())  
    
        base_map = folium.Map(location=[23.6850, 90.3563],tiles="stamen toner",zoom_start = 6) 

        #create the heatmap from our List 
        HeatMapWithTime(weight_list, radius=20,index= lista_index, gradient={0.1: 'blue',0.5: 'green', 0.5: 'yellow', 0.95: 'orange', 1: 'red'}, \
                                                                     
                        auto_play =True, min_opacity=0.5, max_opacity=1, use_local_extrema=True).add_to(base_map) 
                                                                
                                                                     
        folium_static(base_map)

# --------------------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------- Geographic Heatmaps ends here
# --------------------------------------------------------------------------------------------------------------------------------------------------

if option == "Text Analytics Results":

    st.subheader('Results Obtained from The Daily Observer')

    df = pd.read_csv('Datasets/The_Daily_Observer.csv')
    df = df[['News']]
    df.columns = ['full_text']
    df = df.dropna()
    df_ = df.copy()

    def sent_to_words(sentences):
        for sent in sentences:
            sent = re.sub('\S*@\S*\s?', '', sent)  # remove emails
            sent = re.sub('\s+', ' ', sent)  # remove newline chars
            sent = re.sub("\'", "", sent)  # remove single quotes
            sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
            yield(sent)  

    # Convert to list
    data = df['full_text'].values.tolist()
    data_words = list(sent_to_words(data))

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    # !python3 -m spacy download en  # run in terminal once
    def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

        import nltk
        texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
        texts = [bigram_mod[doc] for doc in texts]
        texts = [trigram_mod[bigram_mod[doc]] for doc in texts]
        texts_out = []
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        # remove stopwords once more after lemmatization
        texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
        return texts_out

    data_ready = process_words(data_words)

    col_name = 'full_text'
    from textblob import TextBlob
    df['polarity'] = df[col_name].apply(lambda x: TextBlob(x).polarity)

    type_task = st.radio("Select", ("Show Results", "Show Bigrams & Trigrams", "Show Sentiment Polarity & Subjectivity"))

    if type_task == "Show Results":

        import nltk
        from nltk.corpus import stopwords
        stop_words = stopwords.words('english')
        su_1 = setup(data = df_, target = 'full_text', custom_stopwords=stop_words, session_id=21)
        m1 = create_model(model='lda', multi_core=True)
        lda_data = assign_model(m1)
        st.write(lda_data)
        tmp_download_link = download_link(lda_data, 'hdp.csv', 'Click here to download as CSV')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
        plot_model(m1, plot='topic_distribution', display_format = 'streamlit')
        plot_model(m1, plot = 'tsne', display_format= 'streamlit')
        #plot_model(m1, plot='topic_model')
        

    if type_task == "Show Bigrams & Trigrams":

        from nltk.corpus import stopwords
        stoplist = stopwords.words('english')
        from sklearn.feature_extraction.text import CountVectorizer
                           
        c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2,2))
        # matrix of ngrams
        bgrams = c_vec.fit_transform(df['full_text'])
        # count frequency of ngrams
        count_values = bgrams.toarray().sum(axis=0)
        # list of ngrams
        vocab = c_vec.vocabulary_
        df_bgram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'bigram/trigram'})

        _bgram = df_bgram.head(50)

        import plotly.express as px
        fig = px.bar(_bgram, x='bigram/trigram', y='frequency')
        fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
        fig.update_layout(title_text="Frequency of Bigrams", xaxis_title="Bigrams", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

        from nltk.corpus import stopwords
        stoplist = stopwords.words('english')
        from sklearn.feature_extraction.text import CountVectorizer
                           
        c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(3,3))
        # matrix of ngrams
        tgrams = c_vec.fit_transform(df['full_text'])
        # count frequency of ngrams
        count_values = tgrams.toarray().sum(axis=0)
        # list of ngrams
        vocab = c_vec.vocabulary_
        df_tgram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
            ).rename(columns={0: 'frequency', 1:'bigram/trigram'})

        _tgram = df_tgram.head(50)

        import plotly.express as px
        fig = px.bar(_tgram, x='bigram/trigram', y='frequency')
        fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
        fig.update_layout(title_text="Frequency of Trigrams", xaxis_title="Trigrams", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

    if type_task == "Show Sentiment Polarity & Subjectivity":

        from nltk.corpus import stopwords
        stoplist = stopwords.words('english')
        from sklearn.feature_extraction.text import CountVectorizer

        from sklearn.feature_extraction.text import CountVectorizer
        c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2,3))
        # matrix of ngrams
        ngrams = c_vec.fit_transform(df['full_text'])
        # count frequency of ngrams
        count_values = ngrams.toarray().sum(axis=0)
        # list of ngrams
        vocab = c_vec.vocabulary_
        df_ngram = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
                    ).rename(columns={0: 'frequency', 1:'bigram/trigram'})
        
        df_ngram['polarity'] = df_ngram['bigram/trigram'].apply(lambda x: TextBlob(x).polarity)
        df_ngram['subjective'] = df_ngram['bigram/trigram'].apply(lambda x: TextBlob(x).subjectivity)
        pol = df_ngram.head(200)

        import plotly.express as px
        fig = px.bar(pol, x='bigram/trigram', y='polarity')
        fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                          marker_line_width=1.5, opacity=0.6)
        fig.update_layout(title_text="Polarity of Bigrams & Trigrams", xaxis_title="Bigrams & Trigrams", yaxis_title="Sentiment Polarity")
        st.plotly_chart(fig, use_container_width=True)

        import plotly.express as px
        fig = px.bar(pol, x='bigram/trigram', y='subjective')
        fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
        fig.update_layout(title_text="Subjectivity of Bigrams & Trigrams", xaxis_title="Bigrams & Trigrams", yaxis_title="Subjectivity")
        st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------- Text Analytics ends here
# --------------------------------------------------------------------------------------------------------------------------------------------------


if option == 'Explore Our Awesome Datasets':

    st.subheader('The Daily Observer')

    df1 = pd.read_csv('Datasets/The_Daily_Observer.csv')
    st.write(df1)

    tmp_download_link = download_link(df1, 'daily_observer.csv', 'Click here to download as CSV')
    st.markdown(tmp_download_link, unsafe_allow_html=True)

    st.subheader('Dhaka Tribune')

    df2 = pd.read_csv('Datasets/Dhaka_Tribune.csv')
    st.write(df2)

    tmp_download_link = download_link(df2, 'dhaka_tribune.csv', 'Click here to download as CSV')
    st.markdown(tmp_download_link, unsafe_allow_html=True)

    st.subheader('Samakal')

    df3 = pd.read_csv('Datasets/Samakal.csv')
    st.write(df3)

    tmp_download_link = download_link(df3, 'samakal.csv', 'Click here to download as CSV')
    st.markdown(tmp_download_link, unsafe_allow_html=True)

    st.subheader('United News Bangladesh')

    df4 = pd.read_csv('Datasets/UNB.csv')
    st.write(df4)

    tmp_download_link = download_link(df4, 'unb.csv', 'Click here to download as CSV')
    st.markdown(tmp_download_link, unsafe_allow_html=True)

    st.subheader('New Age BD')

    df5 = pd.read_csv('Datasets/New_Age_BD.csv')
    st.write(df5)

    tmp_download_link = download_link(df5, 'newage_bd.csv', 'Click here to download as CSV')
    st.markdown(tmp_download_link, unsafe_allow_html=True)


    st.subheader('The Daily Star')

    df6 = pd.read_csv('Datasets/The_Daily_Star.csv')
    st.write(df6)

    tmp_download_link = download_link(df6, 'daily_star.csv', 'Click here to download as CSV')
    st.markdown(tmp_download_link, unsafe_allow_html=True)

    st.subheader('The Daily Sun')

    df7 = pd.read_csv('Datasets/The_Daily_Sun.csv')
    st.write(df7)

    tmp_download_link = download_link(df7, 'daily_sun.csv', 'Click here to download as CSV')
    st.markdown(tmp_download_link, unsafe_allow_html=True)

    st.subheader('Prothom Alo')

    df8 = pd.read_csv('Datasets/Prothom_Alo.csv')
    st.write(df8)

    tmp_download_link = download_link(df8, 'prothom_alo.csv', 'Click here to download as CSV')
    st.markdown(tmp_download_link, unsafe_allow_html=True)

    st.subheader('Citation')

    st.write('If you use these datasets, please cite using the following:')

    st.success('Omdena Bangladesh Chapter. Bangladesh Road Accidents Datasets (2016-2021), August 2021. \n\n URL: https://www.linkedin.com/company/omdena-bd-chapter/')