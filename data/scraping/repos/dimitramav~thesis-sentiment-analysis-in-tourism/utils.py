import csv
import string
import re
import pickle
import numpy
import pandas as pd
import statistics 
import ast
from statistics import mode 
from nltk import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords
from langdetect import detect
import gensim
import gensim.corpora as corpora
import mtranslate
import operator
import seaborn as sns
import folium
from folium.plugins import FastMarkerCluster
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from stop_words import get_stop_words
from gensim.models import Word2Vec
from sklearn import svm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from operator import itemgetter
from collections import Counter
from wordcloud import WordCloud
from matplotlib.pyplot import *
from collections import Counter
from langdetect import detect




#remove non-english reviews 
def en_reviews(csvfile,noneng): 
    with open(csvfile) as inp_file ,open('en_reviews.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        for row in csv.reader(inp_file, delimiter=','):
            if not row[1] in noneng:
                writer.writerow(row)
    return	

def make_bigrams(texts):
	bigram = gensim.models.Phrases(texts, min_count=5, threshold=100)
	bigram_mod = gensim.models.phrases.Phraser(bigram)
	return [bigram_mod[doc] for doc in texts]

#remove links, symbols like '!','@' and '#'. Plus convert to small case
def clean_symbols(review):
	translate_table = dict((ord(char), None) for char in string.punctuation)
	review = re.sub(r"http\S+", "", review)  # remove link
	review = ' '.join([word for word in review.split(' ') if not review.startswith('@')])
	review = ''.join([word for word in review if not word.isdigit()])
	review = review.translate(translate_table)  # remove symbols
	review = review.encode('ascii', 'ignore').decode('ascii') #remove emojis
	review = review.lower()
	return review

def tokenize(cleaned_corpus):
	tokens = [word_tokenize(cleaned_corpus)]
	stop_words = list(get_stop_words('en'))         #About 900 stopwords
	nltk_words = list(stopwords.words('english')) #About 150 stopwords
	stop_words.extend(nltk_words)
	tokens = [word for token in tokens for word in token if word not in stop_words]
	tokens = make_bigrams([tokens])
	return tokens

# Map POS tag to first character lemmatize() accepts
def get_wordnet_pos(word):
	tag = pos_tag([word])[0][1][0].upper()
	tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
	return tag_dict.get(tag, wordnet.NOUN)

def lemmatize(tokens):
	with open('lexicon/common_words') as f:
		common_words = f.read().splitlines()
	lemmatizer = WordNetLemmatizer()
	lemmatized = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens[0]]
	lemmatized = [word for word in lemmatized if word not in common_words]
	return lemmatized

def clean_reviews(reviews_dataset):
	reviews_dataset['clean_text'] = reviews_dataset['review_text'].apply(clean_symbols)
	reviews_dataset = reviews_dataset[reviews_dataset['clean_text'].map(lambda d: len(d)) > 0] #keep only non empty reviews
	reviews_dataset['clean_text'] = reviews_dataset['clean_text'].apply(tokenize)
	reviews_dataset = reviews_dataset[reviews_dataset['clean_text'].map(lambda d: len(d)) > 0] #keep only non empty reviews
	reviews_dataset['clean_text'] = reviews_dataset['clean_text'].apply(lemmatize)
	reviews_dataset = reviews_dataset[reviews_dataset['clean_text'].map(lambda d: len(d)) > 0] #keep only non empty reviews
	return reviews_dataset

def language_det(row):
    try:
        language = detect(row)
    except:
        language = "en"
    return language


def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp][0:3])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


def topic_keywords(ldamodel,corpus,texts):
	df_topic_sents_keywords = format_topics_sentences(ldamodel, corpus, texts)
	df_dominant_topic = df_topic_sents_keywords.reset_index()
	df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
	return df_dominant_topic['Keywords'][:3]

def save_to_pickle(filename, data):
	with open(filename+'.pickle','wb') as handle:
		pickle.dump(data,handle,protocol = pickle.HIGHEST_PROTOCOL)

def create_word_embeddings(tweets):
	model = Word2Vec(tweets, size=200, min_count=1) # size of vector is 200
	model.train(tweets, total_examples=model.corpus_count, epochs=model.epochs)  # train word vectors
	word_embeddings = []
	for tweet in tweets:
		vec = numpy.array(model[tweet[0]])
		for word in tweet[1:]:
			vec = vec + numpy.array(model[word])
		vec = vec / len(tweet)
		word_embeddings.append(vec)
	lower, upper = -1, 1
	word_embeddings = [lower + (upper - lower) * x for x in word_embeddings]
	return word_embeddings



def svm_classification(final_train_corpus, final_test_corpus, Y_train, vectorizer):
	if vectorizer == 'BOW':
		vectorizer = CountVectorizer()
	elif vectorizer == 'TFIDF':
		vectorizer = TfidfVectorizer(min_df = 30)
	X_train = vectorizer.fit_transform(final_train_corpus)
	X_test = vectorizer.transform(final_test_corpus)
	clf = svm.SVC(gamma='scale',class_weight='balanced', C = 0.1)
	clf.fit(X_train, Y_train)
	Y_pred = clf.predict(X_test)
	return Y_pred

def translate_pt(review):
    translate_review = mtranslate.translate(review,'en','auto')
    return translate_review

def prepare_training_set(reviews_df):
	train_reviews = reviews_df['clean_text'].tolist()
	final_train_corpus  = [" ".join(str(word) for word in review) for review in train_reviews]
	return final_train_corpus


def prepare_test_set(dataset):
	testset = dataset['clean_text'].tolist()
	final_test_corpus  = [" ".join(str(word) for word in review) for review in testset]
	return final_test_corpus

def nltk_vader(final_train_corpus):
	nltk_vader_list = []
	sid = SentimentIntensityAnalyzer()
	for sentence in final_train_corpus:
		ss = sid.polarity_scores(sentence)
		ss.pop('compound', None) #compound has maximum value so we have to remove it from dictionary
		sentiment = max(ss, key=ss.get)
		sentiment_list = sorted( ss.items(), key=itemgetter(1),reverse = True) 
		if(abs(sentiment_list[0][1]-sentiment_list[1][1])<0.2): #if the sentiment is vague choose the non neutral sentiment
			if sentiment_list[0][0] == 'neu':
				sentiment = sentiment_list[1][0]

		if sentiment == 'neg':
			nltk_vader_list.append(-1)
		elif sentiment == 'pos':
			nltk_vader_list.append(1)
		else:
			nltk_vader_list.append(0)
	return nltk_vader_list


def dominant_sentiment(bow_list,tfidf_list,vader_list):
	predicted_sentiment = []
	diff_sentiments = []
	for f, b, l in zip(bow_list,tfidf_list,vader_list):
		diff_sentiments=[f,b,l]
		try:
			predicted_sentiment.append(mode(diff_sentiments))
		except:
			predicted_sentiment.append(0)
	return predicted_sentiment


def save_wordcloud(total_text,name):
	total_count = Counter(total_text)
	wordcloud = WordCloud(background_color="white").generate_from_frequencies(dict(total_count))
	wordcloud.to_file(name)
	image = wordcloud.to_image()
	return image

def df_chunks(reviews_dataset,n):
	list_df = [reviews_dataset[i:i+n] for i in range(0,reviews_dataset.shape[0],n)]
	for index,df in enumerate(list_df):
	    path = 'df_chunks/df' + str(index)+'.csv'
	    df.to_csv(path,columns=['id', 'origin','date','review_text','clean_text','rating','location_lat','location_long','venue_category','type','topic','address','sentiment'] ,index=False) #save reviews to file for future use
	return


def single_strings_to_multiple(row):
	row = ast.literal_eval(row)
	row = row[0].split(",")
	row_with_no_spaces = [word if not ' ' in word else word.replace(' ','') for word in row]
	return row_with_no_spaces

def specify_location(row):
    if(row.startswith('[Location')):
        if(detect(row.split(',')[2])=='en'):
            row = row.split(',')[2]
            row = row.replace(',','')
        else:
            row = row.split(',')[3]
            row = row.replace(',','')
    return row

def most_common_topic(number, sentiment,dataset,origin = "all"):
	with open('lexicon/common_words') as f:
		common_words = f.read().splitlines()
	if origin == "all":
		total_topic = dataset.loc[dataset['sentiment'] == sentiment]['topic'].tolist()
	else:
		total_topic = dataset.loc[(dataset['sentiment'] == sentiment) & (dataset['origin'] == origin)]['topic'].tolist()
	topic_counter = Counter([word for topic in total_topic for word in topic if word not in common_words])
	return topic_counter.most_common(number) 

def topic_barplot(dataset,color,sentiment,origin):
	df_dataset = pd.DataFrame(list(dict(dataset).items()),columns=['topic', 'frequency'])
	sns.set(style="whitegrid")
	sns.set(rc={'figure.figsize':(12,9)})
	ax = sns.barplot(x="frequency", y="topic", data=df_dataset,color = color)
	title = 'What are the top 20 most ' + sentiment +' common topics of ' + origin +'?'
	ax.set_title(title,fontsize=20)
	ax.set_xlabel("frequency",fontsize=15,fontweight = "bold")
	ax.set_ylabel("topic",fontsize=15, fontweight = "bold")
	return ax

def satisfaction_rate_pie(dataset,origin="all",venue_type = " "):
	if origin == "all":
		positive_count = dataset[dataset['sentiment'] == 1]['id'].count()
		negative_count = dataset[dataset['sentiment'] == -1]['id'].count()
		neutral_count = dataset[dataset['sentiment'] == 0]['id'].count()
	elif venue_type == " ": 
		positive_count = dataset[(dataset['sentiment'] == 1) & (dataset['origin'] == origin)]['id'].count()
		negative_count = dataset[(dataset['sentiment'] == -1) & (dataset['origin'] == origin)]['id'].count()
		neutral_count = dataset[(dataset['sentiment'] == 0) & (dataset['origin'] == origin)]['id'].count()
	else:
		positive_count = dataset[(dataset['sentiment'] == 1) & (dataset['type'] == venue_type)]['id'].count()
		negative_count = dataset[(dataset['sentiment'] == -1) & (dataset['type'] == venue_type)]['id'].count()
		neutral_count = dataset[(dataset['sentiment'] == 0) & (dataset['type'] == venue_type)]['id'].count()
	sizes = [negative_count, neutral_count, positive_count]
	labels = ['Negative', 'Neutral', 'Positive']
	#colors
	colors = ['#FF6961','#FDFD96','#77DD77']
	#explsion
	explode = (0.05,0.05,0.05)
	ax1 = pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=20, pctdistance=0.85, explode = explode, textprops={'fontsize': 12})
	#draw circle
	centre_circle = Circle((0,0),0.70,fc='white')
	fig = gcf()
	fig.gca().add_artist(centre_circle)
	# Equal aspect ratio ensures that pie is drawn as a circle
	axis('equal') 
	if origin == "all":
		 plot_title = 'General satisfaction rate'
	elif venue_type == " ":
		plot_title = 'Satisfaction rate from ' +origin
	else:
		plot_title = 'Satisfaction rate for ' +venue_type + ' category'
	title(plot_title, fontsize=18,fontweight= 'bold')

	tight_layout()
	if venue_type == " ":
		path = 'plots/satisfaction_rate_pies/' +origin +'_pie.png'
	else:
		path = 'plots/satisfaction_rate_pies/' +venue_type +'_pie.png'
	savefig(path,bbox_inches = 'tight')
	close()
	return

def category_barplot(dataset,type,color):
	reviews = dataset[dataset['type']==type]
	category = dict(reviews.groupby(['venue_category'])['id'].count())
	sorted_category = dict(sorted(category.items(), key=operator.itemgetter(1),reverse = True)[:10])
	df_dataset = pd.DataFrame(list(sorted_category.items()),columns=['venue category', 'frequency'])
	sns.set(style="whitegrid")
	sns.set(rc={'figure.figsize':(12,12	)})
	ax = sns.barplot(x="frequency", y="venue category", data=df_dataset,color = color)
	title = 'What venues are there in ' + type +' category?'
	ax.set_title(title,fontsize=20)
	ax.set_xlabel("frequency",fontsize=15,fontweight = "bold")
	ax.set_ylabel("venue category",fontsize=15, fontweight = "bold")
	return ax

def district_barplot(dataset):
	district = dict(dataset.groupby(['address'])['id'].count())
	sorted_district = dict(sorted(district.items(), key=operator.itemgetter(1),reverse = True)[:10])
	df_dataset = pd.DataFrame(list(sorted_district.items()),columns=['district', 'frequency'])
	sns.set(style="whitegrid")
	sns.set(rc={'figure.figsize':(12,9)})
	ax = sns.barplot(x="frequency", y="district", data=df_dataset,color = 'green')
	title = 'What are the top 10 reviewed districts of Athens?'
	ax.set_title(title,fontsize=20)
	ax.set_ylabel("district",fontsize=15,fontweight = "bold")
	ax.set_xlabel("frequency",fontsize=15, fontweight = "bold")
	return ax

def clusters_interactivemap(maps_dataset):
	callback = ('function (row) {' 
                'var color_dict = {"1":"green","-1":"red","0":"blue"};'
                'var icon_dict = {"accommodation":"glyphicon glyphicon-home","entertainment":"glyphicon glyphicon-heart","nightlife":" glyphicon glyphicon-glass","food":"glyphicon glyphicon-cutlery"};'
                'var marker = L.marker(new L.LatLng(row[0], row[1]));'
                'var icon = L.AwesomeMarkers.icon({'
                "icon: icon_dict[row[4]],"
                "iconColor:'white' ,"
                "markerColor: color_dict[row[3]],"
                "prefix: 'glyphicon',"
                "extraClasses: 'fa-rotate-0'"
                    '});'
                'marker.setIcon(icon);'
                "var popup = L.popup({maxWidth: '300'});"
                "const display_text = {text: row[2]};"
                "var mytext = $(`<div id='mytext' class='display_text' style='width: 100.0%; height: 100.0%;'> ${display_text.text}</div>`)[0];"
                "popup.setContent(mytext);"
                "marker.bindPopup(popup);"
                'return marker};')
	lat_df = maps_dataset['location_lat']
	long_df = maps_dataset['location_long']
	mean_location = [numpy.mean(lat_df.values), numpy.mean(long_df.values)]
	location_df = maps_dataset[['location_lat', 'location_long','topic','sentiment','type']]
	m = folium.Map(location=mean_location,tiles='Cartodb Positron',zoom_start=10,prefer_canvas=True)
	marker_cluster = FastMarkerCluster(data=location_df,overlay=True,control=True, callback=callback)
	marker_cluster.add_to(m)
	folium.LayerControl().add_to(m)
	return m

