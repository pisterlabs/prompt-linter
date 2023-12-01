#######################################################################################
### IMPORT LIBRAIRIES #################################################################
#######################################################################################

import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # visualization
from matplotlib.pyplot import imshow
import seaborn as sns # visualization

### LIBRAIRIES FOR NATURAL LANGUAGE PROCESSING 

# nltk.download('all') # uncomment to download nltk libs

# Visualization
from wordcloud import WordCloud
from PIL import Image

from sklearn import feature_extraction, linear_model, model_selection, preprocessing

# NLP
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import words, stopwords, wordnet
from nltk.tokenize import RegexpTokenizer

from collections import Counter

# Import regex and time
import re
import time

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from gensim.models import Phrases

import emoji

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim_models  # don't skip this

# Disable deprecation warnings
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

stop_words = set(stopwords.words('english'))
stop_words.add('food')
lem = WordNetLemmatizer()
tokenizer = nltk.RegexpTokenizer(r'[a-zA-Z]+')

### LIBRAIRIES FOR COMPUTER VISION

from PIL import Image, ImageOps
import cv2

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import InterclusterDistance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import cluster

from keras.layers import Dense, Flatten
from keras import Model

from sklearn.manifold import TSNE

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import os

# DEEP LEARNING

from glob import glob
import sklearn.metrics as metrics
import tensorflow as tf

import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

from tensorflow.keras.applications import VGG16

#######################################################################################
### NATURAL LANGUAGE PROCESSING #######################################################
#######################################################################################

def text_cleaner(text):

    text = emoji.demojize(text, delimiters=("", "")) # demojize the emojis in the docs

    text = text.lower() # to lowercase
    
    text = tokenizer.tokenize(text) # tokenize with regular expressions

    text = [w for w in text if w not in stop_words] # remove stopwords

    text = [lem.lemmatize(w) for w in text] # lemmatize with WordNetLemmatizer

    text = [w for w in text if len(w) > 3] # keep only words longer than 3 characters

    ### keep only nouns 

    pos_tag = nltk.pos_tag(text)
    noun_text = []

    for i in range (0, (len(text)-1)):

        if (pos_tag[i][1] == 'NN'):
            noun_text.append(pos_tag[i][0])

    text = noun_text

    ###

    return text


def display_tokens_info(tokens):
    print(f"Number of tokens: {len(tokens)}, Number of unique tokens: {len(set(tokens))}")
    print(tokens[:30])


def display_original_corpus_info(data):

    corpus_original = wordpunct_tokenize(" ".join(data.text.values))
    display_tokens_info(corpus_original)

    wordcloud = WordCloud(background_color='white',
                            stopwords=[],
                            max_words=100).generate(" ".join(corpus_original))

    common_words = pd.DataFrame(Counter(corpus_original).most_common(30))
    common_words.columns = ('Word', 'Count')

    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.imshow(wordcloud)
    plt.title("Tokens before preprocessing", fontsize=15, pad=15)
    plt.axis("off")

    plt.subplot(1,2,2)
    sns.barplot(x=common_words['Word'], y=common_words['Count'], color='teal')
    plt.xticks(rotation='vertical', fontsize=12)
    plt.title("Key word count in original corpus", fontsize=15, pad=15)

    plt.suptitle("Corpus before preprocessing", fontsize=15)

    plt.show()


def display_clean_corpus_info(data):

    corpus_clean = np.concatenate(data.clean_text)
    display_tokens_info(corpus_clean)

    wordcloud = WordCloud(background_color='white',
                        stopwords=[],
                        max_words=100).generate(" ".join(corpus_clean))

    common_words = pd.DataFrame(Counter(corpus_clean).most_common(30))
    common_words.columns = ('Word', 'Count')

    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.imshow(wordcloud)
    plt.title("Tokens after preprocessing", fontsize=15, pad=15)
    plt.axis("off")

    plt.subplot(1,2,2)
    sns.barplot(x=common_words['Word'], y=common_words['Count'], color='teal')
    plt.xticks(rotation='vertical', fontsize=12)
    plt.title("Key word count in clean corpus", fontsize=15, pad=15)

    plt.suptitle("Corpus after preprocessing", fontsize=15)

    plt.show()


def get_coherence_data(data):

    data_sample = data.sample(n=1000, replace=False)
    data_sample = data_sample.reset_index()
    data_sample.drop(columns=['index'], inplace=True)

    # Create Dictionary
    id2word = corpora.Dictionary(data_sample.clean_text)

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_sample.clean_text]

    topics = []
    coherence = []

    for t in range(3, 7):

        lda = LdaModel( corpus=corpus, 
                        id2word=id2word, 
                        num_topics=t,
                        alpha=50/t,
                        eta=0.1,
                        per_word_topics=True)

        coherence_model_lda = CoherenceModel(model=lda, texts=data_sample.clean_text, dictionary=id2word, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        
        topics.append(t)
        coherence.append(coherence_lda)

    return topics, coherence



def plot_coherence(topics, coherence):

    ax = sns.lineplot(x=topics, y=coherence)
    ax.set_title("Coherence score vs. number of topics", fontsize=15, pad=15)
    ax.set_xlabel("Topics", fontsize=12)
    ax.set_ylabel("Coherence score (c_v)", fontsize=12)



def print_LDA_topics(data):

    # Create Dictionary
    id2word = corpora.Dictionary(data.clean_text)

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data.clean_text]

    lda = LdaModel(     corpus=corpus, 
                        id2word=id2word, 
                        num_topics=4,
                        chunksize=100, # number of documents to consider at once
                        minimum_probability=0.01, #  minimum probability of word in topic
                        passes=100, # how many times the algorithm passes over the whole corpus
                        alpha=50/4, # 
                        eta=0.1,
                        per_word_topics=True) # every word assigned a topic, otherwise omitted
    topics = lda.print_topics()
    return lda, corpus, id2word, topics



def print_topic_wordclouds(lda):
    nbr_rows = int(lda.num_topics/2) if lda.num_topics % 2 == 0 else int((lda.num_topics+1)/2)
    index = 1

    fig, axs = plt.subplots(nbr_rows, 2, figsize=(15,4*nbr_rows))

    for t in range(lda.num_topics):
        plt.subplot(nbr_rows, 2, index)
        plt.imshow(WordCloud(background_color = 'white').fit_words(dict(lda.show_topic(t, 50))))
        plt.axis("off")
        plt.title("Topic " + str(t))

        index+=1

    plt.show()



#######################################################################################
### COMPUTER VISION ###################################################################
#######################################################################################

def get_sample_imgs(imgs_per_label, dir):
    directory = dir
    df = pd.read_json(directory + "\photos.json", lines=True)
    labels = df.label.unique().tolist()

    sample_df = pd.DataFrame()

    for label in labels:
        label_df = df[df["label"] == label].sample(imgs_per_label, random_state=42)
        sample_df = sample_df.append(label_df, ignore_index=True)

    return sample_df, labels


def show_preprocessing_steps(image):
    plt.figure(figsize=(15,4))

    image_color = image
    image_bw = ImageOps.grayscale(image_color)
    image_eq = ImageOps.equalize(image_bw)

    images = [image_color, image_bw, image_eq]
    titles = ['Original image', 'BW Image', 'Equalized Image']

    images_np = [np.array(i) for i in images]
    titles_hists = ['Original Histogram', 'BW Histogram', 'Equalized Histogram']

    # plot images

    for i in range(len(images)):
        plt.subplot(1,len(images),i+1)
        plt.imshow(images[i],'gray')
        plt.axis("off")
        plt.title(titles[i])
    plt.show()

    plt.figure(figsize=(15,4))
    # plot histograms

    for i in range(len(images_np)):
        plt.subplot(1,len(images),i+1)
        plt.hist(images_np[i].flatten(), bins=range(256))
        plt.title(titles_hists[i])
    plt.show()


def get_keypoints(image):
    image_bw = ImageOps.grayscale(image)
    image_eq = ImageOps.equalize(image_bw)

    image_np = np.array(image_eq)

    orb = cv2.ORB_create(nfeatures=100)
    keypoints = orb.detect(image_np, None)
    keypoints, descriptors = orb.compute(image_np, keypoints)
    return keypoints, descriptors, image_np


def plot_keypoints(image, keypoints):
    image_kp = cv2.drawKeypoints(image, keypoints, None, color=(0,255,0), flags=0)
    plt.imshow(image_kp), plt.show()


def get_batch_descriptors(photo_ids, directory):
    sample_images = []

    for i in range(0,len(photo_ids)):
        img = Image.open(directory + "\\photos\\" + photo_ids[i] + ".jpg")
        img = ImageOps.grayscale(img)
        img = ImageOps.equalize(img)
        img = np.array(img)
        sample_images.append(img)

    descriptors = []
    descriptors_len = []

    orb = cv2.ORB_create(nfeatures=100)

    for img in sample_images:
        kp = orb.detect(img, None)
        kp, des = orb.compute(img, kp)
        descriptors.append(des)

    for i in range(len(photo_ids)):
        descriptors_len.append(np.concatenate(descriptors[i])[:2000])

    return descriptors_len



def tsne_visualizer(descriptors, photos_data):
    pca = PCA(n_components=5, random_state=42)
    feat_pca = pca.fit_transform(descriptors)

    tsne = TSNE(n_components=2, n_iter=500, perplexity=100)
    tsne_results = tsne.fit_transform(feat_pca)

    df_tsne = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
    df_tsne['class'] = photos_data["label"]

    df_tsne['tsne1'] = tsne_results[:,0]
    df_tsne['tsne2'] = tsne_results[:,1]

    plt.figure(figsize=(16,8))

    plt.subplot(1,2,1)
    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue="class",
        palette=sns.color_palette('tab10', n_colors=5),
        data=df_tsne,
        legend="full",
        s=50,
        alpha=0.6
    )
    plt.title('TSNE based on actual image labels', fontsize = 20, pad = 35, fontweight = 'bold')
    plt.xlabel('TSNE 1', fontsize = 15, fontweight = 'bold')
    plt.ylabel('TSNE 2', fontsize = 15, fontweight = 'bold')
    plt.legend(prop={'size': 14})

    cls = cluster.KMeans(n_clusters=5, random_state=42)
    cls.fit(tsne_results)

    df_tsne["cluster"] = cls.labels_

    plt.subplot(1,2,2)
    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue="cluster",
        palette=sns.color_palette('tab10', n_colors=5),
        data=df_tsne,
        legend="full",
        s=50,
        alpha=0.6
    )
    plt.title('TSNE based on KMeans predicted clusters', fontsize = 20, pad = 35, fontweight = 'bold')
    plt.xlabel('TSNE 1', fontsize = 15, fontweight = 'bold')
    plt.ylabel('TSNE 2', fontsize = 15, fontweight = 'bold')
    plt.legend(prop={'size': 14})

    print("ARI : ", adjusted_rand_score(df_tsne['class'], df_tsne['cluster']))
    plt.show()


vgg16 = keras.applications.vgg16
vgg = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

for layer in vgg.layers:
    layer.trainable = False # do not train existing weights

x = Flatten()(vgg.output) # flatten the last layer


def extract_vgg16_features(img_path):

    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))

    img_as_array = tf.keras.utils.img_to_array(img)
    img_as_array = img_as_array.reshape((1, img_as_array.shape[0], img_as_array.shape[1], img_as_array.shape[2]))
    
    img_preprocessed = vgg16.preprocess_input(img_as_array)
    features = vgg.predict(img_preprocessed)
    print(features.shape)



def get_vgg16_features(photo_ids, directory):

    vgg16_feats = []

    for i in range (0,len(photo_ids)):
        img_path = str(directory + "\\photos\\" + photo_ids[i] + ".jpg")
        img = tf.keras.utils.load_img(img_path, target_size=(224, 224, 3))

        x = tf.keras.utils.img_to_array(img)
        xs = np.expand_dims(x, axis=0)
        xs = vgg16.preprocess_input(xs)
        features = vgg.predict(xs)

        vgg16_feats.append(features.flatten())

    return vgg16_feats



def prep_tsne(features, true_labels):
    
    features_np = np.array(features)
    
    pca_vgg = PCA(n_components=0.80, random_state=42)
    feat_pca_vgg = pca_vgg.fit_transform(features_np)

    tsne = TSNE(n_components=2, n_iter=500, perplexity=30, random_state=42)

    tsne_results = tsne.fit_transform(feat_pca_vgg)

    df_tsne = pd.DataFrame(tsne_results, columns=['tsne1', 'tsne2'])
    df_tsne['class'] = true_labels   

    tsne_res = np.array(df_tsne[["tsne1", "tsne2"]])

    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    kmeans.fit(tsne_res)
    df_tsne["cluster"] = kmeans.labels_

    return df_tsne



def tsne_visualizer_vgg(df_tsne):

    print("ARI : ", adjusted_rand_score(df_tsne['class'], df_tsne['cluster']))

    plt.figure(figsize=(16,8))

    plt.subplot(1,2,1)

    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue="class",
        palette=sns.color_palette('tab10', n_colors=5),
        data=df_tsne,
        legend="full",
        s=50,
        alpha=0.6)

    plt.title('TSNE based on actual image labels', fontsize = 20, pad = 35, fontweight = 'bold')
    plt.xlabel('TSNE 1', fontsize = 15, fontweight = 'bold')
    plt.ylabel('TSNE 2', fontsize = 15, fontweight = 'bold')
    plt.legend(prop={'size': 14}) 

    plt.subplot(1,2,2)

    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue="cluster",
        palette=sns.color_palette('tab10', n_colors=5),
        data=df_tsne,
        legend="full",
        s=50,
        alpha=0.6)

    plt.title('TSNE based on KMeans predicted clusters', fontsize = 20, pad = 35, fontweight = 'bold')
    plt.xlabel('TSNE 1', fontsize = 15, fontweight = 'bold')
    plt.ylabel('TSNE 2', fontsize = 15, fontweight = 'bold')
    plt.legend(prop={'size': 14}) 

    plt.show()



def display_mislabeled_pics(true_class, predicted_class, df_tsne, photo_ids):

    df_tsne['photo_id'] = photo_ids

    images = []
    titles = []
    
    for i in range(0,4):
        p_id = df_tsne[(df_tsne["class"]==true_class)&(df_tsne["predicted_class"]==predicted_class)].iloc[i]["photo_id"]
        img = Image.open(directory + "\\photos\\" + p_id + ".jpg")
        # predicted_class = df_tsne[(df_tsne["class"]==true_class)&(df_tsne["predicted_class"]==predicted_class)].iloc[i]["predicted_class"]
        title = str("True class: " + true_class + ", Predicted class: " + predicted_class)
        titles.append(title)
        images.append(img)

    # plot images

    plt.figure(figsize=(16,8))

    for i in range(len(images)):
        plt.subplot(2,2,i+1),plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis("off")
    plt.show()