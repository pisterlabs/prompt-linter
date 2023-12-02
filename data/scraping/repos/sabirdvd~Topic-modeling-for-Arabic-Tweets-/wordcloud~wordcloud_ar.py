


import pandas as pd
from bertopic import BERTopic
from flair.embeddings import TransformerDocumentEmbeddings
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
from gensim.models import LdaMulticore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# pip install wordcloud
# pip install arabic-reshaper
# pip install arabic-reshaper

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bidi.algorithm import get_display
from arabic_reshaper import reshape

import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv("arab_gen_twitter.csv")

data.head()

# shape  
data.shape
data = data.dropna()
documents = data['text'].values

model = BERTopic.load("model_twitter_all_v3")


def create_wordcloud(model, topic):
    rtl = lambda w: get_display(reshape(f'{w}'))
    text = {rtl(word): value for word, value in model.get_topic(topic)}
    font_file ='NotoNaskhArabic-Regular.ttf'
    wc = WordCloud(font_path=font_file, background_color="white", max_words=10)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
    
# Show wordcloud
create_wordcloud(model, topic=200)
plt.savefig('wordcloud.png')
