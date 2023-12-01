#!/usr/bin/env python

'''
Import libraries
'''
# standard library
import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(".."))
from pprint import pprint

# data and nlp
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# visualisation
import pyLDAvis.gensim
import seaborn as sns
from matplotlib import rcParams
# figure size in inches
rcParams['figure.figsize'] = 20,10


# LDA tools
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from utils import lda_utils

# warnings
import logging, warnings
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)



'''
Main function
'''
def main():
    data = pd.read_csv("../data/r_wallstreetbets_posts.csv")

    data = data[["title","created_utc","score"]].sample(10000) #Running a sample for 10000 out of the 1.1 mio titles

    print("reading the data")


    #This loop creates all titles as strings which then is appended to output
    nlp.max_length = len(data["title"])
    output = []

    for title in data["title"]: 
        doc = nlp(title)
        output.append(str(doc))
        print(doc)

    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(output, min_count=3, threshold=20) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[output], threshold=20)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)

    data_processed = lda_utils.process_words(output,nlp, bigram_mod, trigram_mod, allowed_postags=["NOUN"])

    id2word = corpora.Dictionary(data_processed)

    corpus = [id2word.doc2bow(text) for text in data_processed]

    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=3, 
                                           random_state=100,
                                           chunksize=10,
                                           passes=10,
                                           iterations=100,
                                           per_word_topics=True, 
                                           minimum_probability=0.0)

    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                         texts=data_processed, 
                                         dictionary=id2word, 
                                         coherence='c_v')

    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    pprint(lda_model.print_topics())

    # Can take a long time to run.
    model_list, coherence_values = lda_utils.compute_coherence_values(texts=data_processed,
                                                                      corpus=corpus, 
                                                                      dictionary=id2word,  
                                                                      start=5, 
                                                                      limit=40,  
                                                                      step=5)
    plt.savefig("../output/cohorence.png")
    df_topic_keywords = lda_utils.format_topics_sentences(ldamodel=lda_model, 
                                                          corpus=corpus, 
                                                          texts=data_processed)

    # Format
    df_dominant_topic = df_topic_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic.sample(3)

    # Display setting to show more characters in column
    pd.options.display.max_colwidth = 100

    sent_topics_sorteddf = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf = pd.concat([sent_topics_sorteddf, 
                                          grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                          axis=0)

    # Reset Index    
    sent_topics_sorteddf.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]

    # Show
    sent_topics_sorteddf.head(10)

    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    # Save the board as a html-file
    pyLDAvis.save_html(vis, "../output/lda-board_wallStreetBets.html")
    
if __name__=="__main__":
        main()
