# standard library
import sys,os
sys.path.append(os.path.join(".."))
from pprint import pprint
from matplotlib import pyplot as plt

# data and nlp
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# visualisation
import pyLDAvis.gensim
#pyLDAvis.enable_notebook()
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


#deffining the main function.
def main():
    #Importing data from kaggle. A dataset from reddit.
    filename = os.path.join("data", "r_wallstreetbets_posts.csv")

    #Making into dataframe
    DATA = pd.read_csv(filename, index_col=0)
    
    
    
    
    #Creating a data frame, that only consists of the following columns:
    data = DATA[["title", "score", "created_utc"]]
    
    
    # I have made the sample here smaller so the code want take to long to run.
    data_sample = data[:10000]
    
    
    #Container for the output of the forloop below
    output = []
    #Running through the title column of the dataset, transforming every title to a doc format and pushes it into the output list.
    for title in data_sample["title"]:
        doc = nlp(title)
        # The code does not work unless I append the doc object as a str.
        output.append(str(doc))
        
    
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(output, min_count=3, threshold=100)
    trigram = gensim.models.Phrases(bigram[output], threshold=100)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    data_processed = lda_utils.process_words(output, #The chunks from the output list above
                                             nlp, # Using nlp
                                             bigram_mod, #Fitting model onto bigrams
                                             trigram_mod, #Fitting model onto trigrams
                                             allowed_postags=["NOUN"]) #only finding nouns
    
    
    
    # Create Dictionary
    id2word = corpora.Dictionary(data_processed)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_processed]
    
    
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus, # a list of lists.
                                           id2word=id2word, # The gensim dictionary
                                           num_topics=4, #I choose 4 topics
                                           random_state=100,
                                           chunksize=100,
                                           passes=10, #Goes through the data 10 times
                                           iterations=100, #A messure of how often it goes over a document.
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
    
    
    
    df_topic_keywords = lda_utils.format_topics_sentences(ldamodel=lda_model, 
                                                      corpus=corpus, 
                                                      texts=data_processed)

    # Format
    df_dominant_topic = df_topic_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    df_dominant_topic.sample(10)
    
    
    
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
    pyLDAvis.save_html(vis, "output/wallstreet_reddit.html")
    
    
    values = list(lda_model.get_document_topics(corpus))
    
    
    
    split = []
    for entry in values:
        topic_prevelance = []
        for topic in entry:
            topic_prevelance.append(topic[1])
        split.append(topic_prevelance)
        
    df = pd.DataFrame(map(list,zip(*split)))
    
    
    sns.lineplot(data=df.T.rolling(50).mean())
    plt.show()
    
    plt.savefig("output/lineplot.png")
    
if __name__ =='__main__':
    main()