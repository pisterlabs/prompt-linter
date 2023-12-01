import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import gensim
from gensim.models import CoherenceModel
import numpy as np

class Utils:
    def plot_cloud(self,wordcloud):
        # Set figure size
        plt.figure(figsize=(40, 30))
        # Display image
        plt.imshow(wordcloud) 
        # No axis details
        plt.axis("off")
        #plt.show()
        #plt.savefig('./out/' + company +  '/_wordcloud.png')
    
    def plot_line(self,x,y):
        # Set figure size
        plt.plot(x, y)
        plt.xlabel("Num Topics")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize = 14)
        #plt.show()
        #Función para encontrar los parametros optimos del modelo k, alpha, beta
    def compute_coherence_values(self,df,corpus, dictionary, k, a, b):
        
        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                            id2word=dictionary,
                                            num_topics=k, 
                                            random_state=100,
                                            chunksize=100,
                                            passes=10,
                                            alpha=a,
                                            eta=b)
        coherence_model_lda = CoherenceModel(model=lda_model, texts=df['clean_text'], dictionary=dictionary, coherence='c_v')
        per=lda_model.log_perplexity(corpus)
        return coherence_model_lda.get_coherence(),per
        

    def general_cloud(self,data,stop_words):
        textos = ",".join(list(data['document'].values))
        wordcloud = WordCloud(width= 3000, height = 2000, random_state=1, 
                        background_color='salmon', colormap='Pastel1', 
                        collocations=False, stopwords = stop_words).generate(textos)
        return wordcloud


    def optimize_model(self,utils,data,corpus,dictionary,min_topics,max_topics):
        grid = {}



        step_size = 1
        topics_range = range(int(min_topics), int(max_topics), step_size)

        # Alpha parameter
        alpha = list(np.arange(0.1,0.5, 0.1))
        #alpha.append('symmetric')
        #alpha.append('asymmetric')

        # Beta parameter
        beta = list(np.arange(0.2, 0.5, 0.1))
        #beta.append('symmetric')

        model_results = {'Topics': [],
                        'Alpha': [],
                        'Beta': [],
                        'Coherence': [],
                        'Perplejidad': []
                        }
        # iterate through number of topics
        for k in topics_range:
        # iterate through alpha values
            for a in alpha:
        # iterare through beta values
                for b in beta:
                    cv = utils.compute_coherence_values(df = data,corpus=corpus, dictionary=dictionary,k=k, a=a, b=b)
                    print('funcionando')
                    # Save the model results
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv[0])
                    model_results['Perplejidad'].append(cv[1])  

        Resultados=pd.DataFrame(model_results)
        print(Resultados[Resultados['Coherence'] == Resultados['Coherence'].max()])

        return (Resultados)
    
    def format_topics_sentences(self,ldamodel, corpus, texts):
        # Init output
        sent_topics_df = pd.DataFrame()

        # Get main topic in each document
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            # Get the Dominant topic, Perc Contribution and Keywords for each document
            for j, (topic_num, prop_topic) in enumerate(row):
                if j == 0:  # => dominant topic
                    wp = ldamodel.show_topic(topic_num)
                    topic_keywords = ", ".join([word for word, prop in wp])
                    sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
                else:
                    break
        sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

        # Add original text to the end of the output
        contents = pd.Series(texts)
        sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
        return(sent_topics_df)
    
    def jaccard_similarity(self,query, document):
        intersection = set(query).intersection(set(document))
        union = set(query).union(set(document))
        return len(intersection)/len(union)
    
    def split_tesauros(self,tesauros):
        xss =[x.split('_') for x in tesauros['Palabras']]
        flat_list = [x for xs in xss for x in xs] 
        tesauro = pd.DataFrame(flat_list,columns = ["Palabras"])
        tesauro['Palabras'] = tesauro['Palabras'].drop_duplicates()
        tesauro = tesauro[tesauro['Palabras'].notna()]
        return(tesauro)
    
    def clean_tesauros(self,utils ,tesauros, capacidades):
        list_cap = []
        for cap in capacidades:
            tesauro_cap = tesauros[tesauros['Nom_Tesauro']==cap]
            tesauro_cap = utils.split_tesauros(tesauro_cap)
            tesauro_cap['Nom_tesauro'] = cap
            list_cap.append(tesauro_cap)
        return(pd.concat(list_cap))
    
