##
 # cluster.py
 #
 # Niyati Bafna
 #
 # TFIDF from BoW, LDA, LSA for (3, 10) topics and their respective cv and umass coherence values
 # Saves the dictionary, and tfidf and model objects in saved/
 ##

import numpy as np
import gensim
from gensim import corpora, models
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from numpy import array

N_DOCUMENTS = 517402
processed_emails = []

for i in range(1, N_DOCUMENTS):
    with open("clean/" + str(i) + ".txt", "r") as inf:
        for line in inf:
            processed_emails.append(line.split(" "))

print("Creating bag of words from clean emails...")
dictionary = gensim.corpora.Dictionary(processed_emails)
dictionary.filter_extremes(no_above=0.5)
bag_of_words = [dictionary.doc2bow(email) for email in processed_emails]
dictionary.save("saved/emails.dictionary")

print("Tf-idf")
from gensim import corpora, models
tfidf_object = models.TfidfModel(bag_of_words)
tfidf_vectors = tfidf_object[bag_of_words]
tfidf_object.save("saved/emails.tfidf")

print("LDA & LSA model gen + coherence")
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LdaMulticore
from gensim.models import LsiModel
j_index = 0
for top in range(3, 10):
    lda_model = LdaMulticore(tfidf_vectors, num_topics=top, id2word=dictionary)
    lda_coherence_cv = CoherenceModel(model=lda_model, texts = processed_emails, dictionary=dictionary, coherence='c_v')
    lda_coherence_umass = CoherenceModel(model=lda_model, texts = processed_emails, dictionary=dictionary, coherence='u_mass')

    lD_name = "saved/models/LDA/lda" + str(j_index) + ".model"
    lD_coh_cv = "saved/models/LDA/cv_lda" + str(j_index) + ".coherence"
    lD_coh_um = "saved/models/LDA/umass_lda" + str(j_index) + ".coherence"

    # save the models to the disk
    lda_model.save(lD_name)
    lda_coherence_cv.save(lD_coh_cv)
    lda_coherence_umass.save(lD_coh_um)


    lsa_model = LsiModel(tfidf_vectors,num_topics=top, id2word=dictionary) 
    lsa_coherence_cv = CoherenceModel(model=lsa_model, texts = processed_emails, dictionary=dictionary, coherence='c_v')
    lsa_coherence_umass = CoherenceModel(model=lsa_model, texts = processed_emails, dictionary=dictionary, coherence='u_mass')
    
    lS_name = "saved/models/LSA/lsa" + str(j_index) + ".model"
    lS_coh_cv = "saved/models/LSA/cv_lsa" + str(j_index) + ".coherence"
    lS_coh_um = "saved/models/LSA/umass_lsa" + str(j_index) + ".coherence"

    lsa_model.save(lS_name)
    lsa_coherence_cv.save(lS_coh_cv)
    lsa_coherence_umass.save(lS_coh_um)

    j_index = j_index+1