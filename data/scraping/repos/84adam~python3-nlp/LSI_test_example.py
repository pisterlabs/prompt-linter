#!/usr/bin/python3

# Prepare Corpus in Advance, e.g. Wikipedia Article Corpus
# git clone https://github.com/84adam/python3-nlp.git
# cp python3-nlp/* ./
# pip3 install -r requirements.txt -q

import sys
import os
import nltk
import spacy
import gensim
import sklearn
import keras
import pandas as pd  
import numpy as np
from itertools import chain
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
nltk.download('wordnet') # run once
nltk.download('stopwords') # run once
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
import pyLDAvis.gensim
from keras.preprocessing.text import text_to_word_sequence
from sklearn.feature_extraction import stop_words
import subprocess
import shlex

df = pd.read_pickle('{corpus_name}.pkl')

# df.head()

"""
index | title | raw | stop
0 | https://en.wikipedia.org/wiki/Josephus | titus flavius josephus dʒoʊˈsiːfəs 2 greek φλά... | [titus, flavius, josephus, greek, φλάβιος, 37,...
1 | https://en.wikipedia.org/wiki/Social_identity_... | social identity is the portion of an individua... | [social, identity, portion, individual's, self...
2 | https://en.wikipedia.org/wiki/WAITS | waits was a heavily modified variant of digita... | [waits, heavily, modified, variant, digital, e...
3 | https://en.wikipedia.org/wiki/Prometheus_Bound | prometheus bound ancient greek προμηθεὺς δεσμώ... | [prometheus, bound, ancient, greek, ancient, g...
4 | https://en.wikipedia.org/wiki/Metaphysical_art | metaphysical art italian pittura metafisica wa... | [metaphysical, art, italian, pittura, metafisi...
"""

# Create an LSI model
# See: https://en.wikipedia.org/wiki/Latent_semantic_indexing

# See: https://mimno.infosci.cornell.edu/info6150/readings/dp1.LSAintro.pdf
# Compared to the standard vector method (essentially LSI without dimension 
# reductions) ceteris paribus LSI was a 16% improvement (Dumais, 1994).

# Another challenge to LSI has been the alleged difficulty in determining the
# optimal number of dimensions to use for performing the SVD. As a general rule,
# fewer dimensions allow for broader comparisons of the concepts contained in a 
# collection of text, while a higher number of dimensions enable more specific 
# (or more relevant) comparisons of concepts. The actual number of dimensions 
# that can be used is limited by the number of documents in the collection. 
# Research has demonstrated that around 300 dimensions will usually provide the 
# best results with moderate-sized document collections (hundreds of thousands 
# of documents) and perhaps 400 dimensions for larger document collections 
# (millions of documents).[51] However, recent studies indicate that 50-1000 
# dimensions are suitable depending on the size and nature of the document 
# collection.[52] Checking the proportion of variance retained, similar to PCA 
# or factor analysis, to determine the optimal dimensionality is not suitable 
# for LSI. Using a synonym test or prediction of missing words are two possible 
# methods to find the correct dimensionality. [53] When LSI topics are used as 
# features in supervised learning methods, one can use prediction error 
# measurements to find the ideal dimensionality.

from collections import defaultdict
from gensim import corpora
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim import models

# example: Use 100 topics to train the model
num_topics = int(input("How many topics should be used to train the model? ")) 
lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=num_topics)

"""
How many topics should be used to train the model? 100

2019-12-17 13:59:59,490 : INFO : using serial LSI version on this node
2019-12-17 13:59:59,491 : INFO : updating model with new documents
2019-12-17 13:59:59,495 : INFO : preparing a new chunk of documents
2019-12-17 14:00:32,839 : INFO : using 100 extra samples and 2 power iterations
2019-12-17 14:00:32,956 : INFO : 1st phase: constructing (109720, 200) action matrix
2019-12-17 14:00:36,351 : INFO : orthonormalizing (109720, 200) action matrix
2019-12-17 14:00:57,490 : INFO : 2nd phase: running dense svd on (200, 20000) matrix
2019-12-17 14:00:58,195 : INFO : computing the final decomposition
2019-12-17 14:00:58,197 : INFO : keeping 100 factors (discarding 17.649% of energy spectrum)
2019-12-17 14:00:58,433 : INFO : processed documents up to #20000
2019-12-17 14:00:58,520 : INFO : topic #0(5439.574): 0.159*"new" + 0.145*"used" + 0.127*"time" + 0.124*"century" + 0.114*"world" + 0.113*"would" + 0.108*"states" + 0.103*"system" + 0.099*"use" + 0.099*"state"
2019-12-17 14:00:58,528 : INFO : topic #1(2335.065): -0.200*"used" + 0.192*"war" + -0.159*"system" + -0.131*"use" + 0.126*"empire" + 0.124*"city" + 0.122*"century" + 0.115*"government" + -0.114*"energy" + -0.113*"systems"
2019-12-17 14:00:58,535 : INFO : topic #2(1972.607): 0.393*"music" + 0.201*"century" + -0.144*"government" + 0.129*"art" + -0.118*"united" + -0.117*"education" + -0.113*"states" + -0.111*"state" + 0.111*"church" + -0.108*"university"
2019-12-17 14:00:58,542 : INFO : topic #3(1867.454): -0.275*"social" + -0.193*"education" + -0.181*"university" + -0.154*"school" + 0.153*"used" + 0.146*"energy" + -0.136*"students" + 0.123*"windows" + -0.114*"schools" + -0.113*"law"
2019-12-17 14:00:58,548 : INFO : topic #4(1840.093): 0.638*"music" + 0.152*"new" + -0.109*"church" + 0.102*"musical" + 0.101*"university" + 0.095*"windows" + -0.095*"theory" + -0.091*"god" + -0.090*"roman" + -0.083*"empire"
2019-12-17 14:00:58,552 : INFO : preparing a new chunk of documents
2019-12-17 14:01:21,966 : INFO : using 100 extra samples and 2 power iterations
2019-12-17 14:01:22,027 : INFO : 1st phase: constructing (109720, 200) action matrix
2019-12-17 14:01:24,421 : INFO : orthonormalizing (109720, 200) action matrix
2019-12-17 14:01:41,078 : INFO : 2nd phase: running dense svd on (200, 14249) matrix
2019-12-17 14:01:41,646 : INFO : computing the final decomposition
2019-12-17 14:01:41,647 : INFO : keeping 100 factors (discarding 18.453% of energy spectrum)
2019-12-17 14:01:41,832 : INFO : merging projections: (109720, 100) + (109720, 100)
2019-12-17 14:01:43,035 : INFO : keeping 100 factors (discarding 5.999% of energy spectrum)
2019-12-17 14:01:43,362 : INFO : processed documents up to #34249
2019-12-17 14:01:43,373 : INFO : topic #0(7097.199): 0.158*"new" + 0.146*"used" + 0.126*"time" + 0.123*"century" + 0.114*"world" + 0.112*"would" + 0.109*"states" + 0.103*"system" + 0.101*"use" + 0.099*"state"
2019-12-17 14:01:43,380 : INFO : topic #1(3044.078): 0.201*"used" + -0.183*"war" + 0.157*"system" + -0.135*"empire" + -0.132*"century" + 0.132*"use" + -0.123*"city" + 0.122*"energy" + 0.116*"systems" + -0.111*"government"
2019-12-17 14:01:43,386 : INFO : topic #2(2562.852): 0.223*"education" + -0.201*"century" + 0.196*"university" + 0.161*"school" + 0.154*"students" + 0.138*"government" + -0.132*"music" + -0.127*"church" + -0.127*"roman" + 0.126*"schools"
2019-12-17 14:01:43,392 : INFO : topic #3(2454.754): -0.251*"social" + -0.198*"education" + -0.174*"school" + -0.165*"university" + 0.145*"energy" + -0.142*"students" + -0.125*"schools" + 0.124*"used" + -0.117*"theory" + -0.112*"science"
2019-12-17 14:01:43,398 : INFO : topic #4(2284.176): 0.587*"music" + -0.159*"law" + 0.134*"new" + -0.131*"social" + 0.130*"university" + 0.119*"art" + -0.107*"state" + 0.106*"education" + 0.103*"school" + 0.103*"musical"
"""

# cosine similarity to determine the similarity of two vectors. 
# See: https://en.wikipedia.org/wiki/Cosine_similarity

# Cosine similarity is a standard measure in Vector Space Modeling, but 
# ...wherever the vectors represent probability distributions, different 
# ...similarity measures may be more appropriate.

# See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Symmetrised_divergence
# Symmetrised divergence
# Kullback and Leibler themselves actually defined the divergence as:
# [omitted] 
# which is symmetric and nonnegative. 
# This quantity has sometimes been used for feature selection in 
# classification problems, where P and Q are the conditional pdfs of a feature 
# under two different classes. In the Banking and Finance industries, this 
# quantity is referred to as Population Stability Index, and is used to assess 
# distributional shifts in model features through time.

# An alternative is given via the lambda divergence,
# [omitted] 
# which can be interpreted as the expected information gain about X from 
# discovering which probability distribution X is drawn from, P or Q, if they 
# currently have probabilities [omitted]
# The value lambda=0.5 gives the Jensen–Shannon divergence, defined by
# [omitted]
# where M is the average of the two distributions,
# [omitted]
# Jensen-Shannon Divergence can also be interpreted as the capacity of a noisy
# information channel with two inputs giving the output distributions P and Q. 
# The Jensen–Shannon divergence, like all f-divergences, is locally proportional
# to the Fisher information metric. It is similar to the Hellinger metric (in 
# the sense that induces the same affine connection on a statistical manifold).

# Initializing query structures
# To prepare for similarity queries, we need to enter all documents which we
# want to compare against subsequent queries. In our case, they are the same 
# documents used for training LSI, converted to 2-D LSA space. But that’s only 
# incidental, we might also be indexing a different corpus altogether.

from gensim import similarities
# transform corpus to LSI space and index it
index = similarities.MatrixSimilarity(lsi[corpus])

"""
2019-12-17 14:07:01,012 : WARNING : scanning corpus to determine the number of features (consider setting `num_features` explicitly)
2019-12-17 14:08:03,420 : INFO : creating matrix with 34249 documents and 100 features
"""

# Index persistency is handled via the standard save() and load() functions:

save_name = 'test.index'

index.save(save_name)
index = similarities.MatrixSimilarity.load(save_name)

"""
2019-12-16 13:39:21,451 : INFO : saving MatrixSimilarity object under test.index, separately None
2019-12-16 13:39:21,519 : INFO : saved test.index
2019-12-16 13:39:21,520 : INFO : loading MatrixSimilarity object from test.index
2019-12-16 13:39:21,568 : INFO : loaded test.index
"""

# Obtain similarities of our query documents against the indexed documents:

# Cosine measure returns similarities in the range <-1, 1> (the greater, the 
# more similar), so that the first document has a score of 0.99809301 etc.

# documents that would never be returned by a standard boolean fulltext search, 
# because they do not share any common words with a given query, will be
# returned using Latent Semantic Analysis. After applying LSI, we can observe 
# this. Semantic generalization is the reason why we apply transformations and 
# do topic modelling.

from operator import itemgetter

# index = similarities.MatrixSimilarity(lsi[corpus])

def lsi_match(query, index):
  vec_bow = dictionary.doc2bow(query.lower().split())
  vec_lsi = lsi[vec_bow]  # convert the query to LSI space
  print(f'\nQuery: {query}\nvec_lsi: {vec_lsi}\nResults: ')
  sims = index[vec_lsi]
  results = []
  for x, y, z in zip(df['title'], df['raw'], sims):
    if z > 0.75:
      results.append((x, y, z))
  results.sort(key=itemgetter(2), reverse=True)
  for i in results[0:5]:
    title = i[0]
    raw = i[1]
    score = i[2]
    print(f'Score: {score:.3f}, Title: {title}, Raw: {raw[0:100]}')

queries = ["Human computer interaction graphical monitor design program", 
           "features in supervised learning methods", 
           "optimal number of dimensions for performing SVD", 
           "finland finnish language grammar", 
           "voice over internet protocol voip ip telephony communications"]

for i in queries:
  lsi_match(i, index)

"""
Query: Human computer interaction graphical monitor design program
vec_lsi: [(0, 0.13707405327635885), (1, 0.2159978851508788), (2, 0.04407301845802505), (3, -0.049277347897273005), (4, 0.014457924371043714), (5, -0.05083655309192249), (6, 0.15187521253514352), (7, -0.047088197073755826), (8, 0.031522418266874894), (9, -0.043931476984699486), (10, -0.025106629129389307), (11, 0.01582374077746986), (12, 0.02390292767935786), (13, -0.05206074471926509), (14, 0.02937573523622239), (15, 0.07767687747422664), (16, -0.2578375303097805), (17, -0.12955251550844232), (18, -0.12512515273934766), (19, -0.008263767125278591), (20, -0.1334877859216173), (21, -0.07417944336385081), (22, -0.12979113863651418), (23, 0.02076487534743865), (24, -0.1858118697847664), (25, 0.10413805564207741), (26, -0.1719565033618589), (27, 0.03945645681140868), (28, 0.06005555040770153), (29, 0.04284439312232846), (30, -0.03257554379325359), (31, 0.07428675998357591), (32, 0.060195675453692296), (33, -0.05920761571711278), (34, 0.1296446572439417), (35, -0.01423175764028989), (36, -0.09899061950911739), (37, -0.09146639537498262), (38, 0.014505460105801875), (39, -0.06134085976602741), (40, 0.13760805509430776), (41, 0.04173445622168695), (42, 0.0063952990373374536), (43, 0.03346660535885038), (44, 0.02936796675694397), (45, -0.18621381313604918), (46, 0.05561703381633061), (47, 0.15641739302675237), (48, 0.08185660117454631), (49, 0.0882787841207093), (50, -0.143257028701576), (51, 0.18151868535212937), (52, -0.26113746189688103), (53, -0.1039264463540601), (54, -0.12600368492401992), (55, -0.15297527378138967), (56, -0.23469187597925173), (57, -0.052063195132383056), (58, -0.07909808416146924), (59, 0.11661577279916126), (60, 0.19825743335517654), (61, 0.17487273215286095), (62, -0.020931835776531602), (63, 0.2067876380110609), (64, -0.0858784886176992), (65, 0.14387770699363783), (66, -0.1842291340987903), (67, 0.01952997789758677), (68, 0.012044838459898247), (69, -0.06162459300143324), (70, -0.04938059417735593), (71, -0.0583842728742691), (72, -0.11084881353803072), (73, -0.03471522603208803), (74, -0.00799013708247643), (75, -0.07958413610470343), (76, -0.08630658265732061), (77, 0.08626711958514056), (78, -0.02445884053137854), (79, -0.13428552976369984), (80, 0.018167076439979453), (81, -0.05204068967757009), (82, -0.14001718680108394), (83, 0.1541668825380149), (84, 0.027085448587589114), (85, -0.09281673858891322), (86, 0.017078138923870716), (87, -0.1888970952786878), (88, -0.051488402506747494), (89, 0.03309720564555853), (90, -0.016326929934542385), (91, 0.26149527654953564), (92, 0.2662520236217624), (93, -0.026569299669805972), (94, 0.08141712383621354), (95, 0.23900289359629517), (96, 0.10731604841672882), (97, 0.014935164961376094), (98, 0.17141048505498918), (99, -0.1651990743274693)]
Results: 
Score: 0.835, Title: https://en.wikipedia.org/wiki/Affective_design, Raw: the notion of affective design emerged from the field of human–computer interaction hci 1 and more s
Score: 0.780, Title: https://en.wikipedia.org/wiki/Generative_design, Raw: generative design is an iterative design process that involves a program that will generate a certai
Score: 0.778, Title: https://en.wikipedia.org/wiki/Algorithms-Aided_Design_(AAD), Raw: algorithms aided design aad is the use of specific algorithms editors to assist in the creation modi
Score: 0.757, Title: https://en.wikipedia.org/wiki/Value_sensitive_design, Raw: value sensitive design vsd is a theoretically grounded approach to the design of technology that acc
Score: 0.755, Title: https://en.wikipedia.org/wiki/Error-tolerant_design, Raw: an error tolerant design also human error tolerant design 1 is one that does not unduly penalize use

Query: features in supervised learning methods
vec_lsi: [(0, 0.05015049477879048), (1, 0.08045899064998223), (2, 0.011497391879206807), (3, -0.04811632353559152), (4, 0.02524988442915319), (5, 0.006630155346334154), (6, 0.049889798303881856), (7, -0.06164268116564601), (8, -0.012864931189607142), (9, -0.021851538360048436), (10, -0.013660125324415679), (11, 0.02850951329383979), (12, 0.004397539476616507), (13, -0.010572624297485234), (14, 0.007814442497170745), (15, 0.004011982196782932), (16, -0.023037767261155178), (17, 0.0017031589838506635), (18, 0.015250215361511764), (19, 0.01636756286407451), (20, -0.00362209495931017), (21, -0.005941900336283989), (22, -0.014775837762579062), (23, 0.013365668638734642), (24, -0.017386446986554425), (25, 0.04735019008148375), (26, -0.024975561430839247), (27, -0.11243646729864071), (28, -0.036321219947782486), (29, -0.010480396195735212), (30, 0.02232014363004371), (31, 0.023043140499803695), (32, 0.0534830914891495), (33, -0.010478217316656513), (34, -0.00451581579175543), (35, 0.005508646217287399), (36, -0.016074374671422174), (37, -0.01367286632529435), (38, 0.002601105938495496), (39, -0.009721414209178903), (40, 0.0003634134984555528), (41, -0.015158844903193568), (42, -0.0206632244716436), (43, -0.058214017966082166), (44, 0.015989308583748904), (45, -0.035757302404690876), (46, -0.020497632576539783), (47, 0.019589126598868266), (48, 0.012558843856962255), (49, 0.020701357739096597), (50, 0.015776677287451798), (51, 0.013571811275993755), (52, 0.048896131649782106), (53, -0.06133309365868915), (54, 0.020733180012099943), (55, 0.06249454817532052), (56, -0.014270920054286722), (57, -0.05818086424888387), (58, 0.0419880456087737), (59, -0.03315020944496991), (60, 0.051734087269791916), (61, 0.02900387298688888), (62, 0.014852035242101509), (63, -0.021659082304001587), (64, -0.06086301473182876), (65, 0.0457959300356181), (66, -0.0284049332191643), (67, 0.058338712731453496), (68, -0.015089988294481523), (69, -0.01336677653324726), (70, 0.09523569887896496), (71, 0.04264005088188507), (72, -0.03525823517884706), (73, -0.007749105966947961), (74, 0.05618028120273806), (75, -0.009493118624443414), (76, -0.008870228770977746), (77, -0.01328828663720543), (78, -0.02332862081200766), (79, -0.00011935974088219054), (80, -0.015423992608187961), (81, -0.016193059340260534), (82, 0.0013518461764975226), (83, -0.0023359133824842897), (84, 0.04023039005199204), (85, -0.06654437364213076), (86, -0.002992543726713802), (87, 0.022722519712046155), (88, -0.037413664843898485), (89, 0.0019436649908215957), (90, 0.013603878035030981), (91, 0.01265347388385176), (92, 0.010637862676213528), (93, -0.012480370115071911), (94, -0.0003498694924716577), (95, -0.01969049236587408), (96, 0.01644993537372714), (97, 0.02804685989678796), (98, -0.007235137192679079), (99, -0.0242968285664351)]
Results: 
Score: 0.863, Title: https://en.wikipedia.org/wiki/List_of_machine_learning_concepts, Raw: the following outline is provided as an overview of and topical guide to machine learning machine le
Score: 0.863, Title: https://en.wikipedia.org/wiki/Outline_of_machine_learning, Raw: the following outline is provided as an overview of and topical guide to machine learning machine le
Score: 0.780, Title: https://en.wikipedia.org/wiki/Automated_machine_learning, Raw: automated machine learning automl is the process of automating the process of applying machine learn
Score: 0.777, Title: https://en.wikipedia.org/wiki/Learning_styles, Raw: learning styles refer to a range of competing and contested theories that aim to account for differe
Score: 0.766, Title: https://en.wikipedia.org/wiki/Machine_learning, Raw: machine learning ml is the scientific study of algorithms and statistical models that computer syste

Query: optimal number of dimensions for performing SVD
vec_lsi: [(0, 0.08028632997509326), (1, 0.05185449637820646), (2, -0.003390647357509529), (3, 0.010275118099581234), (4, 0.016206664354543134), (5, 0.021965751538537286), (6, 0.0028564937128779428), (7, 0.005541348956590897), (8, -0.04027415694413797), (9, -0.002155959996202227), (10, 0.026087784634204633), (11, 0.023483220251547508), (12, -0.015773069944514582), (13, 0.0009348289893149938), (14, -0.015377407970204246), (15, -0.060114832741497634), (16, -0.039476516772073825), (17, 0.004039850628553047), (18, 0.06649477182413084), (19, 0.03815041599098148), (20, 0.061564336667210895), (21, 0.06416718780780209), (22, 0.008827053637384176), (23, -0.003273365885434081), (24, 0.04159586784474983), (25, -0.00012664821714805695), (26, -0.011284148567517081), (27, -0.03457773331382792), (28, 0.06689617442129124), (29, 0.005758825358101068), (30, -0.015138643845296342), (31, -0.0358577362227625), (32, 0.02365635077508897), (33, 0.013074092365117479), (34, 0.0049778043075174834), (35, -0.006373834632188827), (36, 0.019656032405662216), (37, 0.04947939701632408), (38, 0.005115341683088097), (39, 0.04628128470618246), (40, -0.08568811137721774), (41, -0.01787859944266443), (42, 0.08261615702600464), (43, -0.004325836260178691), (44, 0.014949073932038577), (45, -0.009726554964831756), (46, 0.0024681173372367643), (47, -0.014559198451288727), (48, -0.014794118996978549), (49, 0.09956815737091534), (50, 0.02012782258309232), (51, 0.00928438715133517), (52, -0.0337754775046108), (53, -0.009870693491738892), (54, -0.0013420131293340968), (55, -0.021028788275736832), (56, 0.016106141244025592), (57, 0.03655768442973558), (58, -0.03340763403195278), (59, 0.008006914858942856), (60, -0.004944359374655215), (61, -0.0015376734920761307), (62, 0.018649579260762668), (63, 0.03311280646645544), (64, 0.02537949095198067), (65, -0.05611134268651836), (66, 0.05068542991957686), (67, 0.03985185377774021), (68, 0.020572002892852), (69, 0.05989765357711765), (70, 0.052304738622783745), (71, 0.03041085080476092), (72, 0.02254800991856249), (73, -0.006609523843631862), (74, 0.021738873038294918), (75, 0.043869890709499024), (76, -0.04706810150938955), (77, -0.0009917051678637516), (78, 0.038489971781897704), (79, 0.02712012550842693), (80, -0.0002968980219745748), (81, -0.004176773829484445), (82, -0.0038489653856144225), (83, 0.04076741013998485), (84, 0.003840441432143916), (85, -0.02393111179133407), (86, -0.024530766458558565), (87, -0.02199687859277141), (88, -0.04575377716267239), (89, -0.021919623173136192), (90, -0.014951459777631676), (91, 0.06384797706246571), (92, 0.04134745062826761), (93, -0.011022988902178276), (94, 0.03173065524634979), (95, -0.001875988231216191), (96, 0.0035327921612893875), (97, -0.009953686615592346), (98, -0.02971512082889364), (99, 0.05611753970701575)]
Results: 
Score: 0.870, Title: https://en.wikipedia.org/wiki/Prime_number, Raw: a prime number or a prime is a natural number greater than 1 that cannot be formed by multiplying tw
Score: 0.870, Title: https://en.wikipedia.org/wiki/Goldbach%27s_conjecture, Raw: goldbach's conjecture is one of the oldest and best known unsolved problems in number theory and all
Score: 0.856, Title: https://en.wikipedia.org/wiki/Ranking, Raw: a ranking is a relationship between a set of items such that for any two items the first is either '
Score: 0.842, Title: https://en.wikipedia.org/wiki/Elementary_arithmetic, Raw: elementary arithmetic is the simplified portion of arithmetic that includes the operations of additi
Score: 0.816, Title: https://en.wikipedia.org/wiki/Number, Raw: a number is a mathematical object used to count measure and label the original examples are the natu

Query: finland finnish language grammar
vec_lsi: [(0, 0.06845446499875794), (1, -0.0013098864834276153), (2, -0.06686491437815424), (3, -0.07658970151999166), (4, 0.03761117220546937), (5, 0.06001856070599434), (6, 0.11599767966514628), (7, -0.2392171656937063), (8, -0.3560801766131511), (9, 0.024760567798202578), (10, 0.37831727172452284), (11, 0.18656751990884166), (12, -0.22894362908186117), (13, -0.01127266151675023), (14, -0.042031227332125026), (15, -0.05981215585673791), (16, 0.13902786608993475), (17, 0.15584757963258108), (18, -0.11724141473133434), (19, 0.03563057567744052), (20, 0.03679276172783558), (21, -0.14830682767259518), (22, -0.11542408585733828), (23, 0.10892543956521103), (24, -0.025031822502786443), (25, -0.015685482044142697), (26, -0.017498244695924037), (27, 0.03911770774619298), (28, -0.07530772143684134), (29, 0.1252434080643923), (30, 0.02582259671909536), (31, 0.016169097400736063), (32, 0.030288245296446983), (33, 0.032922845862725814), (34, -0.03845876408220117), (35, 0.07756799431390853), (36, -0.06524077048539798), (37, -0.11315518459698747), (38, 0.02384385040626524), (39, -0.005848441082467252), (40, 0.06629110078012013), (41, -0.06325625303247541), (42, 0.01606048452529608), (43, 0.0046253008255814475), (44, -0.02739086908195721), (45, -0.05050041948803959), (46, 0.03590000151446508), (47, 0.00041195106126908564), (48, -0.010218881620998824), (49, -0.045541292803695406), (50, 0.019257599533283095), (51, 0.05554242195063476), (52, -0.0044146422626328745), (53, -0.0311693055826728), (54, -0.08409820185134514), (55, 0.028606856258679425), (56, -0.0010009688307674814), (57, 0.02185402800750804), (58, -0.01749573855564819), (59, 0.019947143269691662), (60, 0.03487739121237827), (61, -0.04805007961443828), (62, -0.0008344839204603351), (63, -0.004649850590970183), (64, -0.06677954652404046), (65, 0.010932754991359029), (66, -0.02776690021226127), (67, -0.045533826540574016), (68, -0.0056548019593724175), (69, -0.0788192266814653), (70, -0.00956961383387015), (71, 0.02155703857452617), (72, -0.10913220837179499), (73, -0.010500387942045364), (74, 0.059121194406844976), (75, 0.07303908181748885), (76, -0.06060341672705707), (77, 0.039107067061390846), (78, -0.051627313532345155), (79, 0.013747743290631564), (80, -0.03172760690143613), (81, -0.0426106363357746), (82, -0.01610820710244274), (83, -0.01566524394331266), (84, -0.06368386232935969), (85, -0.0570452081307471), (86, -0.07611618003002277), (87, 0.06404292405596673), (88, 0.020450194068462945), (89, 0.12518821040951802), (90, -0.042074742810727955), (91, 0.05009696703615121), (92, -0.0035128022668640202), (93, -0.0569678012340217), (94, 0.01606980354612521), (95, -0.014375286725717142), (96, 0.08660978235660712), (97, -0.02882194011779075), (98, -0.00403340066224993), (99, -0.019064112698775126)]
Results: 
Score: 0.985, Title: https://en.wikipedia.org/wiki/List_of_philosophers_of_language, Raw: this is a list of philosophers of language
Score: 0.970, Title: https://en.wikipedia.org/wiki/Tonsea_language, Raw: tonsea tonsea’ is an austronesian language of the northern tip of sulawesi indonesia it belongs to t
Score: 0.967, Title: https://en.wikipedia.org/wiki/Spoken_language, Raw: a spoken language is a language produced by articulate sounds as opposed to a written language many 
Score: 0.966, Title: https://en.wikipedia.org/wiki/Endangered_languages, Raw: an endangered language or moribund language is a language that is at risk of falling out of use as i
Score: 0.965, Title: https://en.wikipedia.org/wiki/Language_death, Raw: in linguistics language death occurs when a language loses its last native speaker by extension lang

Query: voice over internet protocol voip ip telephony communications
vec_lsi: [(0, 0.03310193749051803), (1, 0.05899512541721047), (2, 0.038573352655269094), (3, 0.030375934493718587), (4, 0.04252073679319334), (5, -0.08800019788099223), (6, 0.09084386994160701), (7, -0.029528388980505433), (8, 0.0017245765435606238), (9, 0.020344211724941802), (10, -0.022155557062378595), (11, 0.02446489788103), (12, -0.01219884962738127), (13, -0.03478428612931555), (14, 0.043372678957472946), (15, -0.07487891725870115), (16, 0.015258769867623067), (17, -0.0602244376770924), (18, -0.05151903221777858), (19, 0.027735934891141213), (20, 0.02169118526257693), (21, 0.01150683990256452), (22, -0.011271439944196448), (23, -0.06233939560213482), (24, 0.011250942759313263), (25, 0.10616174096918271), (26, 0.0609792296317231), (27, 0.0045483901759275705), (28, -0.005005100283105329), (29, -0.04941598616367887), (30, 0.010523617140930423), (31, 0.021647193977686333), (32, 0.006906901316479986), (33, -0.09455841545479339), (34, -0.036128791114570256), (35, -0.06583680477138273), (36, 0.061847374569493775), (37, 0.004647692729085592), (38, -0.019946518445805066), (39, -0.008009353144378187), (40, -0.013281523324950093), (41, -0.07915037383674951), (42, 0.04350488846123296), (43, -0.05544134446049302), (44, 0.09226482217718478), (45, 0.032502732404994274), (46, -0.0467820977327559), (47, -0.02070217941448918), (48, -0.06687064576092466), (49, -0.006162160722949403), (50, 0.00010066041368500729), (51, 0.03972777709564979), (52, 0.03783596931963305), (53, 0.020718139088725395), (54, -0.060457105012554684), (55, 0.01861371307777686), (56, 0.045854737357560564), (57, -0.10948500635481981), (58, 0.13273986352799425), (59, 0.007822280932974813), (60, 0.01975911071967852), (61, -0.19408452775596924), (62, -0.035792696077471216), (63, 0.08558905701171425), (64, -0.05665437557086236), (65, -0.010764633419086319), (66, 0.08618739502906451), (67, -0.010425511183481256), (68, -0.026356153244562416), (69, 0.013824330225462835), (70, -0.025693617126362964), (71, -0.023250198627873513), (72, -0.019287985484994073), (73, -0.014927651765844093), (74, -0.004868421964131005), (75, 0.07566504807964941), (76, -0.0012478101015280945), (77, 0.01350343715722645), (78, -0.003670476994436963), (79, -0.056460016459597), (80, -0.06623515042729664), (81, -0.02060406630111417), (82, 0.045796536107135855), (83, 0.03843020476044025), (84, -0.06040872518795328), (85, 0.030165118696574674), (86, 0.058688474106550086), (87, -0.03440515848686719), (88, -0.035709753769632466), (89, -0.018489084326823564), (90, -0.019268349622526583), (91, -0.025474575339729612), (92, -0.04435000298221436), (93, -0.02890732646731483), (94, 0.04845894232412538), (95, 0.06765405534629601), (96, -0.09598350221851533), (97, 0.006409017951587), (98, 0.042938997924440614), (99, 0.04065038759793298)]
Results: 
Score: 0.812, Title: https://en.wikipedia.org/wiki/Internet, Raw: the internet portmanteau of interconnected network is the global system of interconnected computer n
Score: 0.807, Title: https://en.wikipedia.org/wiki/Internet_protocol, Raw: the internet protocol ip is the principal communications protocol in the internet protocol suite for
Score: 0.801, Title: https://en.wikipedia.org/wiki/Overlay_network, Raw: an overlay network is a computer network that is layered on top of another network 1 2 nodes in the 
Score: 0.785, Title: https://en.wikipedia.org/wiki/Internet_backbone, Raw: the internet backbone may be defined by the principal data routes between large strategically interc
Score: 0.785, Title: https://en.wikipedia.org/wiki/Internetwork, Raw: internetworking is the concept of interconnecting different types of networks to build a large globa
"""

sims = index[vec_lsi]

df['sims'] = sims

df.head()

"""
title 	raw 	stop 	sims
0 	https://en.wikipedia.org/wiki/Josephus 	titus flavius josephus dʒoʊˈsiːfəs 2 greek φλά... 	[titus, flavius, josephus, greek, 37, 100, bor... 	0.033839
1 	https://en.wikipedia.org/wiki/Social_identity_... 	social identity is the portion of an individua... 	[social, identity, portion, individual's, self... 	-0.026508
2 	https://en.wikipedia.org/wiki/WAITS 	waits was a heavily modified variant of digita... 	[waits, heavily, modified, variant, digital, e... 	0.091748
3 	https://en.wikipedia.org/wiki/Prometheus_Bound 	prometheus bound ancient greek προμηθεὺς δεσμώ... 	[prometheus, bound, ancient, greek, ancient, g... 	0.001385
4 	https://en.wikipedia.org/wiki/Metaphysical_art 	metaphysical art italian pittura metafisica wa... 	[metaphysical, art, italian, pittura, style, p... 	0.041469
"""

raw = df['raw']

for x, y in zip(raw, sims):
  if y > 0.9:
    print(y, x)

# example output:
"""
0.90301585 the two main official languages of finland are finnish and swedish there are also several official minority languages three variants of sami romani finnish sign language and karelian finnish is the language of the majority 89 2 of the population it is a finnic language closely related to estonian and less closely to the sami languages the finnic languages belong to the uralic language family so finnish is distantly related to languages as diverse as hungarian a ugric language and nenets a samoyedic language in siberia swedish is the main language of 5 3 of the population 2 92 4 in the åland autonomous province down from 14 at the beginning of the 20th century in 2012 44 of finnish citizens with another registered primary language than swedish could hold a conversation in this language 3 swedish is a north germanic language closely related to norwegian and danish as a subbranch of indo european it is also closely related to other germanic languages such as german dutch and english swedish was the language of the administration until the late 19th century today it is one of the two main official languages with a position equal to finnish in most legislation though the working language in most governmental bodies is finnish both finnish and swedish are compulsory subjects in school with an exception for children with a third language as their native language a successfully completed language test is a prerequisite for governmental offices where a university degree is required the four largest swedish speaking communities in finland in absolute numbers are those of helsinki espoo porvoo and vaasa where they constitute significant minorities helsinki the capital had a swedish speaking majority until late in the 19th century currently 5 9 4 of the population of helsinki are native swedish speakers and 15 are native speakers of languages other than finnish and swedish 5 the swedish dialects spoken in finland mainland are known as finland swedish there is a rich finland swedish literature including authors such as tove jansson johan ludvig runeberg edith södergran and zacharias topelius runeberg is considered finland's national poet and wrote the national anthem vårt land which was only later translated to finnish the sami languages are a group of related languages spoken across lapland they are distantly related to finnish the three sami languages spoken in finland northern sami inari sami and skolt sami have a combined native speaker population of roughly 1 800 6 up to world war ii karelian was spoken in the historical border karelian region on the northern shore of lake ladoga after the war immigrant karelians were settled all over finland in 2001 the karelian language society estimated that the language is understood by 11 000–12 000 people in finland most of whom are elderly a more recent estimate is that the size of the language community is 30 000 7 karelian was recognized in a regulation by the president in november 2009 in accordance with the european charter for regional or minority languages 8 the russian language is the third most spoken native language in finland 1 9 the russian language has no official status in finland though historically it served as the third co official language with finnish and swedish for a relatively brief period between 1900 and 1917 all municipalities outside åland where both official languages are spoken by either at least 8 of the population or at least 3 000 people are considered bilingual swedish reaches these criteria in 59 out of 336 municipalities located in åland where this does not matter and the coastal areas of ostrobothnia region southwest finland especially in åboland outside turku and uusimaa outside these areas there are some towns with significant swedish speaking minorities not reaching the criteria thus the inland is officially unilingually finnish speaking finnish reaches the criteria everywhere but in åland and in three municipalities in the ostrobothnia region which is also the only region on the finnish mainland with a swedish speaking majority 52 to 46 the sami languages have an official status in the northernmost finland in utsjoki inari enontekiö and part of sodankylä regardless of proportion of speakers in the bilingual municipalities signs are in both languages important documents are translated and authorities have to be able to serve in both languages authorities of the central administration have to serve the public in both official languages regardless of location and in sami in certain circumstances places often have different names in finnish and in swedish both names being equally official as name of the town for a list see names of places in finland in finnish and in swedish media related to languages of finland at wikimedia commons
0.9523109 ifugao or batad is a malayo polynesian language spoken in the northern valleys of ifugao philippines it is a member of the northern luzon subfamily and is closely related to the bontoc and kankanaey languages 3 it is a dialect continuum and its four main varieties—such as tuwali—are sometimes considered separate languages 4 loanwords from other languages such as ilokano are replacing some older terminology 5 ethnologue reports the following locations for each of the 4 ifugao languages the unified ifugao alphabet is as follows a b d e g h i k l m n ng o p t u w y the letters are pronounced differently depending on the dialect of speaker 6
0.9022218 the nicobarese languages or nicobaric languages form an isolated group of about half a dozen closely related austroasiatic languages spoken by the majority of the inhabitants of the nicobar islands of india they have a total of about 30 000 speakers 22 100 native the majority of nicobarese speakers speak the car language paul sidwell 2015 179 2 considers the nicobarese languages to subgroup with aslian the nicobarese languages appear to be related to the shompen language of the indigenous inhabitants of the interior of great nicobar island blench sidwell 2011 which is usually considered a separate branch of austroasiatic 3 however paul sidwell 2017 4 classifies shompen as a southern nicobaric language rather than as a separate branch of austroasiatic the morphological similarities between nicobarese and austronesian languages have been used as evidence for the austric hypothesis reid 1994 5 from north to south the nicobaric languages are paul sidwell 2017 classifies the nicobaric languages as follows 4
0.9201585 comorian shikomori or shimasiwa the language of islands is the name given to a group of four bantu languages spoken in the comoro islands an archipelago in the southwestern indian ocean between mozambique and madagascar it is named as one of the official languages of the union of the comoros in the comorian constitution shimaore one of the languages is spoken on the disputed island of mayotte a french department claimed by comoros like swahili the comorian languages are sabaki languages part of the bantu language family each island has its own language and the four are conventionally divided into two groups the eastern group is composed of shindzuani spoken on ndzuani and shimaore mayotte while the western group is composed of shimwali mwali and shingazija ngazidja although the languages of different groups are not usually mutually intelligible only sharing about 80 of their lexicon there is mutual intelligibility between the languages within each group suggesting that shikomori should be considered as a two language groups rather than four distinct languages 6 historically the language was written in the arabic script the french colonial administration introduced the latin script of which a modified version was officially decreed in 2009 7 most comorians use the latin script when writing the comorian language it is the language of umodja wa masiwa the national anthem the consonants and vowels in the comorian languages the consonants mb nd b d are phonetically recognized as ranging from ᵐɓ ᵐb ⁿɗ ⁿd ɓ b ɗ d
0.91117495 the mongolic languages are a language family that is spoken in east central asia mostly in mongolia inner mongolia an autonomous region of china xinjiang another autonomous region of china the region of qinghai and also in kalmykia a republic of southern european russia mongolic is a small relatively homogenous and recent language family whose common ancestor proto mongolian was spoken at the beginning of the second millennium ad 1 2 however proto mongolian seems to descend from a common ancestor to languages like khitan which are sister languages of mongolian languages they do not descend from proto mongolian but are sister languages from an even older language from the first millennium ad i e para mongolian 3 4 the mongolic language family has about 6 million speakers the best known member of this language family mongolian is the primary language of most of the residents of mongolia and the mongolian residents of inner mongolia with an estimated 5 2 million speakers 5 hypothetical relation to other language families and their proto languages unclassified languages that may have been mongolic or members of other language families include
0.92173475 the yotayotic languages are a pair of languages of the pama–nyungan family yotayota and yabula yabula 2 dixon 2002 classified them as two separate families but per bowe morey 1999 3 glottolog considers them to be dialects of a single language
0.9140527 the turkic languages are a group of languages spoken across eastern europe the middle east central asia and siberia turkic languages are spoken as native languages by some 170 million people the following is a list of said languages by subfamily hypothetical relation to other language families and their proto languages unclassified languages that may have been turkic or members of other language families
"""
