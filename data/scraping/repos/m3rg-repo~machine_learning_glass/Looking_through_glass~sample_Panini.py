#!/usr/bin/env python
# coding: utf-8

# In[65]:


# Basic Python modules
import pickle
import time
import math
import os
import re
import string
import random
import threading
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# Gensim
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models import TfidfModel
from gensim import corpora, models, similarities
from gensim.parsing.preprocessing import STOPWORDS

#Plotting Tools
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.patheffects as PathEffects
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})
#NLTK
import nltk

#Scikit learn models 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn. feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances

# Import pandas and Numpy
import pandas as pd
import numpy as np

# Chemdataextractor
from chemdataextractor.doc import Document as Doc, Heading, Paragraph


# In[66]:


tag_list=[['AFM','atomic force microscop'],'electron densit',['SEM', 'scanning electron', 'BSE micrograph'], ['STM','Scanning tunneling'], ['band struct','band gap'],['XRD','X-ray diffrac','X ray diffrac'],'Strain','Raman spectr', ['TEM','Transmission electron microscop','HRTEM'],'Current-volt','Current volt','hysteresis','Thermodynamics', 'Network former', 'Network modifier', 'Cation', 'Anion',['DOS','Density of State'],'Phase diagram','Thermal conductivity', ['Scratch resistance','Scratch'],['Dielectric', 'Dielectric constant', 'Dielectric loss', 'breakdown', 'breakdown strength', 'BDS'],['TGA','Thermogravimetry'],['DSC','Differential scanning','Calorimetry', 'heat flow'], ['RI','Refractive ind'],'Cyclic volta',['FEM','finite element', 'FE'],['Magnetization curv', 'magnetization'], 'stress–strain curv',['plastic','ductil'],['tensi'],['compress'],'phonon',
'microstruct',\
'mass spectr', 'current map', 'isotherms',\
['resist', '3D VG theory', '3D VGT formula'], 'conduct',\
['Polarizat','PV','polarizability'], ['Glass transition temperature', 'Glass transition','Tg'], ['Crystal', 'kinetics'],\
['phase separa','immiscibility'],['Liquidus temperature', 'Liquidus', 'solidus'],\
'Charge compensator',\
['BO','Bridging'], ['NBO', 'Non-bridging '], 'Hardness', ['Fracture toughness', 'KIC'], 'Density',\
'Modulus', 'Poisson’s ratio',\
['Thermal expansion', 'CTE', 'TEC'], ['Specific heat', 'heat capacity', 'adiabatic'], \
['Structure factor', 'Neutron scattering', 'FWHM','FSDP'], 'X-ray scattering', 'Indentation', 'radiation', 'bioactive',\
['distribution function'], ['transmission', 'transmi'],\
['absorp','absorb'],['reflect'],\
['MD','molecular dynamics','molecular simulation','BKS','CHIK'],['MC', 'Monte-Carlo', 'Monte carlo'],\
['Thermo lumines', 'Thermo-luminescence', 'TL'], 'Excitation',\
['optically stimulated luminescence', 'OSL','optical lumines'], ['Electron Diffrac','Selected Area Electron Diffraction', 'Thermo luminescence', 'SAED','SEAD','SAD'],\
[ 'EDX','Energy dispersive X-ray spectroscopy', 'X-ray spectroscopy', 'EDS', 'Xray spectroscopy', 'XPS'],\
['Energy diag', 'Energy level diag', 'Energy level transition', 'Energy level scheme', 'energy diagram'],\
['PLE', 'PL', 'Photolumines'],\
'stimulated emission', ['Gain coefficient', 'Gain measurement spectra', 'Optical gain',' Gain spectra', 'Gain cross-section'], 'lifetime',\
'leakage current density', ['lumines', 'Luminescence spectra', 'luminescence decay'],\
['MAS NMR', 'Magic Angle Spinning'] , ['emission', 'emission spectra'],['DTA', 'Differential thermal'],\
['free energy', 'Helmholtz', 'absorbed energy', 'gibbs energy'], ['CIE', 'chromaticity diag'], 'Weight loss',\
['strength', 'tensile strength', 'compressive strength', 'flexural strength', 'yield strength', 'ultimate strength', 'shear strength'],\
 ['anneal','heat-treatment', 'sinter'], ['resonance', 'resonance freq’, ‘Electron paramagnetic resonance'], 'Impedance spect', \
'bonding scheme',['ionic conduc'], ['FTIR Spectr'],'IR emission',['Quantum efficiency','QE'],['iso-conversion plot'],\
['Tc','crystallization temperature','Tp'],'thermodynamic barrier',['UC emission', 'UC Luminescence', 'UC Spectra', 'UC Intensity'],'magnetic entropy', \
'energy trans', 'Cross relax', 'fracture', 'acoustic emiss', 'electric dipole',\
['NIR', 'NIR emission', 'NIR lumines','NIR fluores', 'Near Infra-red'], ['Relaxation time','relaxation'], ['IR transmi'],'quantum yield', ['Crack','Crack propagation','crack deflection'], \
'thermal sensit', 'CRET efficiency', 'spectral power distribution', 'HETCOR spect', ['UV-Vis absorp', 'Visible absorp', 'Visible spect', 'Visible emiss'], \
'fringe pattern', 'Tammann triangle',['lattice parameter', 'lattice constant', 'lattice'], 'current density',\
'FFT', 'Viscosity', \
'IR micrograph', 'Dilatometry', ['XEL spectr', 'X-ray excited luminescence spectr'], \
['Particle size distribution', 'Grain size distribution', 'PSD', 'GSD', 'Particle size', 'Grain size'], 'Optical absorp', ['activation energy','Kissinger', 'kinetics', 'Ozawa', 'JMA'], \
['CCT','correlated color temperature'], ['Eigen freq'], 'Abbe number', ['fluorescence','fluores','fluorescence intensity ratio', 'FIR'],\
['SERS Spect', 'Surface-enhanced Raman spect','SERS', 'Surface enhanced Raman spect'],'Molar Vol', 'IR fluores',\
['radio-lumines', 'RL Spect'], ['interface', 'interfac energ', 'permeation layer'],\
['VG transition line', 'VG'],\
['energy-storage density', 'storage'],\
['Qn', 'Q1', 'Q2', 'Q3', 'Q4'], ['MIR', 'MIR emission'],\
['angular', 'angular distribution'], \
'Z-scan',\
'Gruneisen', 'gamma ray',\
['Optical photograph', 'topology', 'Optical micrograph'],\
['dissolution','dissolve','flows','corros','leach','rotating disk method']]


# In[67]:


ELEMENT_NAMES = ['hydrogen', 'helium', 'lithium', 'beryllium', 'boron', 'carbon', 'nitrogen', 'oxygen', 'fluorine',
                     'neon', 'sodium', 'magnesium', 'aluminium', 'silicon', 'phosphorus', 'sulfur', 'chlorine', 'argon',
                     'potassium', 'calcium', 'scandium', 'titanium', 'vanadium', 'chromium', 'manganese', 'iron',
                     'cobalt', 'nickel', 'copper', 'zinc', 'gallium', 'germanium', 'arsenic', 'selenium', 'bromine',
                     'krypton', 'rubidium', 'strontium', 'yttrium', 'zirconium', 'niobium', 'molybdenum', 'technetium',
                     'ruthenium', 'rhodium', 'palladium', 'silver', 'cadmium', 'indium', 'tin', 'antimony', 'tellurium',
                     'iodine', 'xenon', 'cesium', 'barium', 'lanthanum', 'cerium', 'praseodymium', 'neodymium',
                     'promethium', 'samarium', 'europium', 'gadolinium', 'terbium', 'dysprosium', 'holmium', 'erbium',
                     'thulium', 'ytterbium', 'lutetium', 'hafnium', 'tantalum', 'tungsten', 'rhenium', 'osmium',
                     'iridium', 'platinum', 'gold', 'mercury', 'thallium', 'lead', 'bismuth', 'polonium', 'astatine',
                     'radon', 'francium', 'radium', 'actinium', 'thorium', 'protactinium', 'uranium', 'neptunium',
                     'plutonium', 'americium', 'curium', 'berkelium', 'californium', 'einsteinium', 'fermium',
                     'mendelevium', 'nobelium', 'lawrencium', 'rutherfordium', 'dubnium', 'seaborgium', 'bohrium',
                     'hassium', 'meitnerium', 'darmstadtium', 'roentgenium', 'copernicium', 'nihonium', 'flerovium',
                     'moscovium', 'livermorium', 'tennessine', 'oganesson', 'ununennium']


# In[68]:


ELEMENTS = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K',
                'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I',
                'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr',
                'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf',
                'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Uue']


# In[69]:


def caption_labeller(DF,tag_list=tag_list):
    
    print("Commencing labeling...")
    label=0
    
    for t in tag_list:
        
        if type(t)==list:
            
            for tag in t:
            
                tag=tag.lower()
                    
                DF.loc[DF['Caption'].str.contains(tag,case=False,na=False,regex=True),'tag']=t[0]
                DF.loc[DF['Caption'].str.contains(tag,case=False,na=False,regex=True),'label']=label
        
        if type(t)==str:
            
            tag=t.lower()
             
            DF.loc[DF['Caption'].str.contains(tag,case=False,na=False,regex=True),'tag']=t
            DF.loc[DF['Caption'].str.contains(tag,case=False,na=False,regex=True),'label']=label
        
        label+=1
    
    DF.dropna(inplace=True)
    
    size_dict={}
    
    for tag in tag_list:
        if type(tag)==list:
            tag=tag[0]
        val=len(DF[DF['tag']==tag])
        size_dict[tag]=val
    
    for tag in tag_list:
        if type(tag)==list:
            tag=tag[0]
        size_dict[tag]=np.around(size_dict[tag]*20/max(list(size_dict.values()))+7,2)
    
    sorted_dict={}
    for w in sorted(size_dict, key=size_dict.get):
        sorted_dict[w]=size_dict[w]
        
    print("Labelling complete...")
            
    return sorted_dict


# In[70]:


def process_data(DF, column_name):
    data = DF[column_name].tolist()
    data_words = list(sent_to_words(data))
    
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100) 
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    data_words_nostops = [[word for word in simple_preprocess(str(doc)) if word not in STOPWORDS] for doc in data_words]
    data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]
    
    id2word = corpora.Dictionary(data_words_bigrams)
    texts = data_words_bigrams
    corpus = [id2word.doc2bow(text) for text in texts]
    
    return corpus, id2word


# In[71]:


def folder_search(Query, pii, base):
        print(pii)
        pii_path=os.path.join(base,Query,str(pii))
        
        TempDF=pd.DataFrame(columns=['Figname','No','Caption'])
        
        if os.path.isdir(pii_path):

                cap_path=os.path.join(pii_path,pii+'-caption.txt')

                if os.path.exists(cap_path):
                    
                        TempDF=pd.read_csv(cap_path,header=None, names=['Figname','No','Caption'])
                        TempDF['Query']=Query
                        TempDF['PII']=pii
                
        
        return TempDF


# In[72]:


ascii4=re.compile('\W+')
number=re.compile('\d+')
extra_space=re.compile('\s\s+')

def text_prepare(text):
    temp=text.split(' ') 
    for word in STOPWORDS:
        temp=[x for x in temp if x!=word]
    text=' '.join(temp)
    return text

def preparer(DF):

    
    text=list(DF['Caption'])
    
    new_text=[]
    for t in text:
        if type(t)==str:
            t=t.replace(' ','ABC')
            t=ascii4.sub('',t)
            t=number.sub('ABC',t)
            t=t.replace('ABC',' ')
            t=extra_space.sub(' ',t)
            new_text.append(t)
    
    prepared = []
    for line in new_text:
        line = text_prepare(line.strip())
        prepared.append(line)
        
    return prepared

def vectorizer(prepared,DF,i):
    
    print('Featurizing\n')
    min_val=10
    tfidf=TfidfVectorizer(min_df=min_val,max_df=0.9,ngram_range=(1,3))
    
    features=tfidf.fit_transform(prepared)
    names=tfidf.get_feature_names()
    
    featDF=pd.DataFrame(features.todense(),columns=tfidf.get_feature_names())
    
    print('Cosine Matrix \n')
    X=featDF.values
    distance_matrix = pairwise_distances(X, X, metric='cosine', n_jobs=-1)
    
    print('TSNE \n')
    state=random.randint(0,200)
    time_start=time.time()
    tsne=TSNE(n_components=2,metric="precomputed",verbose=1,random_state=state,perplexity=40,n_iter=300)
    tsne_results=tsne.fit_transform(distance_matrix)

    print('t-SNE done! Time.elpased: {}'.format(time.time()-time_start))
    
    DF['State_'+str(i)]=state
    DF['x-label-'+str(i)]=tsne_results[:,0]
    DF['y-label-'+str(i)]=tsne_results[:,1]
    
    return None

def ccp_scatter(DF,sorted_dict,title,location,i=0):
    
    palette = np.array(sns.color_palette("hls", len(sorted_dict)))
    
    f = plt.figure(figsize=(12, 12))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(DF['x-label-'+str(i)], DF['y-label-'+str(i)], lw=0, s=20,c=palette[DF['label'].astype(np.int)])
    ax.axis('off')
    ax.axis('tight')
    
    txts = []
    for item in sorted_dict.keys():
        # Position of each label.
        xtext, ytext = np.median(DF[DF['tag']==item][['x-label-'+str(i),'y-label-'+str(i)]],axis=0)
        if not math.isnan(xtext) and not math.isnan(ytext):
            txt = ax.text(xtext, ytext, item, fontsize=sorted_dict[item])
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)
    
    plt.savefig(os.path.join(location, title+'.png'))
    plt.show()
    
    return None


# In[73]:


def sent_to_words(sentences):
        for sentence in sentences:
             yield(simple_preprocess(str(sentence), deacc=True)) 
        
def preprocess(text):
    result = []
    text=" ".join(text.split())
    for token in simple_preprocess(text):
        if token not in STOPWORDS and len(token) > 3:
            result.append((token))
    return result

def sim_calculator(DF, column_name):

    print("Number of {}: {}".format(column_name,len(DF[column_name])))

    #Preprocessing
    print('\nCreating Dictionary...')
    processed_docs = DF[column_name].map(preprocess)

    #Generating dictionary
    dictionary = corpora.Dictionary(processed_docs)
    # dictionary.filter_extremes(no_below=100,no_above=0.9, keep_n=100000)
    dictionary.filter_extremes(no_below=3,no_above=0.9, keep_n=None) #modified
    print('Dictionary created')
    print("Size of vocabularly: ",len(dictionary))

    #Bag of words
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    #Tfid vectorization
    print('\nRunning TFIDF vectorization...')
    model2 = TfidfModel(bow_corpus) 
    abs_tfidf=model2[bow_corpus]
    print('TFIDF complete')

    #Calculating similarties
    print('\nCalculating Similarity Matrix...')
    index = similarities.MatrixSimilarity(abs_tfidf)
    sims = index[abs_tfidf]
    print("size of similarity matrix: ", sims.shape)

    return sims

def tsne(sims,title,DF,i):

    #run PCA
    print("\nBeginning clustering...")
    N=3 #100
    pca = PCA(n_components=N)
    pca_result = pca.fit_transform(sims)
    var=np.sum(pca.explained_variance_ratio_)
    print("\nPCA calculated")
    print('Cumulative explained variation for {} principal components: {}'.format(N,var))

    #Running TSNE
    print("Creating TSNE labels...")
    tsne=TSNE(n_components=2,verbose=1,perplexity=40,n_iter=300)
    tsne_results=tsne.fit_transform(pca_result)
    print("TSNE complete")
    DF['x-label-'+str(i)]=tsne_results[:,0]
    DF['y-label-'+str(i)]=tsne_results[:,1]

    return str(np.round(var*100,2))

def dictionarizer(DF,topic_num):
    
    topic_num=str(topic_num)
    size_dict={}
    tag_list=list(set(DF[topic_num+'-topic']))

    for tag in tag_list:
        val=len(DF[DF[topic_num+'-topic']==tag])
        size_dict[str(tag)]=val

    for tag in tag_list:
        size_dict[str(tag)]=np.around(size_dict[str(tag)]*20/max(list(size_dict.values()))+7,2)

    sorted_dict={}
    for w in sorted(size_dict, key=size_dict.get):
        sorted_dict[w]=size_dict[w]

    return sorted_dict

def lda_scatter(DF,sorted_dict,topic_num,i,title,location,var='62.3'):
    
    sample_size=len(DF)
    
    topic_num=str(topic_num)
    print("\nCreating LDA plot...")

    if len(DF)>=50000:
        s=10
    else:
        s=40

    palette = np.array(sns.color_palette("hls", len(sorted_dict)+5))

    f = plt.figure(figsize=(12, 12))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(DF['x-label-'+str(i)], DF['y-label-'+str(i)], s,lw=0,c=palette[DF[topic_num+'-topic'].astype(np.int)])
    ax.axis('off')
    ax.axis('tight')

    txts = []
    for item in sorted_dict.keys():
        # Position of each label.
        xtext, ytext = np.median(DF[DF[topic_num+'-topic']==float(item)][['x-label-'+str(i),'y-label-'+str(i)]],axis=0)
        if not math.isnan(xtext) and not math.isnan(ytext):
            txt = ax.text(xtext, ytext, item, fontsize=sorted_dict[item])
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)

    PCA_text='PCA: 100 \nTotal Variance: '+var

    plt.title('Cluster plot of '+str(sample_size)+' Abstracts with '+topic_num+' topics')
    plt.annotate(PCA_text, xy=(0,0), xytext=(12, 80), va='top',xycoords='axes fraction', textcoords='offset points')
    plt.savefig(os.path.join(location, 'LDA plot.png'))
    print("\nLDA plot saved as {} at {}".format(title+'.png',location))
    plt.show()
    
    return None


# In[74]:


def color(DF,X):
    if DF[X]:
        return 'red'
    else:
        return'grey'
    
def size(DF,X):
    if DF[X]:
        return 20
    else:
        return 5


# In[75]:


def chem_finder(DF,column_name):
    
    doc=Doc(DF[column_name])
    records=doc.records.serialize()
    
    try:
        chem_rec='_'
        for rec in records:
            
            if 'names' in str(rec.keys()):
                chem_rec+=', '+rec['names'][0]
            else:
                pass
        
    except:
        return None
    
    return chem_rec


# In[76]:


def bar_plotter(dictionary, number, xlabel, ylabel, title, location, xticks= False, save=True):
    plt.figure(figsize=(7,7))
    plt.bar(list(dictionary.keys())[:number], list(dictionary.values())[:number])
    plt.xlabel(xlabel)
    if xticks:
        plt.xticks(rotation=90)
    plt.ylabel(ylabel)
    plt.title(title)
    if save: 
        plt.savefig(location)
    plt.show
    
    return None


# In[89]:


class Document:
    
    def __init__(self,address):
        
        self.session_id=self.randomString()
        
        if type(address)==str:
            self.DF=pd.read_csv(address)
        elif isinstance(address, pd.DataFrame):
            self.DF=address.copy()
        
        self.size=len(self.DF)
        
        self.lda_model=[]
        
        self.caption_DF=pd.DataFrame()
        self.lda_topic_DF=pd.DataFrame()
        
        self.caption_labels_dictionary={}
        self.chemicals_dictionary={}
        self.tag_dict={}
        self.lda_dict={}
        
        self.location=os.path.join(os.getcwd(),self.session_id)
        os.mkdir(self.location)
        
    def randomString(self,stringLength=8):
        lettersAndDigits = string.ascii_letters + string.digits
        return ''.join(random.choice(lettersAndDigits) for i in range(stringLength))
    
    def generate_Captions(self,PII='PII',Query='Query',base=os.getcwd()):
        Cap_list=[]
        for row in self.DF.index:
            pii=self.DF.loc[row,PII]
            Q=self.DF.loc[row,Query]
            TempDF=folder_search(Q,pii,base)
            Cap_list.append(TempDF)
            
        self.caption_DF=pd.concat(Cap_list)
        
        print("Caption generation complete. \nNo of Captions found : {}".format(len(self.caption_DF)))
            
        return None
    
    def plot_caption_cluster(self, iterations=1):
        prepared=preparer(self.caption_DF)
        
        for i in range(iterations):
            TDF=vectorizer(prepared,self.caption_DF,i) 
            title='Caption cluster plot of '+str(len(self.caption_DF))+'Captions-'+str(i)
            ccp_scatter(self.caption_DF,self.caption_labels_dictionary,title,self.location, i)
            
        return None
        
    def label_Captions(self,tag_list=tag_list):
        
        if len(self.caption_DF)>0:
            self.caption_labels_dictionary=caption_labeller(self.caption_DF,tag_list=tag_list)
        else:
            print("No captions found. Run DF.generate_Captions()")
        
        return None
    
    def run_LDA(self,column_name, topic_num, passes=10, save= True):
        
        print("Preparing data to run LDA...")
        corpus, id2word=process_data(self.DF, column_name)

        print('Running LDA...')

        start=time.time()
        lda_model = gensim.models.LdaMulticore(corpus,num_topics=topic_num, id2word=id2word, passes=passes, workers=None,
                                               chunksize=1000)
        stop=time.time()

        print("LDA complete")
        print("Total time: {}".format(stop-start))

        print("\nStoring data in dataframe")
        
        self.lda_model.append((topic_num,lda_model, self.session_id))
        
        if save:
            F=open(os.path.join(self.location,'Lda_model.pickle'),'wb')
            pickle.dump(self.lda_model,F)
            F.close()

        for i,j in enumerate(self.DF.index):
            index, score = sorted(lda_model[corpus[i]], key=lambda tup: -1*tup[1])[0]
            self.DF.loc[j,str(topic_num)+'-topic']=index
            self.DF.loc[j,str(topic_num)+'-topic-score']=score

        print("Done")
        
        return None
        
    
    def generate_LDA_plot(self,column_name,topic_num, repeats=1, pca=True):
        
        if len(self.lda_model)>0:
            
            for i in range(repeats):
                column_name='Abstract_x'
                title='LDA Plot for '+str(self.size)+' '+column_name+' for '+str(topic_num)+'topics-'+str(i)
                sims=sim_calculator(self.DF, column_name)
                var=tsne(sims,topic_num,self.DF,i)
                sorted_dict=dictionarizer(self.DF,topic_num)
                lda_scatter(self.DF,sorted_dict,topic_num,i,title,self.location,var)
        
        else:
            print("No lda models found. Please run DF.run_LDA()")
                
        return None
    
    def generate_LDA_topic_df(self, n=10, save=False):
        topic_list=[]
        for topic in self.lda_model[0][1].print_topics(-1):
            titles=topic[1].split("\"")
            A=2*n #len(titles)
            word_list=[]
            for i in range(1,A,2):
                word_list.append(titles[i])
            topic_list.append(word_list)  
            
        X=np.arange(len(self.lda_model[0][1].print_topics(-1)))
        DF=pd.DataFrame([X,topic_list]).T
        DF.columns=['Topic No','Keywords']
        DF.set_index('Topic No', inplace=True)
        
        self.lda_topic_DF=DF
        print(self.lda_topic_DF)
        
        if save:
            self.lda_topic_DF.to_csv(os.path.join(self.location,'lda-topics-df.csv'))
            
        return None
    
    def extract_chemicals(self, column_name):
        
        print("Starting Chemical Extraction")
        self.DF['Chemicals']=None
        self.DF['Chemicals']=self.DF.apply(chem_finder, args=(column_name,),axis=1)
        print("Chemical Extraction complete")
        
        return None
    
    def create_chemical_dictionary(self):
        print("\nGenerating Chemical Dictionary")
        chem_dict={}
        
        for no,index  in enumerate(self.DF.index):
            
            chemicals=self.DF.loc[index,'Chemicals']
            
            if type(chemicals)==str:
                
                tokens=chemicals.split(',')[1:]
                for tok in tokens:
                    tok=tok.strip()
                    if tok in chem_dict.keys():
                        chem_dict[tok]+=1
                    else:
                        chem_dict[tok]=1
                        
        self.chemicals_dictionary = dict(sorted(chem_dict.items(), key=lambda x: x[1], reverse=True))
                        
        print("Chemical Dictionary generated")
        
        F=open('chemical_dictionary.pickle','rb')
        Chem_dict=pickle.load(F)
        F.close()
        
        print("Creating Chemical columns")
        
        for el in ELEMENTS:
            self.DF[el]=0

        for no, index in enumerate(self.DF.index):
            
            chemicals=self.DF.loc[index,'Chemicals']
            
            if type(chemicals)==str:
                
                Chem_list=[i.strip() for i in chemicals.split(',')]
                
                for c in Chem_list:
                    if c in Chem_dict.keys():
                        for j in Chem_dict[c].keys():
                            self.DF.loc[index,j]=1
        
        
        return None
    
    def generate_elemental_maps(self):
        j=0    
    
        for i in ELEMENTS:
            self.DF['color']=self.DF.apply(color,args=[i,],axis=1)
            self.DF['size']=self.DF.apply(size,args=[i,],axis=1)
            
            f = plt.figure(figsize=(12, 12))
            ax = plt.subplot(aspect='equal')
            
            sc = ax.scatter(self.DF['x-label-'+str(j)],self.DF['y-label-'+str(j)], s=self.DF['size'],lw=0,c=self.DF['color'])
            
            ax.title.set_text(i)
            ax.axis('off')
            ax.axis('tight')
            plt.savefig(os.path.join(self.location,i+'_15.png'))
            plt.show()
            plt.close()    
            
        return None
    
    def generate_records(self,topic_num):
        lda_dict={}
        topic_set=set(self.DF[str(topic_num)+'-topic'])
        for topic in topic_set:
            lda_dict[topic]=len(self.DF[self.DF[str(topic_num)+'-topic']==topic])
        lda_dict=dict(sorted(lda_dict.items(), key=lambda x: x[1], reverse=True))
        
        self.lda_dict=lda_dict
        if len(self.lda_dict)>0:
            bar_plotter(self.lda_dict, 15, 'Topic No', 'Count', 'LDA Topic Distribution', os.path.join(self.location, 'lda_topic_dist.jpg'), save=True)
            
        tag_dict={}
        tag_set=set(self.caption_DF['tag'])
        for tag in tag_set:
            tag_dict[tag]=len(self.caption_DF[self.caption_DF['tag']==tag])
        tag_dict=dict(sorted(tag_dict.items(), key=lambda x: x[1], reverse=True))
        
        self.tag_dict=tag_dict
        if len(self.tag_dict)>0:
            bar_plotter(self.tag_dict, 25, 'Tags', 'Count', 'Tags Distribution', os.path.join(self.location, 'tags_dist.jpg'), xticks=True, save=True)           
        
        if len(self.chemicals_dictionary)>0:
            bar_plotter(self.chemicals_dictionary, 25, 'Chemicals', 'Count', 'Chemical Distribution', os.path.join(self.location, 'chem_dist.jpg'), xticks=True, save=True)
        
        return None
    
    def save(self):
        self.DF.to_csv(os.path.join(self.location, 'DataRecords.csv'))
        self.caption_DF.to_csv(os.path.join(self.location, 'Captions.csv'))
        
        return None


# In[ ]:




