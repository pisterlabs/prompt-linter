import os,pickle,random 
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from gensim import corpora, models
from gensim.models.phrases import Phrases, Phraser
import gensim
from sklearn.metrics import fbeta_score
from sklearn.metrics import f1_score
from gensim.models import CoherenceModel
from scipy.stats import entropy
import logging
import Model

class Validation(Model.Model):
    def __init__(self):
        super().__init__()
        self.df=Model.Model().df_pkl
        self.EC_train,self.EC_test,self.train_df,self.test_df=Model.Model().get_train_and_test(Model.Model().df_pkl)
        self.dictionary,self.corpus=Model.Model().get_dict_corpus(self.EC_train)
        #self.model=gensim.models.ldamodel.LdaModel.load('models/lda_train_150topics_100passes.model')
        self.path='images/'
        self.total_words=list(set([j for i in self.EC_train for j in i]))

    def plot_a_distribution(self,listtoplot,xlabel,xlim_tuple,ylabel,title):
        plt.figure(figsize=(8,5))
        sns.set_style("whitegrid")
        g=sns.distplot(listtoplot,kde=True)
        sns.set(font_scale=1.5);
        g.set_xlabel(xlabel,fontsize=20,);
        g.set_ylabel(ylabel,fontsize=20,);
        g.set_title(title,fontsize=20,pad=30)
        g.set_xlim([xlim_tuple[0],xlim_tuple[1]])
        plt.tight_layout()
        plt.savefig(self.path+title+'.png',type='png');

    def get_topic_similarity(self,topics,model,distance='jaccard'):
        mdiff, annotation = model.diff(model, distance=distance, num_words=100)
        fig, ax = plt.subplots(figsize=(18, 14))
        data = ax.imshow(mdiff, cmap='RdBu_r', origin='lower')
        plt.title(distance+'distance')
        plt.colorbar(data)
        sim_list=[j for i in mdiff for j in i]
        plt.savefig(self.path+distance+'.png',type='png');
        self.plot_a_distribution(sim_list,xlabel='Similarity',
                                xlim_tuple=(0,1),
                                ylabel='Density',title='Distance similarity between topics')
        return sim_list

    def compare_2_lists(self,list_sol,list_true):
        TP=len([i for i in list_sol if i in list_true])
        FP=len([i for i in list_sol if i not in list_true])
        FN=len([i for i in list_true if i not in list_sol])
        TN=len([i for i in [i for i in self.total_words if i not in list_true] 
                    if i not in list_sol])
        return (TP,FP,FN,TN)

    def get_specificity(self,input_tuple):
        if (input_tuple[1]+input_tuple[3])>0:
            TPR=input_tuple[3]/(input_tuple[1]+input_tuple[3])
        else:
            TPR=0.0
        return TPR

    def get_recall(self,input_tuple):
        if (input_tuple[0]+input_tuple[2])>0:
            recall=input_tuple[0]/(input_tuple[0]+input_tuple[2])
        else:
            recall=0.0
        return recall

    def get_precision(self,input_tuple):
        if (input_tuple[0]+input_tuple[1])>0:
            precision=(input_tuple[0]/(input_tuple[0]+input_tuple[1]))
        else:
            precision=0.0
        return precision

    def get_accuracy(self,input_tuple):
        accuracy=((input_tuple[0]+input_tuple[3])/
                (input_tuple[0]+input_tuple[1]+input_tuple[2]+input_tuple[3]))
        return accuracy

    def get_total_words(self,df):
        EC_unique=set([j for i in df['EC'].to_list() for j in i])
        return EC_unique

    def convert_list_to_df(self,list_to_df):
        d={'EC': list_to_df}
        df=pd.DataFrame(d)
        return df
        
    def compare_sol_true(self,test,true):
        input_tuple=self.compare_2_lists(test,true)
        return input_tuple

    def get_solution(self,bow,model):
        Total_sol_enz_list=[]
        doc_distribution=model.get_document_topics(bow=bow,minimum_probability= 0.01)
        idx=[i[0] for i in sorted(doc_distribution, key = lambda x: x[1])[::-1] ]
        for i in idx:
            #print ("fetching topic {}".format(i))
            topic=model.print_topic(topicno=i,topn=3500)
            #take the entire topic strip and make it readable
            entire_list=topic.strip().split('+')
            #fetch all enzymes from the readable topic list
            enzyme_list=[topic.strip().split('+')[i].split('*')[1].replace("\"","").replace(" ","" )
                        for i in range(0,len(entire_list))]
            #only take enzymes/words that have a greater than zero probability
            prob_list=[topic.strip().split('+')[i].split('*')[0] 
                        for i in range(0,len(entire_list))
                        if float(topic.strip().split('+')[i].split('*')[0]) >0.0]
            Total_sol_enz_list=Total_sol_enz_list+enzyme_list[:len(prob_list)]
        #return unique set of enzymes
        Total_sol_enz_list=list(set(Total_sol_enz_list))
        return Total_sol_enz_list

    def solution_for_one_random_doc(self,df,dictionary):
        random_pathway_index = np.random.randint(len(df))
        bow = self.dictionary.doc2bow(df.iloc[random_pathway_index,2])
        pathway=df.iloc[random_pathway_index,1]
        true= list(set(df.iloc[random_pathway_index,2]))
        total_sol_enz_list=self.get_solution(bow)
        #print ("test pathway:{}".format(pathway))
        return tota_sol_enz_list,true

    def evaluate_one_random_doc(self,df):
        Total_sol_enz_list,true=solution_for_one_random_doc(df)
        input_tuple=self.compare_sol_true(Total_sol_enz_list,true)
        return input_tuple

    def solution_for_one_df(self,df,dictionary,model):
        recall_list=[]; precision_list=[];accuracy_list=[];F1_list=[];specificity_list=[]
        for index in range(0,len(df)):
            bow = dictionary.doc2bow(df.iloc[index,2])
            pathway=df.iloc[index,1]
            true= list(set(df.iloc[index,2]))
            #print ("test pathway:{}".format(pathway))
            total_sol_enz_list=self.get_solution(bow,model)
            input_tuple=self.compare_sol_true(total_sol_enz_list,true)
            print ("test pathway:{}".format(pathway))
            print ("Tp,Fp,Fn,Tn:{}; true length:{}".format(input_tuple,len(true)))
            recall_list.append(self.get_recall(input_tuple))
            precision_list.append(self.get_precision(input_tuple))
            accuracy_list.append(self.get_accuracy(input_tuple))
            specificity_list.append(self.get_specificity(input_tuple))
            if (self.get_recall(input_tuple)+self.get_precision(input_tuple))>0:
                F1_list.append((2*self.get_precision(input_tuple)*self.get_recall(input_tuple))/
                        (self.get_recall(input_tuple)+self.get_precision(input_tuple)))
            else:
                F1_list.append(0)
        return (recall_list,precision_list,accuracy_list,F1_list,specificity_list)

    def print_testing_results(self,tuple_of_lists):
        recall_list,precision_list,accuracy_list,F1_list,specificity_list=tuple_of_lists
        print ("Recall mean: {} and median {}".format(np.mean(recall_list),np.median(recall_list)))
        print ("Precision mean: {} and median {}".format(np.mean(precision_list),np.median(precision_list)))
        print ("Accuracy mean: {} and median {}".format(np.mean(accuracy_list),np.median(accuracy_list)))
        print ("F1 mean: {} and median {}".format(np.mean(F1_list),np.median(F1_list)))
        print ("specificity mean: {} and median {}".format(np.mean(specificity_list),np.median(specificity_list)))

    def nfold_cv(self,n=1):
        mean_recall=[];mean_precision=[];mean_acc=[];mean_f1=[];mean_specificity=[]
        for i in range(0,n):
            EC_list_train,EC_list_test,train_df,test_df=Model.Model().get_train_and_test(self.df)
            dictionary,corpus=Model.Model().get_dict_corpus(EC_list_train)
            model=Model.Model().MyLDA(corpus=corpus,
                                     dictionary=dictionary,
                                     num_topics=100,
                                     random_state=200,
                                     passes=100
                                    )
            tuple_of_lists=self.solution_for_one_df(test_df,dictionary,model)
            recall_list,precision_list,accuracy_list,F1,specificity_list=tuple_of_lists
            mean_recall.append(np.mean(recall_list))
            mean_precision.append(np.mean(precision_list))
            mean_acc.append(np.mean(accuracy_list))
            mean_f1.append(np.mean(F1))
            mean_specificity.append(np.mean(specificity_list))
        return (mean_recall,mean_precision,mean_acc,mean_f1,mean_specificity)

    def jensen_shannon(self,query, matrix):
        # lets keep with the p,q notation above
        p = query[None,:].T # take transpose
        q = matrix.T # transpose matrix
        m = 0.5*(p + q)
        return np.sqrt(0.5*(entropy(p,m) + entropy(q,m)))

    def get_similarity(self,query,matrix):
        sims = self.jensen_shannon(query,matrix) # list of jensen shannon distances
        return sims # the top k positional index of the smallest Jensen Shannon distances

    def topic_array_from_train_df(self,df,model,dictionary):
        train_pathway_dict={}
        for i in range(0,len(df)):
            bow = dictionary.doc2bow(df.iloc[i,2])
            pathway=df.iloc[i,1]
            doc_distribution=model.get_document_topics(bow=bow,minimum_probability= 0.00)
            compare_doc=np.stack([np.array([tup[1] for tup in doc_distribution])])
            train_pathway_dict[pathway]=compare_doc
        return train_pathway_dict

    def topic_array_from_test_df(self,df,model,dictionary):
        test_pathway_dict={}
        for i in range(0,len(df)):
            bow = dictionary.doc2bow(df.iloc[i,2])
            pathway=df.iloc[i,1]
            doc_distribution=model.get_document_topics(bow=bow,minimum_probability= 0.00)
            test_doc=np.array([tup[1] for tup in doc_distribution])
            test_pathway_dict[pathway]=test_doc
        return test_pathway_dict

    def compare_test_train_docs(self,df_test,df_train,model,dictionary):
        test_dict=self.topic_array_from_test_df(df_test,model,dictionary)
        train_dict=self.topic_array_from_train_df(df_train,model,dictionary)
        cols = ['Pathway_test', 'Pathway_train', 'Similarity']
        lst=[];
        for pathway_test,test_doc in test_dict.items():
            for pathway_train,train_doc in train_dict.items():
                most_sim_ids = self.get_similarity(test_doc,train_doc)
                lst.append([pathway_test,pathway_train,most_sim_ids[0]])
        df=pd.DataFrame(lst,columns=cols)
        return (df)

    def print_heat_map_for_one(self,dist_df,test_df):
        for i in range(0,len(test_df)):
            df_sorted=dist_df.loc[dist_df['Pathway_test'] == test_df['Name'].to_list()[i]].sort_values(by=['Similarity']).head()
            Z=[df_sorted['Similarity'].to_list()]
            title=test_df['Name'].to_list()[i]
            x_axis=df_sorted['Pathway_train'].to_list()
            plt.figure(figsize=(12,2))
            ax0=plt.subplot()
            c = ax0.pcolor(Z,cmap='RdBu', vmin=0, vmax=1,edgecolors='k', linewidths=1)
            labels = [item.get_text() for item in ax0.get_xticklabels()]
            labels = x_axis
            ax0.set_xticklabels(labels,rotation=90,ha='right', minor=False)
            ax0.set_yticklabels([],rotation=90,ha='right', minor=False)
            ax0.set_title(title)
            plt.colorbar(c, ax=ax0)
            plt.show()
            plt.tight_layout()
            plt.savefig(self.path+'similarityheatmap',type='png');

    def print_heat_map_for_one_FT(self,dist_df,test_df):
        for i in range(0,len(test_df)):
            df_sorted=dist_df.loc[dist_df['Pathway_test'] == test_df['Name'].to_list()[i]].sort_values(by=['Similarity']).tail()
            Z=[df_sorted['Similarity'].to_list()]
            title=test_df['Name'].to_list()[i]
            x_axis=df_sorted['Pathway_train'].to_list()
            plt.figure(figsize=(12,2))
            ax0=plt.subplot()
            c = ax0.pcolor(Z,cmap='RdBu_r', vmin=0.0, vmax=1,edgecolors='k', linewidths=1)
            labels = [item.get_text() for item in ax0.get_xticklabels()]
            labels = x_axis
            ax0.set_xticklabels(labels,rotation=90,ha='right', minor=False)
            ax0.set_yticklabels([],rotation=90,ha='right', minor=False)
            ax0.set_title(title)
            plt.colorbar(c, ax=ax0)
            plt.show()
            plt.tight_layout()
            plt.savefig(self.path+'similarityheatmap',type='png');


if __name__=='__main__':
    Validation()
