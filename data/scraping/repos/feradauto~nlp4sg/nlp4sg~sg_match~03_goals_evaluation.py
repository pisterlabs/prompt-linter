import os
import sys
import openai
import pandas as pd
import numpy as np
import argparse
from transformers import BloomTokenizerFast, BloomModel,BloomForCausalLM
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as skm
import ast

mapping={
'No Poverty':'01.No Poverty',
'Zero Hunger':'02.Zero Hunger',
'Good Health and Well-Being':'03.Good Health and Well-Being',
'Quality Education':'04.Quality Education',
'Gender Equality':'05.Gender Equality',
'Clean Water and Sanitation':'06.Clean Water and Sanitation',
'Affordable and Clean Energy':'07.Affordable and Clean Energy',
'Decent Work and Economic Growth':'08.Decent Work and Economic Growth',
'Industry, Innovation and Infrastructure':'09.Industry, Innovation and Infrastructure',
'Reduced Inequalities':'10.Reduced Inequalities',
'Sustainable Cities and Communities':'11.Sustainable Cities and Communities',
'Responsible Consumption and Production':'12.Responsible Consumption and Production',
'Climate Action':'13.Climate Action',
'Life Below Water':'14.Life Below Water',
'Life on Land':'15.Life on Land',
'Peace, Justice and Strong Institutions':'16.Peace, Justice and Strong Institutions',
'Partnership for the Goals':'17.Partnership for the Goals',
    'Partnership for the goals':'17.Partnership for the Goals',
}

goal_list=[
'01.No Poverty',
'02.Zero Hunger',
'03.Good Health and Well-Being',
'04.Quality Education',
'05.Gender Equality',
'06.Clean Water and Sanitation',
'07.Affordable and Clean Energy',
'08.Decent Work and Economic Growth',
'09.Industry, Innovation and Infrastructure',
'10.Reduced Inequalities',
'11.Sustainable Cities and Communities',
'12.Responsible Consumption and Production',
'13.Climate Action',
'14.Life Below Water',
'15.Life on Land',
'16.Peace, Justice and Strong Institutions',
'17.Partnership for the Goals'
    ]

def get_sn_rules_classification(df_classification):
    sd_pivot=pd.melt(df_classification,id_vars=['ID'],
            value_vars=['Reduced Inequalities', 'Sustainable Cities and Communities',
           'Responsible Consumption and Production', 'Good Health and Well-Being',
           'Life on Land', 'Life Below Water',
           'Peace, Justice and Strong Institutions',
           'Decent Work and Economic Growth', 'Partnership for the Goals',
           'Affordable and Clean Energy', 'Clean Water and Sanitation',
           'Industry, Innovation and Infrastructure', 'Quality Education',
           'Gender Equality', 'No Poverty', 'Climate Action', 'No Hunger',
            'Disinformation and fake news', 'Privacy protection', 'Deception detection','Hate speech'],value_name='proba',var_name="social_need")

    sd_pivot=sd_pivot.sort_values('proba',ascending=False)
    sd_pivot=sd_pivot.groupby(['ID']).head(2)
    sd_pivot=sd_pivot.assign(place=sd_pivot.groupby(['ID']).cumcount())
    sd_pivot=sd_pivot.fillna("")
    sn_top2=pd.pivot_table(sd_pivot,index=['ID'],columns=['place'],values=['social_need'],
                  aggfunc=np.sum)
    sn_top2=sn_top2.reset_index()
    sn_top2.columns=[col[0]+""+str(col[1]) for col in sn_top2.columns]
    sd_pivot_max=sd_pivot.groupby(['ID']).proba.max().reset_index().rename(columns={'proba':'proba_max'})
    df_classification=df_classification.merge(sn_top2,on=['ID'],how='left').merge(sd_pivot_max,on=['ID'])

    df_sn=df_classification.loc[:,['ID','social_need0','social_need1',"proba_max"]]
    return df_sn

def get_sn_rules_classification_normal(df_classification):
    sd_pivot=pd.melt(df_classification,id_vars=['ID'],
            value_vars=['Reduced Inequalities', 'Sustainable Cities and Communities',
           'Responsible Consumption and Production', 'Good Health and Well-Being',
           'Life on Land', 'Life Below Water',
           'Peace, Justice and Strong Institutions',
           'Decent Work and Economic Growth', 'Partnership for the Goals',
           'Affordable and Clean Energy', 'Clean Water and Sanitation',
           'Industry, Innovation and Infrastructure', 'Quality Education',
           'Gender Equality', 'No Poverty', 'Climate Action', 'No Hunger'],value_name='proba',var_name="social_need")

    sd_pivot=sd_pivot.sort_values('proba',ascending=False)
    sd_pivot=sd_pivot.groupby(['ID']).head(2)
    sd_pivot=sd_pivot.assign(place=sd_pivot.groupby(['ID']).cumcount())
    sd_pivot=sd_pivot.fillna("")
    sn_top2=pd.pivot_table(sd_pivot,index=['ID'],columns=['place'],values=['social_need'],
                  aggfunc=np.sum)
    sn_top2=sn_top2.reset_index()
    sn_top2.columns=[col[0]+""+str(col[1]) for col in sn_top2.columns]
    sd_pivot_max=sd_pivot.groupby(['ID']).proba.max().reset_index().rename(columns={'proba':'proba_max'})
    df_classification=df_classification.merge(sn_top2,on=['ID'],how='left').merge(sd_pivot_max,on=['ID'])

    df_sn=df_classification.loc[:,['ID','social_need0','social_need1',"proba_max"]]
    return df_sn

def get_gold_and_predicted(df):
    prediction=[]
    for i,d in df.iterrows():
        pred=[]
        if d['sdg1']==1:
            pred.append(goal_list[0])
        if d['sdg2']==1:
            pred.append(goal_list[1])
        if d['sdg3']==1:
            pred.append(goal_list[2])
        if d['sdg4']==1:
            pred.append(goal_list[3])
        if d['sdg5']==1:
            pred.append(goal_list[4])
        if d['sdg6']==1:
            pred.append(goal_list[5])
        if d['sdg7']==1:
            pred.append(goal_list[6])
        if d['sdg8']==1:
            pred.append(goal_list[7])
        if d['sdg9']==1:
            pred.append(goal_list[8])
        if d['sdg10']==1:
            pred.append(goal_list[9])
        if d['sdg11']==1:
            pred.append(goal_list[10])
        if d['sdg12']==1:
            pred.append(goal_list[11])
        if d['sdg13']==1:
            pred.append(goal_list[12])
        if d['sdg14']==1:
            pred.append(goal_list[13])
        if d['sdg15']==1:
            pred.append(goal_list[14])
        if d['sdg16']==1:
            pred.append(goal_list[15])
        if d['sdg17']==1:
            pred.append(goal_list[16])
        prediction.append(pred)

    actual=[]
    for i,d in df.iterrows():
        act=[d['Goal1_numeric']]
        if d['Goal2_filled']!='':
            act.append(d['Goal2_filled'])
        if d['Goal3_filled']!='':
            act.append(d['Goal3_filled'])
        actual.append(act)
        
    return actual,prediction

def format_gold(df):
    df['Goal1_numeric']=df.goal1.replace(mapping)
    df['Goal2_numeric']=df.goal2.replace(mapping)
    df['Goal3_numeric']=df.goal3.replace(mapping)
    df['Goal2_filled']=df.Goal2_numeric.fillna('')
    df['Goal3_filled']=df.Goal3_numeric.fillna('')
    return df

def extract_predictions_gpt3(df):
    df=df.assign(sdg1=np.where(df['GPT3_response'].str.lower().str.contains("goal 1 |goal 1:|poverty"),1,0))
    df=df.assign(sdg2=np.where(df['GPT3_response'].str.lower().str.contains("goal 2|hunger"),1,0))
    df=df.assign(sdg3=np.where(df['GPT3_response'].str.lower().str.contains("goal 3|health"),1,0))
    df=df.assign(sdg4=np.where(df['GPT3_response'].str.lower().str.contains("goal 4|education"),1,0))
    df=df.assign(sdg5=np.where(df['GPT3_response'].str.lower().str.contains("goal 5|gender"),1,0))
    df=df.assign(sdg6=np.where(df['GPT3_response'].str.lower().str.contains("goal 6|clean water"),1,0))
    df=df.assign(sdg7=np.where(df['GPT3_response'].str.lower().str.contains("goal 7|clean energy"),1,0))
    df=df.assign(sdg8=np.where(df['GPT3_response'].str.lower().str.contains("goal 8|decent work"),1,0))
    df=df.assign(sdg9=np.where(df['GPT3_response'].str.lower().str.contains("goal 9|industry|innovation"),1,0))
    df=df.assign(sdg10=np.where(df['GPT3_response'].str.lower().str.contains("goal 10|inequal"),1,0))
    df=df.assign(sdg11=np.where(df['GPT3_response'].str.lower().str.contains("goal 11|sustainable cities"),1,0))
    df=df.assign(sdg12=np.where(df['GPT3_response'].str.lower().str.contains("goal 12|responsible consumption"),1,0))
    df=df.assign(sdg13=np.where(df['GPT3_response'].str.lower().str.contains("goal 13|climate"),1,0))
    df=df.assign(sdg14=np.where(df['GPT3_response'].str.lower().str.contains("goal 14|life below water"),1,0))
    df=df.assign(sdg15=np.where(df['GPT3_response'].str.lower().str.contains("goal 15|life on land"),1,0))
    df=df.assign(sdg16=np.where(df['GPT3_response'].str.lower().str.contains("goal 16|peace|justice"),1,0))
    df=df.assign(sdg17=np.where(df['GPT3_response'].str.lower().str.contains("goal 17|partnership"),1,0))
    return df

def extract_predictions_mnli(df):
    mapping_mnli={
    'No Poverty':'sdg1',
    'Zero Hunger':'sdg2',
    'No Hunger':'sdg2',
    'Good Health and Well-Being':'sdg3',
    'Quality Education':'sdg4',
    'Gender Equality':'sdg5',
    'Clean Water and Sanitation':'sdg6',
    'Affordable and Clean Energy':'sdg7',
    'Decent Work and Economic Growth':'sdg8',
    'Industry, Innovation and Infrastructure':'sdg9',
    'Reduced Inequalities':'sdg10',
    'Sustainable Cities and Communities':'sdg11',
    'Responsible Consumption and Production':'sdg12',
    'Climate Action':'sdg13',
    'Life Below Water':'sdg14',
    'Life on Land':'sdg15',
    'Hate speech':'sdg16',
    'Disinformation and fake news':'sdg16',
    'Deception detection':'sdg16',
    'Privacy protection':'sdg16',
    'Peace, Justice and Strong Institutions':'sdg16',
    'Partnership for the Goals':'sdg17',
    }


    
    df=df.assign(sdg1=np.where(df['No Poverty']>=0.5,1,0))
    df=df.assign(sdg2=np.where(df['No Hunger']>=0.5,1,0))
    df=df.assign(sdg3=np.where(df['Good Health and Well-Being']>=0.5,1,0))
    df=df.assign(sdg4=np.where(df['Quality Education']>=0.5,1,0))
    df=df.assign(sdg5=np.where(df['Gender Equality']>=0.5,1,0))
    df=df.assign(sdg6=np.where(df['Clean Water and Sanitation']>=0.5,1,0))
    df=df.assign(sdg7=np.where(df['Affordable and Clean Energy']>=0.5,1,0))
    df=df.assign(sdg8=np.where(df['Decent Work and Economic Growth']>=0.5,1,0))
    df=df.assign(sdg9=np.where(df['Industry, Innovation and Infrastructure']>=0.5,1,0))
    df=df.assign(sdg10=np.where(df['Reduced Inequalities']>=0.5,1,0))
    df=df.assign(sdg11=np.where(df['Sustainable Cities and Communities']>=0.5,1,0))
    df=df.assign(sdg12=np.where(df['Responsible Consumption and Production']>=0.5,1,0))
    df=df.assign(sdg13=np.where(df['Climate Action']>=0.5,1,0))
    df=df.assign(sdg14=np.where(df['Life Below Water']>=0.5,1,0))
    df=df.assign(sdg15=np.where(df['Life on Land']>=0.5,1,0))
    df=df.assign(sdg16=np.where(df['Peace, Justice and Strong Institutions']>=0.5,1,0))
    df=df.assign(sdg16=np.where(df['Privacy protection']>=0.5,1,df.sdg16))
    df=df.assign(sdg16=np.where(df['Deception detection']>=0.5,1,df.sdg16))
    df=df.assign(sdg16=np.where(df['Hate speech']>=0.5,1,df.sdg16))
    df=df.assign(sdg16=np.where(df['Disinformation and fake news']>=0.5,1,df.sdg16))
    df=df.assign(sdg17=np.where(df['Partnership for the Goals']>=0.5,1,0))

    df_top=get_sn_rules_classification(df)
    df_top=get_sn_rules_classification_normal(df)
    df=df.merge(df_top,on=['ID'],how='left')
    for i,d in df.iterrows():
        df.at[i,mapping_mnli[d['social_need0']]]=1
    return df

def fix_mnli(mnli_classifier,test_set_complete):
    mnli_classifier=mnli_classifier.drop_duplicates(subset=['ID'],keep='first').reset_index(drop=True)
    mnli_classifier.loc[:,'label_complete']=mnli_classifier.iloc[:].label_complete.apply(lambda x:ast.literal_eval(x))

    mnli_classifier.loc[0,'label_complete']=[mnli_classifier.loc[0,'label_complete']]

    mnli_classifier=mnli_classifier.loc[:,['ID', 'goal1_raw', 'goal2_raw', 'goal3_raw', 'goal1', 'goal2', 'goal3',
           'title_clean', 'abstract_clean', 'title_abstract_clean', 'year', 'text',
           'label_complete','Goal', 'cosine_similarity']]

    probas=mnli_classifier.label_complete.apply(lambda x:pd.Series(x[0]))

    mnli_classifier=mnli_classifier.merge(probas,left_index=True,right_index=True)
    mnli_classifier=mnli_classifier.loc[(~(mnli_classifier.ID.isin(test_set_complete.iloc[2000:].ID.unique()))),:]
    return mnli_classifier

def assign_goal_alldf(df_all_goals):
    df_all_goals.goal2_raw=df_all_goals.goal2_raw.fillna('')
    df_all_goals.goal3_raw=df_all_goals.goal3_raw.fillna('')

    df_all_goals=df_all_goals.assign(goal1=np.where(df_all_goals['goal1_raw'].str.lower().str.contains("education"),'Quality Education',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("poverty"),'No Poverty',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("hunger"),'Zero Hunger',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("clean_water"),'Clean Water and Sanitation',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("clean_energy"),'Affordable and Clean Energy',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("life_land"),'Life on Land',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("marine_life"),'Life Below Water',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("health"),'Good Health and Well-Being',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("climate"),'Climate Action',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("peace|privacy|disinformation_and_fake_news|deception|hate"),'Peace, Justice and Strong Institutions',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("social biases|race & identity"),'Reduced Inequalities',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("industry|innovation|research"),'Industry, Innovation and Infrastructure',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("sustainable cities|sustainable_cities"),'Sustainable Cities and Communities',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("gender"),'Gender Equality',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("decent work|decent_work_and_economy"),'Decent Work and Economic Growth',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("partnership"),'Partnership for the goals',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("responsible_consumption_and_production"),'Responsible Consumption and Production',
                        np.where(df_all_goals['goal1_raw'].str.lower().str.contains("reduced|social_equality"),'Reduced Inequalities',''
                              )))))))))))))))))))

    df_all_goals=df_all_goals.assign(goal2=np.where(df_all_goals['goal2_raw'].str.lower().str.contains("education"),'Quality Education',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("poverty"),'No Poverty',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("hunger"),'Zero Hunger',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("clean_water"),'Clean Water and Sanitation',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("clean_energy"),'Affordable and Clean Energy',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("life_land"),'Life on Land',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("marine_life"),'Life Below Water',         
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("health"),'Good Health and Well-Being',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("climate"),'Climate Action',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("peace|privacy|disinformation_and_fake_news|deception|hate"),'Peace, Justice and Strong Institutions',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("social biases|race & identity"),'Reduced Inequalities',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("industry|innovation|research"),'Industry, Innovation and Infrastructure',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("sustainable cities|sustainable_cities"),'Sustainable Cities and Communities',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("gender"),'Gender Equality',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("decent work|decent_work_and_economy"),'Decent Work and Economic Growth',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("partnership"),'Partnership for the goals',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("responsible_consumption_and_production"),'Responsible Consumption and Production',
                        np.where(df_all_goals['goal2_raw'].str.lower().str.contains("reduced|social_equality"),'Reduced Inequalities',''
                              )))))))))))))))))))

    df_all_goals=df_all_goals.assign(goal3=np.where(df_all_goals['goal3_raw'].str.lower().str.contains("education"),'Quality Education',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("poverty"),'No Poverty',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("hunger"),'Zero Hunger',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("clean_water"),'Clean Water and Sanitation',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("clean_energy"),'Affordable and Clean Energy',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("life_land"),'Life on Land',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("marine_life"),'Life Below Water',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("health"),'Good Health and Well-Being',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("climate"),'Climate Action',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("peace|privacy|disinformation_and_fake_news|deception|hate"),'Peace, Justice and Strong Institutions',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("social biases|race & identity"),'Reduced Inequalities',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("industry|innovation|research"),'Industry, Innovation and Infrastructure',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("sustainable cities|sustainable_cities"),'Sustainable Cities and Communities',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("gender"),'Gender Equality',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("decent work|decent_work_and_economy"),'Decent Work and Economic Growth',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("partnership"),'Partnership for the goals',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("responsible_consumption_and_production"),'Responsible Consumption and Production',
                        np.where(df_all_goals['goal3_raw'].str.lower().str.contains("reduced|social_equality"),'Reduced Inequalities',''
                              )))))))))))))))))))
    return df_all_goals

total=[['01.No Poverty'],
['02.Zero Hunger'],
['03.Good Health and Well-Being'],
['04.Quality Education'],
['05.Gender Equality'],
['06.Clean Water and Sanitation'],
['07.Affordable and Clean Energy'],
['08.Decent Work and Economic Growth'],
['09.Industry, Innovation and Infrastructure'],
['10.Reduced Inequalities'],
['11.Sustainable Cities and Communities'],
['12.Responsible Consumption and Production'],
['13.Climate Action'],
['14.Life Below Water'],
['15.Life on Land'],
['16.Peace, Justice and Strong Institutions'],
['17.Partnership for the Goals']]


mlb = MultiLabelBinarizer()

mlb.fit(total)

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=26):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    sns.set(font_scale=2.4)
    try:
        heatmap = sns.heatmap(df_cm,square=True, annot=True, fmt="d", cmap="Blues",cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    axes.set_title(class_label)

goal_list_names=[
'G1. Poverty',
'G2. Hunger',
'G3. Health',
'G4. Education',
'G5. Gender',
'G6. Water',
'G7. Energy',
'G8. Economy',
'G9. Innovation',
'G10. Inequalities',
'G11. Sustainable Cities',
'G12. Consumption',
'G13. Climate',
'G14. Life Below Water',
'G15. Life on Land',
'G16. Peace',
'G17. Partnership'
    ]

def print_confusion_matrix2(confusion_matrix, axes, class_label, class_names, fontsize=15):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    sns.set(font_scale=1.3)
    try:
        heatmap = sns.heatmap(df_cm,square=True, annot=True, fmt="d", cmap="Blues",cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)

    axes.set_title(class_label)

def main():
    data_path="../data/"
    outputs_path="../outputs/"
    test_set_complete=pd.read_csv(outputs_path+"general/test_set_final.csv")
    train_set=pd.read_csv(outputs_path+"general/train_set_final.csv")
    dev_set=pd.read_csv(outputs_path+"general/dev_set_final.csv")

    gpt3_goals=pd.read_csv(outputs_path+"sg_goals/gpt3_un_singular_ff.csv")
    gpt3_low=pd.read_csv(outputs_path+"sg_goals/gpt3_un_singular_ff_extra.csv")
    #gpt3_goals=pd.read_csv(outputs_path+"sg_goals/gpt3_un_plural_ff.csv")
    #gpt3_low=pd.read_csv(outputs_path+"sg_goals/gpt3_un_plural_ff_extra.csv")

    #mnli_classifier=pd.read_csv(outputs_path+"sg_goals/goal_classifier_test_desc_debertav3.csv")
    #mnli_classifier=pd.read_csv(outputs_path+"sg_goals/goal_classifier_test_desc_distilbert.csv")
    #mnli_classifier=pd.read_csv(outputs_path+"sg_goals/goal_classifier_test_desc.csv")
    #mnli_classifier_extra=pd.read_csv(outputs_path+"sg_goals/goal_classifier_test_desc_extra.csv")
    #mnli_classifier=pd.concat([mnli_classifier,mnli_classifier_extra])

    gpt3_goals=pd.concat([gpt3_goals,gpt3_low])

    gpt3_goals=gpt3_goals.drop_duplicates(subset=['ID'],keep='first')


    test_set=test_set_complete.iloc[:2000]
    low_ocurrence=pd.read_csv(data_path+"test_data/low_occurrence_annotated.csv")
    low_ocurrence=low_ocurrence.rename(columns={'Most Related SG goal':'goal1_raw',
           '(if exists) 2nd Related SG Goal':'goal2_raw', '(if exists) 3rd Related SG Goal':'goal3_raw'})
    low_ocurrence=low_ocurrence.rename(columns={"SG_or_not":"label"})
    low_ocurrence["label"]=low_ocurrence["label"].fillna(0)

    df_all_goals=pd.concat([dev_set,train_set,test_set,low_ocurrence])
    df_all_goals=df_all_goals.drop_duplicates(subset=['ID'],keep='first').reset_index(drop=True)
    df_all_goals=df_all_goals.loc[df_all_goals.label==1]

    gpt3_goals=gpt3_goals.loc[gpt3_goals.ID.isin(df_all_goals.ID.unique())]
    gpt3_goals=gpt3_goals.loc[:,['ID', 'url', 'label', 'task_annotation', 'method_annotation',
           'org_annotation', 'goal1_raw', 'goal2_raw', 'goal3_raw',
           'title_abstract_clean', 'title', 'abstract', 'title_clean',
           'abstract_clean', 'acknowledgments_clean', 'text', 'year', 'Goal',
           'goal1', 'goal2', 'goal3', 'statement','full_prompt', 'GPT3_response']]


    df_all_goals=assign_goal_alldf(df_all_goals)

    gpt3_goals=gpt3_goals.loc[:,['ID', 'url', 'label', 'task_annotation', 'method_annotation',
           'org_annotation', 'goal1_raw', 'goal2_raw', 'goal3_raw',
           'title_abstract_clean', 'title', 'abstract', 'title_clean',
           'abstract_clean', 'acknowledgments_clean', 'text', 'year', 'Goal',
            'statement', 'full_prompt', 'GPT3_response']]

    gpt3_goals=gpt3_goals.merge(df_all_goals.loc[:,['ID','goal1','goal2','goal3']],on=['ID'],how='left')

    gpt3_goals=format_gold(gpt3_goals)

    df=gpt3_goals.copy()
    df=extract_predictions_gpt3(df)

    mnli_classifier=fix_mnli(mnli_classifier,test_set_complete)

    del mnli_classifier['goal1']
    del mnli_classifier['goal2']
    del mnli_classifier['goal3']

    mnli_classifier=mnli_classifier.merge(df_all_goals.loc[:,['ID','goal1','goal2','goal3']],on=['ID'],how='left')


    mnli_classifier=mnli_classifier.loc[mnli_classifier.ID.isin(df_all_goals.ID.unique())]
    mnli_classifier=format_gold(mnli_classifier)

    df=mnli_classifier.copy()
    df=extract_predictions_mnli(df)

    df=df.assign(goal2=np.where(df.goal1==df.goal2,'',df.goal2))
    df=df.assign(goal3=np.where(df.goal2==df.goal3,'',df.goal3))

    gold_labels,prediction=get_gold_and_predicted(df)

    gold = mlb.transform(gold_labels)

    pred = mlb.transform(prediction)

    cm = skm.multilabel_confusion_matrix(gold, pred)

    ## singular
    fig, ax = plt.subplots(6, 3,figsize=(15, 30))
    c=0

    for axes, cfs_matrix, label in zip(ax.flatten()[:-1], cm, goal_list_names):
        if c==0:
            print_confusion_matrix(cfs_matrix, axes, label, ["0", " 1"])
            axes.set_xlabel('Prediction',size=22)
            axes.set_ylabel('Ground truth',size=22)
        else:
            print_confusion_matrix(cfs_matrix, axes, label, ["", " "])
        c+=1

    #fig.subplots_adjust(wspace=0.1, hspace=0.4)
    #axes.set_xlabel('Prediction',size=20)
    #fig.text(0.01, 0.5, 'Ground truth', va='center', rotation='vertical')
    #fig.text(0.5, 0, 'Prediction', ha='center')
    ax[5,2].axis('off')
    fig.tight_layout()
    plt.savefig("cm_multilabel_2.pdf",dpi=400,bbox_inches='tight')
    #plt.show()


    partial=0
    total=0
    for g,p in zip(gold_labels,prediction):
        matches = set(g).intersection(set(p))
        lm=len(matches)
        lt=(len(set(g))+len(set(p)))/2
        if lm>=1:
            partial+=1
        if lm==lt:
            total+=1

    ## un singular new
    print("Partial: ",partial/len(prediction))
    print("Total: ",total/len(prediction))
    print(classification_report(gold, pred,digits=4))
if __name__ == '__main__':
    main()
