import os
import sys
import openai
import pandas as pd
import numpy as np
import argparse
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def main():
    data_path="../data/"
    outputs_path="../outputs/"
    test_set=pd.read_csv(outputs_path+"general/test_set_final.csv")
    train_set=pd.read_csv(outputs_path+"general/train_set_final.csv")
    dev_set=pd.read_csv(outputs_path+"general/dev_set_final.csv")
    low_ocurrence=pd.read_csv(data_path+"test_data/low_occurrence_annotated.csv")

    low_ocurrence=low_ocurrence.rename(columns={'Most Related SG goal':'goal1_raw',
           '(if exists) 2nd Related SG Goal':'goal2_raw', '(if exists) 3rd Related SG Goal':'goal3_raw'})
    low_ocurrence=low_ocurrence.rename(columns={"SG_or_not":"label"})
    low_ocurrence["label"]=low_ocurrence["label"].fillna(0)
    low_ocurrence.abstract_clean=low_ocurrence.abstract_clean.fillna('')
    low_ocurrence=low_ocurrence.assign(text=low_ocurrence.title_clean+". "+low_ocurrence.abstract_clean)

    df_all_goals=pd.concat([dev_set,train_set,test_set,low_ocurrence])
    df_all_goals.goal1_raw=df_all_goals.goal1_raw.fillna('')
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

    df_all_goals_sg=df_all_goals.loc[df_all_goals.label==1]

    df=df_all_goals_sg.reset_index(drop=True).copy()

    openai.api_key = os.getenv("OPENAI_API_KEY")

    preprompt="There is an NLP paper with the title and abstract:\n"
    question="Which of the UN goals does this paper directly contribute to? Provide the goal number and name."
    df=df.assign(statement=preprompt+df.text+"\n"+question)

    for i,d in df.iterrows():
        input_prompt=d['statement']
        completion = openai.Completion.create(engine="text-davinci-002", prompt=input_prompt,temperature=0,max_tokens=100,logprobs=1)
        dict_norm={}
        dict_uniques={}

        df.loc[i,'full_prompt']=input_prompt
        df.loc[i,'GPT3_response']=completion.choices[0].text


    df.to_csv(outputs_path+"sg_goals/gpt3_un_singular_ff.csv",index=False)
if __name__ == '__main__':
    main()
