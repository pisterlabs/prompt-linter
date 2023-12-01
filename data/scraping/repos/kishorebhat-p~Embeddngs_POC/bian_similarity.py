
#!pip install  num2words matplotlib plotly scipy scikit-learn pandas tiktoken
#!pip install openai

import openai
import os
import pandas as pd
import numpy as np
from openai.embeddings_utils import get_embedding, cosine_similarity

def convertfileToArray(argument):
  filename= "./persistnpy/" + argument + ".txt.npy"
  data2=np.load(filename)
  return data2

def convertbianfileToArray(argument):
  filename= "./bianembed/" + argument + ".txt.npy"
  data2=np.load(filename)
  return data2

def remove(string):
    return string.replace(" ", "")

#df=pd.read_excel('App_Short.xlsx')
df=pd.read_excel('AppInventory_Sample_0822.xlsx')
df=df.sort_values("IDENTIFIER", ascending=True)
df['embedding']=df.IDENTIFIER.apply(lambda x: convertfileToArray(str(x)))
top_n=6

df_bian=pd.read_excel('./bian_data_v2.xlsx')
df_bian['Filename'] = df_bian.apply(lambda x: remove(x.Name), axis=1)
df_bian['embedding']=df_bian.Filename.apply(lambda x: convertbianfileToArray(str(x)))


total_df = pd.DataFrame()

for i in range(len(df)):
    identifier=df.loc[i,'IDENTIFIER']
    embedding=df.loc[i,'embedding']
    softdesc=df.loc[i,'Software Description']
    appName=df.loc[i,'App Name']
    df_bian["similarities"] = df_bian.embedding.apply(lambda x: cosine_similarity(x, embedding))
    df1=df_bian[df_bian['similarities']> 0.8]
    if df1.empty == False :
      print("the data has value " + str(identifier))
      df1['UBS_IDENTIFIER']=identifier
      df1['UBS_SOFT_DESC']=softdesc
      df1['UBS_APP_NAME']=appName
      df1=df1.drop(columns=['embedding'])
      total_df = pd.concat([total_df,df1])

total_df.to_excel("./similarity/bianconsolidated_data.xlsx")
