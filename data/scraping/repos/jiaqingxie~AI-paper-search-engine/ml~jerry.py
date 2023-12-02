import pandas as pd
import os

all_the_keywords = []

base_df = pd.DataFrame()

clean_data_folder = "../PapersOriginal"

for filename in os.listdir(clean_data_folder):
    full_path1 = f"{clean_data_folder}/{filename}"
    for filename2 in os.listdir(full_path1):
        full_path = f"{clean_data_folder}/{filename}/{filename2}"
#         print(full_path)
        
        # load data into a DataFrame
        new_df = pd.read_csv(full_path)

        # merge into the base DataFrame
        base_df = pd.concat([base_df, new_df])
        

dropList = []
for i in range(len(base_df)):
    if len(str(base_df['abstract'].values[i]))<=100:
#         base_df = base_df.drop(i)
        dropList.append(i)
#         print(i)
base_df = base_df.drop(base_df.index[dropList])

# dropping ALL duplicate values
base_df.drop_duplicates(subset =["title", "abstract"],inplace = True)
 
    
import re
for i in range(len(base_df)):
    base_df['title'].values[i] = str(base_df['title'].values[i])
    if ';' in str(base_df['title'].values[i]):
#         base_df = base_df.drop(i)
#         dropList.append(i)
        base_df['title'].values[i] = re.sub(';', ',', base_df['title'].values[i])
    
    base_df['abstract'].values[i] = str(base_df['abstract'].values[i])
    if ';' in str(base_df['abstract'].values[i]): 
        base_df['abstract'].values[i] = re.sub(';', ',', base_df['abstract'].values[i])
    
    base_df['conference'].values[i] = str(base_df['conference'].values[i])
    
    if base_df['url'].isnull().values[i]:
        base_df['url'].values[i]= 'https://google.com/search?q=' + "+".join(base_df['title'].values[i].split())
    
    base_df['url'].values[i] = str(base_df['url'].values[i])
    if base_df['url'].values[i]=="None":
        base_df['url'].values[i]= 'https://google.com/search?q=' + "+".join(base_df['title'].values[i].split())
    
    base_df['authors'].values[i] = str(base_df['authors'].values[i])
#     base_df['citations'].values[i] = int(base_df['citations'].values[i])
    if base_df['citations'].isnull().values[i]:
        base_df['citations'].values[i]= 0
#     base_df['citations'].values[i] = int(base_df['citations'].values[i])

#attempt to convert 'rebounds' column from float to integer
base_df['citations'] = base_df['citations'].astype(int)

# Do the above again to make sure no ';'
for i in range(len(base_df)):
    base_df['title'].values[i] = str(base_df['title'].values[i])
    if ';' in str(base_df['title'].values[i]):
#         base_df = base_df.drop(i)
#         dropList.append(i)
        base_df['title'].values[i] = re.sub(';', ',', base_df['title'].values[i])
    
    base_df['abstract'].values[i] = str(base_df['abstract'].values[i])
    if ';' in str(base_df['abstract'].values[i]): 
        base_df['abstract'].values[i] = re.sub(';', ',', base_df['abstract'].values[i])
    
    base_df['conference'].values[i] = str(base_df['conference'].values[i])
    base_df['url'].values[i] = str(base_df['url'].values[i])
    base_df['authors'].values[i] = str(base_df['authors'].values[i])
#     base_df['citations'].values[i] = int(base_df['citations'].values[i])
    if base_df['citations'].isnull().values[i]:
        base_df['citations'].values[i]= 0
#     base_df['citations'].values[i] = int(base_df['citations'].values[i])


import RAKE
import operator
import nltk
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag, word_tokenize
# from nltk import StanfordTagger

# from nltk.tag.stanford import StanfordPOSTagger
# jar = "./stanford-postagger.jar"
# model = "./english-bidirectional-distsim.tagger"
# pos_tagger = StanfordPOSTagger(model, jar, encoding = "utf-8")

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stop_dir = "SmartStoplist.txt"
rake_object = RAKE.Rake(stop_dir)

def get_keywords(textContent):

    def Sort_Tuple(tup):
        tup.sort(key = lambda x: x[1])
        return tup
    def stemsentence(arr):
        return  " ".join([stemmer.stem(aa) for aa in arr.split(" ")])


    keywords = Sort_Tuple(rake_object.run(textContent))[-10:]
    keywords = keywords[::-1]
    res = []
    for keyword in keywords:
        # postag = pos_tagger.tag(word_tokenize(keyword[0]))
        postag = pos_tag(word_tokenize(keyword[0]))
        k = []
        for (a, b) in postag:
            if not 'V' in b[0]: # and not 'J' in b[0]:
                k.append(a)
        if len(k)>1:
            res.append(" ".join(k))

    res = list(set(res))

    res2 = res.copy()
    for i in range(len(res)):
        r = res[i]
        others = res[:i] + res[i+1:]
        r2 = "".join(stemsentence(r))
        others = "".join([stemsentence(other) for other in others])
        if r2 in others:
            res2.remove(r)
    
    res3 = []
    delete_keywords = ['paper','describe', 'article', 'journal', 'present', 'investigate' ]
    keep = True

    for rr in res2:
        rr_list = rr.split()
        if len(rr_list)>=2:
            for delete in delete_keywords:
                if(delete in rr):
                    keep = False
                    break
            if keep:
                if (rr[0].isalpha() or rr[0].isnumeric() or rr[0]== '"' or rr[0]== '“' or rr[0]== '(' or rr[0]== '[' or rr[0]== '{'):
                    res3.append(rr)
                else:
                    print("Errored: ", rr)
                    for st in range(len(rr)):
                        if rr[st].isalpha() or rr[st].isnumeric() or rr[st]== '"' or rr[st]== '“' or rr[st]== '(' or rr[st]== '[' or rr[st]== '{':
                            rr = rr[st:]
                            break
                    for st in range(len(rr)-1, -1, -1):
                        if rr[st].isalpha() or rr[st].isnumeric() or rr[st]== '"' or rr[st]== '”' or rr[st]== ')' or rr[st]== ']' or rr[st]== '}':
                            rr = rr[:(st+1)]
                            break

                    print("Modified: ", rr)
                    rr_list_new = rr.split()
                    if len(rr_list_new)>=2:
                        res3.append(rr)
    if len(res3)==0:
        res3 = res2

    keywords_lists = ", ".join(res3)

    for all_the in res3:
        all_the_keywords.append(all_the)

    return keywords_lists

N = len(base_df)
all_keywords = []
for i in range(N):
    if i%100==0:
        print("progress: ", i, "/", N)
    kk = get_keywords(base_df['abstract'].values[i])
    all_keywords.append(kk)

base_df.insert(0, 'index', [i for i in range(N)])

base_df.insert(6, 'keywords', all_keywords)

all_the_keywords = list(set(all_the_keywords))
# print(all_the_keywords)

# OpenAI GPT-3 translation to chinese
# import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")
# abst = ",".join(all_the_keywords)
# print(abst)
# text = "Translate this into 1. Chinese:\n\n"+ abst + "\n\n1."
# response = openai.Completion.create(
#   engine="text-davinci-002",
#   prompt=text,
#   temperature=0.3,
#   max_tokens=100,
#   top_p=1,
#   frequency_penalty=0,
#   presence_penalty=0
# )
# result = response.choices[0].text
# print(result)

# OpenAI GPT-3 Keywords
# import openai
# openai_keywords = []

# for i in range(N):
#     if i%10==0:
#         print("progress: ", i, "/", N)

#     abst = base_df['abstract'].values[i]
#     text = "Extract keywords from this text:\n\n"+ abst

#     response = openai.Completion.create(
#       engine="text-davinci-002",
#       prompt=text,
#       temperature=0.3,
#       max_tokens=2500,
#       top_p=1.0,
#       frequency_penalty=0.8,
#       presence_penalty=0.0
#     )

#     result = response.choices[0].text


#     result_list = result.split("\n")
#     result_list = [ress[1:-1] for ress in result_list]
#     # if re.match("^[A-Za-z0-9_-]*$", my_little_string):

#     resp = []

#     for re in result_list:
#       re_list = re.split()
#       if len(re_list)>=2:
#         resp.append(re)
#     temp_res = ", ".join(resp)
#     openai_keywords.append(temp_res)


# base_df.insert(7, 'keywordsOpenAI', openai_keywords)


#Because of a wired column 'Unnamed: 0' appears: Index(['index', 'Unnamed: 0', 'conference', 'url', 'title', 'authors',
      #  'abstract', 'keywords', 'citations'], dtype='object')
# base_df.drop(base_df.columns[1], axis=1, inplace=True)
# base_df.drop(columns='Unnamed: 0', inplace=True)
# Select the ones you want
df= base_df[['index', 'conference', 'url', 'title', 'authors', 'abstract', 'keywords', 'citations']]  #, 'keywordsOpenAI'
print(df.columns)

df.to_csv(f"./new2", index=False, header=False, sep=';')
df.to_csv(f"./new2.csv", index=False, header=True)

with open('keywords_list_all_new2.txt', 'w') as f:
    for item in all_the_keywords:
        f.write("%s\n" % item)
