#!/usr/bin/env python3
# -*- coding utf-8 -*-

import pandas as pd
import numpy as np
import openai, tiktoken, backoff, yaml
from openai.embeddings_utils import get_embeddings
from sklearn.cluster import KMeans
from IPython.display import display

'''
先使用utils.py的twenty_newsgroup_to_csv的方法下载数据文件
先用chatgpt获取向量
然后通过kmean进行聚类
'''

def get_api_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        openai.api_key = yaml_data["openai"]["api_key"]


# 向量化参数
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002
batch_size = 2000
max_tokens = 8000  # the maximum for text-embedding-ada-002 is 8191
COMPLETIONS_MODEL = "text-davinci-003"


# 读取数据，并对数据进行预处理
def read_data_to_df():
    df = pd.read_csv('20_newsgroup.csv')
    print("Number of rows before null filtering:", len(df))
    df = df[df['text'].isnull() == False]

    # 去掉超长的数据
    encoding = tiktoken.get_encoding(embedding_encoding)
    df["n_tokens"] = df.text.apply(lambda x: len(encoding.encode(x)))
    print("Number of rows before token number filtering:", len(df))
    df = df[df.n_tokens <= max_tokens]
    print("Number of rows data used:", len(df))
    return df


# 批量接口获取文件向量
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_embeddings_with_backoff(prompts, engine):
    embeddings = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        embeddings += get_embeddings(list_of_text=batch, engine=engine)
    return embeddings


# 试用批处理接口，获取文件向量，并保持为parquet格式
def get_embddings_to_parquet(df):
    prompts = df.text.tolist()
    prompt_batches = [prompts[i:i+batch_size] for i in range(0, len(prompts), batch_size)]

    embeddings = []
    for batch in prompt_batches:
        batch_embeddings = get_embeddings_with_backoff(prompts=batch, engine=embedding_model)
        embeddings += batch_embeddings

    df["embedding"] = embeddings
    df.to_parquet("data/20_newsgroup_with_embedding.parquet", index=False)


# 聚类cluster数量
num_of_clusters = 20

# 从parquet获取文件向量，并用k-means聚类
def kmeans_cluster():
    embedding_df = pd.read_parquet("data/20_newsgroup_with_embedding.parquet")

    matrix = np.vstack(embedding_df.embedding.values)
    
    kmeans = KMeans(n_clusters=num_of_clusters, init="k-means++", n_init=10, random_state=42)
    kmeans.fit(matrix)
    labels = kmeans.labels_
    embedding_df["cluster"] = labels

    return embedding_df


#评估聚类效果
def cluster_eval(embedding_df):
    # 统计每个cluster的数量
    new_df = embedding_df.groupby('cluster')['cluster'].count().reset_index(name='count')

    # 统计每个cluster里最多的分类的数量
    title_count = embedding_df.groupby(['cluster', 'title']).size().reset_index(name='title_count')
    first_titles = title_count.groupby('cluster').apply(lambda x: x.nlargest(1, columns=['title_count']))
    first_titles = first_titles.reset_index(drop=True)
    new_df = pd.merge(new_df, first_titles[['cluster', 'title', 'title_count']], on='cluster', how='left')
    new_df = new_df.rename(columns={'title': 'rank1', 'title_count': 'rank1_count'})

    # 统计每个cluster里第二多的分类的数量
    second_titles = title_count[~title_count['title'].isin(first_titles['title'])]
    second_titles = second_titles.groupby('cluster').apply(lambda x: x.nlargest(1, columns=['title_count']))
    second_titles = second_titles.reset_index(drop=True)
    new_df = pd.merge(new_df, second_titles[['cluster', 'title', 'title_count']], on='cluster', how='left')
    new_df = new_df.rename(columns={'title': 'rank2', 'title_count': 'rank2_count'})

    # 计算每个分类中排名第一分类的百分比
    new_df['first_percentage'] = (new_df['rank1_count'] / new_df['count']).map(lambda x: '{:.2%}'.format(x))
    # 将缺失值替换为 0
    new_df.fillna(0, inplace=True)
    # 输出结果
    display(new_df)

    return new_df


#每个cluster提供10条内容，提供给chatgpt，让其给出中文类名
def trans_cluster_name(embedding_df, new_df):
    items_per_cluster = 10

    for i in range(num_of_clusters):
        cluster_name = new_df[new_df.cluster == i].iloc[0].rank1
        print(f"Cluster {i}, Rank 1: {cluster_name}, Theme:", end=" ")

        # 采样
        content = "\n".join(
            embedding_df[embedding_df.cluster == i].text.sample(items_per_cluster, random_state=42).values
        )

        response = openai.Completion.create(
            model=COMPLETIONS_MODEL,
            prompt=f'''我们想要给下面的内容，分组成有意义的类别，以便我们可以对其进行总结。请根据下面这些内容的共同点，总结一个50个字以内的新闻组的名称。比如 “PC硬件”\n\n内容:\n"""\n{content}\n"""新闻组名称：''',
            temperature=0,
            max_tokens=100,
            top_p=1,
        )
        print(response["choices"][0]["text"].replace("\n", ""))


#每个cluster提供1条内容，提供给chatgpt，让其给出中文翻译，从而验证类名是否正确
def eval_trans_cluster_name(embedding_df, new_df):
    items_per_cluster = 1

    for i in range(num_of_clusters):
        cluster_name = new_df[new_df.cluster == i].iloc[0].rank1
        print(f"Cluster {i}, Rank 1: {cluster_name}, 抽样翻译:", end=" ")

        # 采样
        content = "\n".join(
            embedding_df[(embedding_df.cluster == i) & (embedding_df.n_tokens > 100)].text.sample(items_per_cluster, random_state=42).values
        )

        response = openai.Completion.create(
            model=COMPLETIONS_MODEL,
            prompt=f'''请把下面的内容翻译成中文\n\n内容:\n"""\n{content}\n"""翻译：''',
            temperature=0,
            max_tokens=2000,
            top_p=1,
        )
        print(response["choices"][0]["text"].replace("\n", ""))


if __name__ == '__main__':
    get_api_key()
    df = read_data_to_df()
    get_embddings_to_parquet(df)
    embedding_df = kmeans_cluster()
    new_df = cluster_eval(embedding_df)
    trans_cluster_name(embedding_df, new_df)
    eval_trans_cluster_name(embedding_df, new_df)
