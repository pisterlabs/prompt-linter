
import numpy as np
import pandas as pd
import openai, os
from sklearn.cluster import KMeans
from IPython.display import display

embedding_df = pd.read_parquet("20_newsgroup_with_embedding.parquet")

# 通过 NumPy 的 stack 函数，把所有的 Embedding 放到一个矩阵里面，设置一下要聚合出来的类的数量，然后运行一下 K-Means 算法的 fit 函数，就好了
matrix = np.vstack(embedding_df.embedding.values)
num_of_clusters = 20

kmeans = KMeans(n_clusters=num_of_clusters, init="k-means++", n_init=10, random_state=42)
kmeans.fit(matrix)
labels = kmeans.labels_
embedding_df["cluster"] = labels

# 统计一下聚类之后的每个类有多少条各个 newsgroups 分组的数据

# 通过 groupby 可以把之前的 DataFrame 按照 cluster 进行聚合，统计每个 cluster 里面数据的条数。
new_df = embedding_df.groupby('cluster')['cluster'].count().reset_index(name='count')

# 要统计某一个 cluster 里面排名第一的分组名称和数量的时候，我们可以通过 groupby，把数据按照 cluster + title 的方式聚合。
title_count = embedding_df.groupby(['cluster', 'title']).size().reset_index(name='title_count')
# 通过 cluster 聚合后，使用 x.nlargest 函数拿到里面数量排名第一的分组的名字和数量。
first_titles = title_count.groupby('cluster').apply(lambda x: x.nlargest(1, columns=['title_count']))
first_titles = first_titles.reset_index(drop=True)
new_df = pd.merge(new_df, first_titles[['cluster', 'title', 'title_count']], on='cluster', how='left')
new_df = new_df.rename(columns={'title': 'rank1', 'title_count': 'rank1_count'})

# 把数据里排名第一的去掉之后，又统计了一下排名第二的分组
second_titles = title_count[~title_count['title'].isin(first_titles['title'])]
second_titles = second_titles.groupby('cluster').apply(lambda x: x.nlargest(1, columns=['title_count']))
second_titles = second_titles.reset_index(drop=True)
new_df = pd.merge(new_df, second_titles[['cluster', 'title', 'title_count']], on='cluster', how='left')
new_df = new_df.rename(columns={'title': 'rank2', 'title_count': 'rank2_count'})
new_df['first_percentage'] = (new_df['rank1_count'] / new_df['count']).map(lambda x: '{:.2%}'.format(x))
# 将缺失值替换为 0
new_df.fillna(0, inplace=True)
# 输出结果
display(new_df)


# 给分类取名
items_per_cluster = 10
COMPLETIONS_MODEL = "text-davinci-003"
openai.api_key = os.environ.get("OPENAI_API_KEY")

for i in range(num_of_clusters):
    cluster_name = new_df[new_df.cluster == i].iloc[0].rank1
    print(f"Cluster {i}, Rank 1: {cluster_name}, Theme:", end=" ")

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