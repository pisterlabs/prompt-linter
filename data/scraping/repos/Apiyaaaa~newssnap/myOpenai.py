import openai
import re
import tiktoken
import asyncio
import numpy as np
from sklearn.cluster import KMeans
import time
from sklearn.metrics import silhouette_score


class MyOpenAI():
    # set timeout
    def __init__(self, text):
        self.text = text
        self.summarymodel = 'gpt-3.5-turbo'
        self.embeddingmodel = 'text-embedding-ada-002'
        self.summary = None
        self.vectors = None
        self.max_reduce_count = 3
        self.token_useage = {
            "token_type": {
                "summary": 0,
                "embedding": 0
            },
            "token_price": {
                "summary": 0.001,
                "embedding": 0.0001
            },
        }

    def total_cost(self):
        summary = self.token_useage['token_type']['summary'] / \
            1000 * self.token_useage['token_price']['summary']
        embedding = self.token_useage['token_type']['embedding'] / \
            1000 * self.token_useage['token_price']['embedding']
        return summary + embedding

    async def reduce_text(self, splited_text):
        # Cluster
        vectors = [embedding for embedding in await asyncio.gather(*[self.aembedding(i) for i in splited_text])]
        reduced_list = []
        matrix = np.array(vectors)
        n_cluster = self.best_n_clusters(matrix)
        kmeans = KMeans(n_clusters=n_cluster, random_state=42,
                        n_init='auto').fit(matrix)
        for i in range(n_cluster):
            distance = np.linalg.norm(
                matrix - kmeans.cluster_centers_[i], axis=1)
            index = np.argmin(distance)
            reduced_list.append(index)
        reduced_list = sorted(reduced_list)
        return [splited_text[i] for i in reduced_list]

    def split_text(self, text, max_token=500):
        splited_text = re.findall('.*?[。？！；\n]', text)
        curr_token_num = 0
        cur_str = ''
        merger_list = []
        for i in splited_text:
            if curr_token_num + self.num_tokens_from_string(i, self.summarymodel) >= max_token:
                merger_list.append(cur_str)
                cur_str = ''
                curr_token_num = 0
            cur_str += i
            curr_token_num += self.num_tokens_from_string(i, self.summarymodel)
        if cur_str != '':
            merger_list.append(cur_str)
        return merger_list

    def num_tokens_from_string(self, string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding("cl100k_base")
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    async def get_summary(self, text):
        n = 0
        while n < 3:
            try:
                response = await openai.ChatCompletion.acreate(
                    model=self.summarymodel,
                    messages=[
                        {"role": "system", "content": "你是一位新闻编辑，你需要用中文生成以下新闻的摘要："},
                        {"role": "user", "content": text}
                    ],
                    temperature=0,
                )
                break
            except Exception as e:
                print(e, '重试', n)
                n += 1
                time.sleep(5)

        # Culmulate token useage
        self.token_useage['token_type']['summary'] += response.usage['total_tokens']
        return response.choices[0]['message']['content']

    async def reduce_summary(self, text=None):
        self.max_reduce_count -= 1
        if self.max_reduce_count <= 0:
            return text
        # 如果text为None 则使用self.text （初始状态）
        if text is None:
            raw_text = self.text
        else:
            raw_text = text
        # 如果text的token数量小于3000 则直接生成摘要
        if self.num_tokens_from_string(raw_text, self.summarymodel) >= 3000:
            splited_text = self.split_text(raw_text, max_token=300)
        else:
            return await self.get_summary(raw_text)
        # 如果text的token数量大于3000 则先进行聚类 取出每个簇的中心点
        # 再对每个中心点生成摘要
        splited_text = await self.reduce_text(splited_text)
        results = await asyncio.gather(*[self.get_summary(i) for i in splited_text])
        finished_text = self.split_text(''.join(results), max_token=3000)
        if len(finished_text) > 1:
            return await self.reduce_summary(''.join(finished_text))
        else:
            self.summary = ''.join(finished_text)
            return self.summary

    async def aembedding(self, text):
        n = 0
        while n < 3:
            try:
                res = await openai.Embedding.acreate(
                    input=text,
                    model=self.embeddingmodel,
                )
                self.token_useage['token_type']['embedding'] += res.usage['total_tokens']
                return res['data'][0]['embedding']
            except Exception as e:
                n += 1
                print(e, '重试', n)
                time.sleep(5)

    async def summarizeEmbedding(self):
        try:
            res = await self.reduce_summary(self.text)
            if self.vectors is None:
                vectors = await self.aembedding(self.text)
                self.vectors = vectors
            embedding = self.vectors
            return {
                "summary": res,
                "embedding": embedding
            }
        except Exception as e:
            print(e)

    def get_Object(self):
        asyncio.run(self.summarizeEmbedding())
        print("Total cost: ", self.total_cost())
        return {
            "summary": self.summary,
            "embedding": self.vectors
        }

    async def aget_Object(self):
        obj = await self.summarizeEmbedding()
        return obj

    def best_n_clusters(self, matrix):
        # 最多7个簇 降低成本
        if len(matrix) < 7:
            max_cluster = len(matrix)
        else:
            max_cluster = 7
        silhouette_scores = []
        cluster_range = range(2, max_cluster)
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters,
                            init="k-means++", random_state=42, n_init='auto')
            kmeans.fit(matrix)
            labels = kmeans.labels_
            score = silhouette_score(matrix, labels)
            silhouette_scores.append(score)
        best_n_clusters = cluster_range[silhouette_scores.index(
            max(silhouette_scores))]
        return best_n_clusters
