import cohere  #Install with: pip install cohere
import hnswlib #Install with: pip install hnswlib
import pandas as pd

def sort_array_by_frequency(arr):
    # Step 1: Create a dictionary to store item counts
    item_counts = {}

    # Step 2: Count the occurrences of each item
    for item in arr:
        if item in item_counts:
            item_counts[item] += 1
        else:
            item_counts[item] = 1

    # Step 3: Sort the array based on counts
    sorted_arr = sorted(arr, key=lambda x: item_counts[x], reverse=True)

    return sorted_arr

class embed:
    def __init__(self, key, dataset):
        self.co = cohere.Client(key) 

        self.posts = pd.read_csv(dataset)

        self.posts = self.posts.drop_duplicates(subset="text")

        # Reset the index of the DataFrame
        self.posts.reset_index(drop=True, inplace=True)

        docs = []

        for i in range(0, len(self.posts)):
            docs.append(self.posts["text"][i])

        #Get your document embeddings
        doc_embs = self.co.embed(texts=docs, model='embed-multilingual-v2.0').embeddings

        self.posts["embeddings"] = doc_embs

        #Create a search index
        self.search_index = hnswlib.Index(space='ip', dim=768)
        self.search_index.init_index(max_elements=len(doc_embs), ef_construction=512, M=64)
        self.search_index.add_items(doc_embs, list(range(len(doc_embs))))


    def recomend_music(self, input):
        query_emb = self.co.embed(texts=[input], model='embed-multilingual-v2.0').embeddings
        indexs = self.search_index.knn_query(query_emb, k=10)[0][0]

        musicas = []

        for index in indexs:
            musicas.append(self.posts["musicMeta/musicAuthor"][index] + ":" + " " + str(self.posts["musicMeta/musicName"][index]))
        
        arr = set(sort_array_by_frequency(musicas))

        if len(arr) > 5:
            arr = list(arr)
            return arr[0:5]
        
        return arr