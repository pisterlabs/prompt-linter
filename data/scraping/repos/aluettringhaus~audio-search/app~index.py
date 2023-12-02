import json

import cohere
import faiss
import numpy as np
import redis

import config

settings = config.Settings()

r = redis.Redis(**settings.redis)
co = cohere.Client(settings.cohere["api-key"])

faiss_idx = {}


class Index:
    def __init__(self, index_name):
        idx = json.loads(r.get(index_name))
        self.name = idx["name"]
        self.bucket_field = idx["bucket_field"]
        self.txt_field = idx["txt_field"]

    @staticmethod
    def create_index(idx):
        r.set(idx["name"], json.dumps(idx))

    def get_index(self, index_name):
        return r.get(self.get_redis_address([index_name]))

    def get_item(self, item_id):
        return r.get(self.get_redis_address([item_id]))

    def set_item(self, item):
        bucket = self.bucket_field and item[self.bucket_field]

        # This creates the index at faiss_idx if it is not initialized yet.
        self.check_or_create_faiss_index(bucket)

        # Remove the item before indexing again in case the set operation is an update
        if self.get_item(item["id"]):
            self.remove_item(item["id"])

        embedding = self._get_embedding(item[self.txt_field])
        embedding = np.array(embedding).astype("float32")
        # to use cosine similarity as a search metric the vector needs to be normalized
        # before adding it to the InnerProduct index.
        faiss.normalize_L2(embedding)

        # faiss can only hold integer ids
        # At each index redis holds a list of empty strings. One string for each item that was added to the faiss index.
        # rpush adds an item to the tail of the list and returns its new length. The length will be used as an unique id for faiss.
        faiss_id = r.rpush(self.get_redis_address(["faiss"]), "")

        if bucket:
            faiss_idx[self.name][bucket].add_with_ids(embedding, np.array([faiss_id]))
        else:
            faiss_idx[self.name].add_with_ids(embedding, np.array([faiss_id]))

        # a pipeline reduces roundtrips and guarantees atomicity
        with r.pipeline() as pipe:
            item_id = self.get_redis_address([item["id"]])
            item_idx = self.get_redis_address(["faiss", item["id"]])
            faiss_lookup = self.get_redis_address(["faiss", faiss_id])
            pipe.set(faiss_lookup, item_id)
            pipe.set(item_idx, faiss_lookup)
            pipe.set(item_id, json.dumps(item))
            pipe.execute()
        return embedding.tolist()

    def query_item(self, query):
        embedding = self._get_embedding(query["query"])
        embedding = np.array(embedding).astype("float32")

        faiss.normalize_L2(embedding)
        k = query["limit"]
        bucket = query.get("bucket", None)

        # if no vectors are indexed an empty array should be returned
        if not faiss_idx.get(self.name, None):
            return []

        if self.bucket_field:
            # see above
            if not faiss_idx[self.name].get(bucket, None):
                return []
            # depending on the definition of a bucket field for an index the index
            # might be located at faiss_idx[idx_name] or faiss_id[<idx_name>][<bucket_field>]
            idx = faiss_idx[self.name][bucket]
        else:
            idx = faiss_idx[self.name]

        distances, indices = idx.search(embedding, k)
        distances = distances[0].tolist()

        # faiss will return a -1 index if there are less results than k
        # these indices should be filtered out
        zero = np.zeros((k,))
        filtered_indices = indices[indices > zero]
        faiss_ids = []
        scores = []
        for i, index in enumerate(filtered_indices.tolist()):
            faiss_ids.append(self.get_redis_address(["faiss", index]))
            scores.append(distances[i])

        item_ids = r.mget(faiss_ids)
        items = r.mget(item_ids)

        out = []
        for i, item in enumerate(items):
            it = {}
            it["item"] = json.loads(item)
            it["score"] = scores[i]
            out.append(it)
        return out

    def _get_embedding(self, text):
        embeds = co.embed(
            texts=[text],
            model=settings.cohere["model"],
            truncate=settings.cohere["truncate"],
        ).embeddings

        return embeds

    # as per the redis convention namespaces are created using the index name as a namespace
    # attributes are separated by a colon (e.g. bookindex:uuid or bookindex:faiss:faissid<int> )
    def get_redis_address(self, path):
        out = str(self.name)
        for p in path:
            out = out + ":" + str(p)
        return out

    def remove_item(self, item_id):
        redis_id = self.get_redis_address([item_id])
        item = json.loads(r.get(redis_id))
        redis_faiss_id = self.get_redis_address(["faiss", item_id])
        faiss_item = r.get(redis_faiss_id)
        faiss_id = int(faiss_item.split(":")[-1])

        with r.pipeline() as pipe:
            pipe.delete(redis_id)
            pipe.delete(redis_faiss_id)
            pipe.delete(faiss_item)
        if self.bucket_field:
            try:
                faiss_idx[self.name][item[self.bucket_field]].remove_ids(
                    np.array([int(faiss_id)])
                )
            except:
                print(f"failed to remove {faiss_id}")
        else:
            try:
                faiss_idx[item[self.bucket_field]].remove_ids(np.array([int(faiss_id)]))
            except:
                print(f"failed to remove {faiss_id}")

    def check_or_create_faiss_index(self, bucket=None):
        ns = faiss_idx.get(self.name, None)
        if not ns:
            faiss_idx[self.name] = {}

        if bucket:
            idx = faiss_idx[self.name].get(bucket, None)
            if not idx:
                idx = faiss.IndexFlatIP(4096)
                idx = faiss.IndexIDMap2(idx)
                faiss_idx[self.name][bucket] = idx
        elif not ns:
            idx = faiss.IndexFlatIP(4096)
            idx = faiss.IndexIDMap2(idx)
            faiss_idx[self.name] = idx

        return True
