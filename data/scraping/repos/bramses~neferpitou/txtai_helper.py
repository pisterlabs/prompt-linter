from txtai.embeddings import Embeddings
import openai_helper
import json

class EmbeddingsWrapper:
    def __init__(self, transform=openai_helper.transform, content=True, objects=True, DEBUG=False):
        self.embeddings =  Embeddings({ "method": "external", "transform": transform, "content": content, "objects": objects })
        self.debug = DEBUG
        self.limit = 1000

    def get_embeddings(self):
        if self.debug:
            print(f"--TXTAI-- getting embeddings")
        return self.embeddings

    def set_transform(self, transform):
        if self.debug:
            print(f"--TXTAI-- setting transform to {transform}")
        self.embeddings.config['transform'] = transform

    def create_index(self, documents, tags=None):
        if self.debug:
            print(f"--TXTAI-- creating index with {len(documents)} documents")
        return self.embeddings.index([(val['chunk_uuid'] if val.__contains__('chunk_uuid') else uid, { "text": val['text'], "filename": val['filename'] }, tags) for uid, val in enumerate(documents)])
        
    def search(self, query, n=1, transform=openai_helper.transform_query):
        if self.debug:
            print(f"--TXTAI-- searching for {query} with {n} results")

        if transform:
            self.set_transform(transform)

        return self.embeddings.search(query, n)

    def save_index(self, path):
        if self.debug:
            print(f"--TXTAI-- saving index to {path}")
        self.embeddings.save(path)

    def info(self):
        if self.debug:
            print(f"--TXTAI-- getting info")
        return self.embeddings.info()
    
    def load_index(self, path):
        if self.debug:
            print(f"--TXTAI-- loading index from {path}")
        if not self.exists(path):
            return False
        if self.debug:
            print(f"--TXTAI-- index exists")
        self.embeddings.load(path)
    
        return True

    def upsert_index(self, documents, tags=None, transform=openai_helper.transform):
        if self.debug:
            print(f"--TXTAI-- upserting index with document ids {[doc['chunk_uuid'] for doc in documents]}")

        if transform:
            self.set_transform(transform)

        self.embeddings.upsert([(val['chunk_uuid'], { "text": val['text'], "filename": val['filename'] }, tags) for uid, val in enumerate(documents)])

    def delete_ids(self, ids):
        if self.debug:
            print(f"--TXTAI-- deleting ids {ids}")
        return self.embeddings.delete(ids)

    def list_filenames(self):
        # this method (SELECT DISTINCT) is not implemented in txtai -- so it may break
        if self.debug:
            print(f"--TXTAI-- listing filenames")
        
        count = len(self.embeddings.search(f"SELECT count(*) FROM txtai"))
        files = set()
        if count > self.limit:
            # paginate with offset
            offset = 0
            while True:
                search_results = self.embeddings.search(f"SELECT filename FROM txtai limit {self.limit} offset {offset}")
                for result in search_results:
                    files.add(result['filename'])
                offset += self.limit
                if len(search_results) < self.limit:
                    break
        else:
            search_results = self.embeddings.search(f"SELECT filename FROM txtai limit {self.limit}")
            for result in search_results:
                files.add(result['filename'])

        return files

    def find_ids_by_filename(self, filename):
        if self.debug:
            print(f"--TXTAI-- finding ids by filename {filename}")

        search_results = self.embeddings.search(f"select * from txtai where filename in ('{filename}') limit {self.limit}")
        
        if self.debug:
            print(f"--TXTAI-- ids: {[result['id'] for result in search_results]}")

        return search_results

    def exists(self, path):
        if self.debug:
            print(f"--TXTAI-- checking if index exists")
        return self.embeddings.exists(path)

    def update_documents_text(self, documents, new_text):
        if len(documents) != len(new_text):
            raise Exception("Number of documents and new text must be the same")
        updated_documents = []
        if self.debug:
            print(f"--TXTAI-- updating document text")
        
        idx = 0
        for document in documents:
            if self.debug:
                print(f"--TXTAI-- updating document {document['indexid']}")
            data = json.loads(document['data'])
            updated_documents.append({ "indexid": document['indexid'], "text": new_text[idx], "filename": data['filename'] })
            idx += 1

        return updated_documents
