import numpy
import os
import re
import json
from typing import Any
import requests
from io import StringIO
import pandas as pd
from s3 import S3
from openai_api import OpenAIAPI
import faiss
import sqlite3
import hashlib
import math
from hf_embed import HFEmbed
import random
from web3storage import Web3StorageAPI
import hnswlib
import pickle

embedding_models = [
    "text-embedding-ada-002",
    "gte-large",
    "gte-base",
    "bge-base-en-v1.5",
    "bge-large-en-v1.5",
    "instructor",
    "instructor-xl"
]

summarization_models = [
    'gpt-3.5-turbo',
    'gpt-4','gpt-3.5-turbo-16k',
    'gpt-3.5-turbo-instruct'
]

vector_index_ids = {
    "text-embedding-ada-002" : 1536,
    "gte-large": 1024,
    "gte-base": 768,
    "bge-base-en-v1.5": 768,
    "bge-large-en-v1.5": 1024,
    "instructor": 768,
    "instructor-xl": 768
}

class KNN:
    cwd = os.getcwd()
    dir = os.path.dirname(__file__)
    modelsDir = os.path.join(dir, "models")

    def __init__(self, resources, meta):
        self.cwd = os.getcwd()
        self.dir = os.path.dirname(__file__)
        self.parentDir = os.path.dirname(self.dir)
        self.modelsDir = os.path.join(self.parentDir, "models")
        self.model = None
        self.bucket = None
        self.bucket_files = None
        self.search_query = None
        self.search_results = None
        self.resources = resources
        self.method = None
        self.k = None
        self.indexes = {}

        if meta is not None:
            if "config" in meta:
                if meta['config'] is not None:
                    self.config = meta['config']
                    pass
                pass
            if "openai_api_key" in meta:
                if meta['openai_api_key'] is not None:
                    self.openai_api_key = meta['openai_api_key']
                    pass
                pass
            pass
            if "web3_api_key" in meta:
                if meta['web3_api_key'] is not None:
                    self.web3_api_key = meta['web3_api_key']
                    pass
                pass
        
        self.s3 = S3(resources, meta)
        self.web3 = Web3StorageAPI(resources, meta)
        self.openai = OpenAIAPI(resources, meta)
        self.datastore = {}

    def __call__(self, method, **kwargs):
        if "bucket" in kwargs:
            self.model = kwargs['bucket']
        if "bucket" in self:
            self.model = self.bucket
        if method == 'query':
            return self.query(**kwargs)
        if method == 'ingest':
            return self.ingest(**kwargs)
        if method == 'create':
            return self.create(**kwargs)
        if method == 'append':
            return self.append(**kwargs)
        if method == 'pop':
            return self.pop(**kwargs)
        pass

    def save_missing_embeddings(self, bucket, dir, **kwargs):
        database = self.load_database(bucket, dir, **kwargs)
        select_column = "embedding"
        query = "SELECT * FROM "+dir+"_doc_store where "+select_column+" is NULL"
        database.execute(query)
        rows = database.fetchall()
        if len(rows) > 0:
            for row in rows:
                id = row[0]
                text = row[7]
                embedding = self.generate_embedding([text], None, **kwargs)
                update = "UPDATE "+dir+"_doc_store SET "+select_column+" = '"+embedding+"' WHERE id = '"+id+"'"
                database.execute(update)
                pass
        else:
            return None
        
    def pdf_to_text(self, file, **kwargs):

        return None

    def retrieve_doc_chunk(self, src, doc_store, doc_index, node_id):
        
        if src == "web3":
            doc_store_uri = "https://"+ doc_store + ".ipfs.dweb.link"
            this_doc_store = requests.get(doc_store)
            doc_index_uri = "https://"+ doc_index + ".ipfs.dweb.link"
            this_doc_index = requests.get(doc_index)

            pass

        return None

    def retrieve_index_metadata(self, src, index, **kwargs):
        if src == "web3":
            ls_files = self.web3.list(**kwargs)

            pass
        return None

    def load_embeddings_column(self, bucket, dir, **kwargs):
        database = self.load_database(bucket, dir, **kwargs)
        select_column = "embedding"
        query = "SELECT "+select_column+" FROM "+dir+"_doc_store"
        database.execute(query)
        rows = database.fetchall()
        if len(rows) > 0:
            return rows
        else:
            return None

    def load_database(self, bucket, dir, **kwargs):
        if bucket is not None:
            self.bucket = bucket
            self.model = bucket
        
        if os.path.isdir(self.modelsDir):
            modelDir = os.path.join(self.modelsDir, self.model+"@knn")
            if os.path.isdir(modelDir):
                    datafile = os.path.join(modelDir, bucket+".sqlite")
                    if os.path.isfile(datafile):
                        conn = sqlite3.connect(datafile)
                        self.cursor = conn.cursor()
                        return self.cursor
            return None

    def load_text(self, source, bucket, dir, **kwargs):
        sources = ["s3", "sqlite", "json", "raw"]
        chunks = {}
        if bucket is not None:
            self.bucket = bucket
            self.model = bucket

        if os.path.isdir(self.modelsDir):
            modelDir = os.path.join(self.modelsDir, self.model)

        if source not in sources:
            raise Exception('bad source: %s' % source)
        else:
            self.source = source
        
        if source == "s3":
            self.s3_dir = self.s3.s3_read_dir(self.config.dir, self.config.bucket, self.config)
            files = []
            ## make an interable ##
            for file in self.s3_dir:
                this_file = { "key": file.key, "size": file.size, "s3url": "s3://"+bucket+"/"+dir+"/"+file.key}
                files.append(this_file["s3url"])
            
            iterable = iter(files)

            chunks = (self.s3.s3_read_file(self.config.dir, self.config.bucket, self.config, file) for file in iterable)

            pass
        elif source == "sqlite":
            if os.path.isdir(modelDir):
                datafile = os.path.join(modelDir, bucket+".sqlite")
                if os.path.isfile(datafile):
                    conn = sqlite3.connect(datafile)
                    cursor = conn.cursor()
                    cursor.execute("SELECT * FROM "+bucket)
                    rows = cursor.fetchall()
                    columns = [description[0] for description in cursor.description]
                    for row in rows:
                        if dir in columns:
                            if row[dir] is not None:
                                chunks.append(row[dir])
                    conn.close()
            pass
        
        elif source == "json":
            if os.path.isdir(modelDir):
                datafile = os.path.join(modelDir, bucket+".json")
                if os.path.isfile(datafile):
                    with open(datafile, 'r') as f:
                        data = f.read()
                        json_data = json.loads(data)
                        if type(json_data) is list:
                            chunks = json_data
                        elif type(json_data) is dict:
                            chunks = json_data[dir] 

        elif source == "raw":
            modelDir = os.path.join(self.modelsDir, self.model+"@knn")
            if os.path.isdir(modelDir):
                dataDir = os.path.join(modelDir, dir)
                if os.path.isdir(dataDir):
                    dataDirFiles = os.listdir(dataDir)
                    for dataDirFile in dataDirFiles:
                        dataDirFilePath = os.path.join(dataDir, dataDirFile)
                        if os.path.isfile(dataDirFilePath):
                            with open(dataDirFilePath, 'r') as f:
                                data = f.read()
                                chunks[dataDirFilePath] = data
            pass

        return chunks
    
    def save_database(self, dest, bucket, dir, documentdb, **kwargs):
        tables = documentdb.keys()
        if bucket is not None:
            self.bucket = bucket
            self.model = bucket
        if dest == "sqlite":
            if os.path.isdir(self.modelsDir):
                modelDir = os.path.join(self.modelsDir, self.model+"@knn")
                if os.path.isdir(modelDir):
                    datafile = os.path.join(modelDir, dir+".sqlite")
                    if not os.path.isfile(datafile):
                        conn = sqlite3.connect(datafile)
                        cursor = conn.cursor()
                        for table in tables:
                            columns = documentdb[table].keys()
                            execute = "CREATE TABLE "+table+" ("
                            for column in columns:
                                execute += column+" TEXT,"
                            execute = execute[:-1]
                            execute += ")"
                            cursor.execute(execute)
                            conn.commit()
                    if os.path.isfile(datafile):
                        conn = sqlite3.connect(datafile)
                        cursor = conn.cursor()
                        for table in tables:
                            datakey = list(documentdb[table].keys())[0]
                            items = list(documentdb[table][datakey].keys())
                            if(table == "vector_store"):
                                columns = ["embedding_id", "embedding_model_id", "embedding"]
                                ## create table if not exist ##
                                execute = "CREATE TABLE IF NOT EXISTS "+table+" ("
                                for column in columns:
                                    execute += column+" TEXT,"
                                execute = execute[:-1]
                                execute += ")"
                                cursor.execute(execute)
                                conn.commit()
                                vector_model_ids = list(documentdb[table][datakey].keys())
                                for vector_model in vector_model_ids:
                                    embeddings = list(documentdb[table][datakey][vector_model].keys())
                                    for embedding_id in embeddings:
                                        this_embedding = documentdb[table][datakey][vector_model][embedding_id]
                                        this_embedding = json.dumps(this_embedding)
                                        ## insert data ##
                                        execute = "INSERT INTO "+table+" VALUES ("
                                        execute += "'"+embedding_id+"',"
                                        execute += "'"+vector_model+"',"
                                        execute += "'"+this_embedding+"'"
                                        execute += ")"
                                        cursor.execute(execute)
                                        conn.commit()                        
                            else:
                                for item in items:
                                    item_keys = list(documentdb[table][datakey][item].keys())
                                    this_item = documentdb[table][datakey][item]["__data__"]
                                    columns = list(this_item.keys())
                                    values = []
                                    for column in columns:
                                        values.append(this_item[column])
                                            
                                    ## create table if not exist
                                    execute = "CREATE TABLE IF NOT EXISTS "+table+" ("
                                    for column in columns:
                                        execute += column+" TEXT,"
                                    execute = execute[:-1]
                                    execute += ")"
                                    cursor.execute(execute)
                                    
                                    execute = "INSERT INTO "+table+" VALUES ("
                                    for value in values:
                                        if type(value) is dict or type(value) is list:
                                            value = json.dumps(value)
                                        if type(value) is int:
                                            value = str(value)
                                        if "'" in value:
                                            value = value.replace("'", "''")
                                        execute += "'"+value+"',"
                                    execute = execute[:-1]
                                    execute += ")"
                                    cursor.execute(execute)
                                    conn.commit()
                        conn.close()
                    pass
                pass
            pass
        elif dest == "json":
            if os.path.isdir(self.modelsDir):
                modelDir = os.path.join(self.modelsDir, self.model+"@knn")
                if os.path.isdir(modelDir):
                    vector_index = documentdb["vector_index"]
                    vector_store = documentdb["vector_store"]
                    doc_index = documentdb["doc_index"]
                    doc_store = documentdb["doc_store"]
                    ## write these all to files ##
                    if not os.path.isdir(modelDir):
                        Exception("datafolder does not exist")
                    vector_index_file = os.path.join(modelDir, "vector_index.json")
                    vector_store_file = os.path.join(modelDir, "vector_store.json")
                    doc_index_file = os.path.join(modelDir, "doc_index.json")
                    doc_store_file = os.path.join(modelDir, "doc_store.json")
                    with open(vector_index_file, 'w') as f:
                        json.dump(vector_index, f)
                    with open(vector_store_file, 'w') as f:
                        json.dump(vector_store, f)
                    with open(doc_index_file, 'w') as f:
                        json.dump(doc_index, f)
                    with open(doc_store_file, 'w') as f:
                        json.dump(doc_store, f)
                pass
            return True
        elif dest == "s3":
            pass
        elif dest == "postrges":
            pass
        elif dest == "web3":
            if os.path.isdir(self.modelsDir):
                modelDir = os.path.join(self.modelsDir, self.model+"@knn")
                if os.path.isdir(modelDir):
                    vector_index = documentdb["vector_index"]
                    vector_store = documentdb["vector_store"]
                    doc_index = documentdb["doc_index"]
                    doc_store = documentdb["doc_store"]
                    ## write these all to files to web3storage
                    vector_index_cid = self.web3.upload("vector_index.json",  None, json.dumps(vector_index))
                    vector_store_cid = self.web3.upload("vector_store.json",  None, json.dumps(vector_store))
                    doc_index_cid = self.web3.upload("doc_index.json", None, json.dumps(doc_index))
                    doc_store_cid = self.web3.upload("doc_store.json", None, json.dumps(doc_store))
                    metadata_json = {}
                    metadata_json["vector_index.json"] = vector_index_cid
                    metadata_json["vector_store.json"] = vector_store_cid
                    metadata_json["doc_index.json"] = doc_index_cid
                    metadata_json["doc_store.json"] = doc_store_cid
                    metadata_cid = self.web3.upload("metadata.json", None,  json.dumps(metadata_json))
                pass
            return metadata_cid
        return False
    
    def generate_document(self, doc_text, document_id, embeddings, metadata, relationships, ctx_start, ctx_end, **kwargs):
        document = {}
        document["__type__"] = "1"
        document["__data__"] = {}
        document["__data__"]["id"] = document_id
        document["__data__"]["embedding"] = embeddings
        document["__data__"]["metadata"] = metadata
        document["__data__"]["excluded_embed_metadata_keys"] = []
        document["__data__"]["excluded_llm_metadata_keys"] = []
        document["__data__"]["relationships"] = relationships
        document["__data__"]["hash"] = hashlib.sha256(doc_text.encode('utf-8')).hexdigest()
        #document["__data__"]["text"] = doc_text
        document["__data__"]["start_char_idx"] = ctx_start
        document["__data__"]["end_char_idx"] = ctx_end
        document["__data__"]["text_template"] = "{metadata_str}\n\n{content}"
        document["__data__"]["metadata_template"] = "{key}: {value}"
        document["__data__"]["metadata_seperator"] = "\n"
        return document
    
    def convert_to_uuid(self, text, **kwargs):
        if type(text) is not str:
            text = str(text)
        if type(text) is numpy.ndarray:
            text = text.tolist()
            newlist = []
            for item in text:
                item = str(item)
                newlist.append(item)
            text = newlist.join("")

        text = hashlib.sha256(text.encode('utf-8')).hexdigest()
        text = text[:36]
        text = text[:8] + "-" + text[8:12] + "-" + text[12:16] + "-" + text[16:20] + "-" + text[20:]
        return text

    def ingest_8k_split(self, tokens, bucket, dir, parent, parent_hash, parent_text, ctx_start, ctx_end, filename, s3uri, web3uri, embedding_model, summary_model, **kwargs):
        token_length = 8191
        if len(tokens) > 8191:
            Exception("tokens too long")
        results = {}
        subdocument_text = self.openai.detokenize(tokens, None, **kwargs)
        subdocument_summary = self.generate_summary(subdocument_text, summary_model, **kwargs)
        subdocument_id = self.convert_to_uuid(subdocument_text)
        subdocument_embedding = self.generate_embedding(subdocument_text, embedding_model, **kwargs)
        subdocument_embedding_id = self.convert_to_uuid(subdocument_embedding)
        subdocument_embedding_model_id = self.convert_to_uuid(embedding_model)
        subdocument_chunk_ids = []
        subdocument_chunks = []
        subdocument_embedding_ids = []
        subdocument_ctx_end = 512
        subdocument_ctx_start = 0
        subdocuments = []
        parent_dict = {}
        parent_dict["__node__"] = parent
        parent_dict["__type__"] = "1"
        parent_dict["__data__"] = {}
        parent_dict["__data__"]["hash"] = parent_hash
        parent_dict["__data__"]["node_type"] = 1
        parent_dict["__data__"]["node_id"] = parent
        parent_dict["__data__"]["metadata"] = self.extract_metadata(parent_text, bucket, dir, parent, filename, s3uri, web3uri, ctx_start, ctx_end)
        parent_dict["__data__"]["relationships"] = self.extract_relationships(parent, parent_text, None, None, parent_dict, parent_dict["__data__"]["metadata"])

        while subdocument_ctx_end <= 8191:
            subdocument_chunk_tokens = tokens[subdocument_ctx_start:subdocument_ctx_end]
            results = self.ingest_512_split(subdocument_chunk_tokens, bucket, dir,  subdocument_id, ctx_start, subdocument_ctx_start, subdocument_ctx_end, filename, s3uri, web3uri, "bge-large-en-v1.5", None, **kwargs)
            subdocument_chunk_id = results["subdocument_id"]
            subdocument_chunk_ids.append(results["subdocument_id"])
            subdocument_ctx_end = int(math.floor(subdocument_ctx_end + ((results["ctx_end"] - results["ctx_start"]) / 2)))
            subdocument_ctx_start = int(math.floor(subdocument_ctx_start + ((results["ctx_end"] - results["ctx_start"]) / 2)))
            subdocument_metadata = self.extract_metadata(subdocument_text, bucket, dir, subdocument_id, filename,  s3uri, web3uri, ctx_start, ctx_end)
            subdocument_dict = {}
            subdocument_dict["__node__"] = subdocument_chunk_id
            subdocument_dict["__type__"] = "1"
            subdocument_dict["__data__"] = {}
            subdocument_dict["__data__"]["hash"] = hashlib.sha256(subdocument_text.encode('utf-8')).hexdigest()
            subdocument_dict["__data__"]["node_type"] = 1
            subdocument_dict["__data__"]["node_id"] = subdocument_chunk_id
            subdocument_dict["__data__"]["metadata"] = subdocument_metadata
            subdocuments.append(subdocument_dict)

        subdocument_summary_id = self.convert_to_uuid(subdocument_summary)
        subdocument_summary_metadata = self.extract_metadata(subdocument_summary, bucket, dir, subdocument_summary_id, filename, s3uri, web3uri, None, None)
        subdocument_summary_dict = {}
        subdocument_summary_dict["node_id"] = hashlib.sha256(subdocument_text.encode('utf-8')).hexdigest()
        subdocument_summary_dict["node_type"] = 1
        subdocument_summary_dict["metadata"] = subdocument_summary_metadata
        subdocument_summary_dict["text"] = subdocument_summary

        relationships = self.extract_relationships(subdocument_id, subdocument_text, subdocument_summary_dict, subdocuments, subdocument_dict, subdocument_metadata)
        subdocument = self.generate_document(subdocument_text,subdocument_id, subdocument_summary, subdocument_metadata, relationships, ctx_start, ctx_end)              
        self.datastore["vector_index"]["vector_index/data"][subdocument_embedding_model_id]["__data__"]["nodes_dict"][subdocument_embedding_id] = subdocument_id
        self.datastore["vector_store"]["vector_store/data"][subdocument_embedding_model_id][subdocument_embedding_id] = subdocument_embedding["data"]                    
        self.datastore["doc_store"]["doc_store/data"][subdocument_id] = subdocument
        return_results = {}
        return_results["summary"] = subdocument_summary
        return_results["doc_id"] = subdocument_id
        return_results["embedding_id"] = subdocument_embedding_id
        return_results["ctx_start"] = int(math.floor(ctx_start + (token_length /2)))
        return_results["ctx_end"] = int(math.floor(ctx_end + (token_length /2)))
        return return_results
    
    def ingest_512_split(self, tokens, bucket, dir, parent_id, parent_ctx_start, ctx_start, ctx_end, filename, s3uri, web3uri, embedding_model, embedding_instruction, **kwargs):
        token_length = 512
        if len(tokens) > 512:
            Exception("tokens too long")
        subdocument_chunk_text = self.openai.detokenize(tokens, None, **kwargs)
        subdocument_chunk_id = self.convert_to_uuid(subdocument_chunk_text)
        subdocument_chunk_embedding = self.generate_embedding(subdocument_chunk_text, embedding_model, **kwargs)
        subdocument_chunk_embedding_id = self.convert_to_uuid(subdocument_chunk_embedding)
        subdocument_chunk_embedding_model_id = self.convert_to_uuid(embedding_model)
        self.datastore["vector_index"]["vector_index/data"][subdocument_chunk_embedding_model_id]["__data__"]["nodes_dict"][subdocument_chunk_embedding_id] = subdocument_chunk_id
        self.datastore["vector_store"]["vector_store/data"][subdocument_chunk_embedding_model_id][subdocument_chunk_embedding_id] = subdocument_chunk_embedding
        subdocument_chunk_metadata = self.extract_metadata(subdocument_chunk_text, bucket, dir, subdocument_chunk_id, filename, s3uri, web3uri, ctx_start, ctx_end)
        subdocument_chunk_dict = {}
        subdocument_chunk_dict["__node__"] = parent_id
        subdocument_chunk_dict["__type__"] = "1"
        subdocument_chunk_dict["__data__"] = {}
        subdocument_chunk_dict["__data__"]["hash"] = hashlib.sha256(subdocument_chunk_text.encode('utf-8')).hexdigest()
        subdocument_chunk_dict["__data__"]["node_type"] = 1
        subdocument_chunk_dict["__data__"]["node_id"] = parent_id
        subdocument_chunk_dict["__data__"]["metadata"] = subdocument_chunk_metadata
        relationships = self.extract_relationships(subdocument_chunk_id, subdocument_chunk_text, None, None, subdocument_chunk_dict, subdocument_chunk_metadata)
        subdocument_chunk = self.generate_document(subdocument_chunk_text, subdocument_chunk_id, subdocument_chunk_embedding_id, subdocument_chunk_metadata, relationships, ctx_start, ctx_end)
        self.datastore["doc_store"]["doc_store/data"][subdocument_chunk_id] = subdocument_chunk
        results = {}
        results["subdocument_id"] = subdocument_chunk_id
        results["embedding_id"] = subdocument_chunk_embedding_id
        results["ctx_start"] = parent_ctx_start +  (ctx_start + (token_length /2))
        results["ctx_start"] = int(math.floor(results["ctx_start"]))
        results["ctx_end"] = parent_ctx_start + (ctx_end + (token_length /2))
        results["ctx_end"] = int(math.floor(results["ctx_end"]))
        return results
    
    def ingest(self, src, dst, bucket, dir, **kwargs):
        documents = self.load_text(src, bucket, dir, **kwargs)
        document_index = {}
        self.datastore["doc_store"] ={
            "doc_store/data": {

            }
        }
        self.datastore["doc_index"] = {
            "doc_index/data": {

            }
        }
        self.datastore["vector_store"] = {
            "vector_store/data": {

            }
        }
        self.datastore["vector_index"] = {
            "vector_index/data": {

            }
        }

        ## prepare the vector index
        for vector_index_id in embedding_models:
            converted_vector_index_id = hashlib.sha256(vector_index_id.encode('utf-8')).hexdigest()
            converted_vector_index_id = converted_vector_index_id[:36]
            converted_vector_index_id = converted_vector_index_id[:8] + "-" + converted_vector_index_id[8:12] + "-" + converted_vector_index_id[12:16] + "-" + converted_vector_index_id[16:20] + "-" + converted_vector_index_id[20:]
            self.datastore["vector_index"]["vector_index/data"][converted_vector_index_id] = {}
            self.datastore["vector_index"]["vector_index/data"][converted_vector_index_id]["__type__"] = "vector_store"
            self.datastore["vector_index"]["vector_index/data"][converted_vector_index_id]["__data__"] = {}
            self.datastore["vector_index"]["vector_index/data"][converted_vector_index_id]["__data__"]["index_id"] = converted_vector_index_id
            self.datastore["vector_index"]["vector_index/data"][converted_vector_index_id]["__data__"]["index_model"] = vector_index_id
            self.datastore["vector_index"]["vector_index/data"][converted_vector_index_id]["__data__"]["nodes_dict"] = {}
            self.datastore["vector_store"]["vector_store/data"][converted_vector_index_id] = {}

        for document in documents:
            document_dict = {}
            s3uri = None
            web3uri = None
            modelDir = os.path.join(self.modelsDir, self.model+"@knn" + "/" + dir)
            filename = document.replace(modelDir +"/","")
            document = documents[document]
            tokens = []
            tokens = self.openai.tokenize(document, None, None, **kwargs)
            text = self.openai.detokenize(tokens, None, **kwargs)
            dochash = hashlib.sha256(document.encode('utf-8')).hexdigest()
            nodeid = hashlib.sha256(dochash.encode('utf-8')).hexdigest()
            dochash = dochash[:64]
            nodeid = dochash[:36]
            document_dict["__node__"] = nodeid
            document_dict["__type__"] = "1"
            document_dict["__data__"] = {}
            document_dict["__data__"]["hash"] = dochash
            document_dict["__data__"]["node_type"] = 1
            document_dict["__data__"]["filename"] = filename
            if dst == "s3":
                s3uri =  "s3://"+bucket+"/"+dir+"/"+filename
                document_dict["__data__"]["s3uri"] = "s3://"+bucket+"/"+dir+"/"+filename
            if dst == "web3":
                web3uri = self.web3.upload(filename, None, text)
                document_dict["__data__"]["web3storage"] = "https://" + web3uri + ".ipfs.w3s.link",
            document_index[filename] = document_dict
            nodetype = 1
            docid = self.convert_to_uuid(text)
            embedding = None
            start_char_idx = 0
            end_char_idx = len(tokens)
            text_template = "{metadata_str}\n\n{content}"
            metadata_template = "{key}: {value}"
            metadata_seperator = "\n"
            excluded_llm_metadata_keys = []
            excluded_embed_metadata_keys = []
            subdocument_embedding_ids = []
            subdocuments = {}
            subdocument_embedding_model_id = None
            subdocument_embedding_id = None

            if len(tokens) <= 512 and len(tokens) > 0:
                results = self.ingest_512_split(tokens, bucket, dir, docid, 0, 0, start_char_idx, end_char_idx, filename,  s3uri, web3uri, "bge-large-en-v1.5", **kwargs)
                pass
            elif len(tokens) > 512 and len(tokens) < 8191:
                results = self.ingest_8k_split(tokens, bucket, dir, docid, dochash, text, start_char_idx, end_char_idx, filename,  s3uri, web3uri, "bge-large-en-v1.5", "gpt-3.5-turbo-16k", **kwargs)
                pass
            elif len(tokens) >= 8191:
                ctx_start = 0
                ctx_end = 8191
                processed_tokens = 0
                document_text = self.openai.detokenize(tokens, None, **kwargs)
                document_id = self.convert_to_uuid(document_text)
                documents = []
                document_embeddings = []
                subdocument_summaries = []
                document_count = 0
                subdocuments = []
                subdocument_chunks = []
                subdocument_ids = []
                subdocument_embeddings = []
                subdocument_chunk_embedding_ids = []
                subdocument_count = 0
                subdocument_ctx_start = 0
                subdocument_ctx_end = 8191
                while processed_tokens < len(tokens):
                    subdocument_tokens = tokens[subdocument_ctx_start:subdocument_ctx_end]
                    subdocument_text = self.openai.detokenize(subdocument_tokens, None, **kwargs)
                    results = self.ingest_8k_split(subdocument_tokens, bucket, dir,  document_id, dochash, subdocument_text, subdocument_ctx_start, subdocument_ctx_end, filename,  s3uri, web3uri, "text-embedding-ada-002", "gpt-3.5-turbo-16k", **kwargs)
                    subdocument_summaries.append(results["summary"])
                    subdocument_ids.append(results["doc_id"])
                    subdocument_embeddings.append(results["embedding_id"])
                    subdocument_ctx_start = subdocument_ctx_start + int(math.floor((results["ctx_end"] - results["ctx_start"]) / 2))
                    subdocument_ctx_end = subdocument_ctx_end + int(math.floor((results["ctx_end"] - results["ctx_start"]) / 2))
                    processed_tokens = processed_tokens + int(math.floor(((results["ctx_end"] - results["ctx_start"]) / 2)))
                    subdocument_metadata = self.extract_metadata(subdocument_text, bucket, dir, results["doc_id"], filename, s3uri, web3uri, ctx_start, ctx_end)
                    subdocument_dict = {}
                    subdocument_dict["__node__"] = document_id
                    subdocument_dict["__type__"] = "1"
                    subdocument_dict["__data__"] = {}
                    subdocument_dict["__data__"]["hash"] = hashlib.sha256(subdocument_text.encode('utf-8')).hexdigest()
                    subdocument_dict["__data__"]["node_type"] = 1
                    subdocument_dict["__data__"]["node_id"] = document_id
                    subdocument_dict["__data__"]["metadata"] = subdocument_metadata
                    subdocuments.append(subdocument_dict)

                concat_summaries = " ".join(subdocument_summaries)
                concat_summarties_tokens = self.openai.tokenize(concat_summaries, None, None, **kwargs)
                if len(concat_summarties_tokens) > 15 * 1024:
                    Exception("concat summary too long")
                super_summary = self.generate_summary(concat_summaries, "gpt-3.5-turbo-16k", **kwargs)
                super_summary_metadata = self.extract_metadata(super_summary, bucket, dir, document_id, filename, s3uri, web3uri, None, None)
                super_summary_embedding = self.generate_embedding(super_summary, "text-embedding-ada-002", **kwargs)
                super_summary_dict = {}
                super_summary_dict["text"] = super_summary
                super_summary_dict["node_id"] = hashlib.sha256(concat_summaries.encode('utf-8')).hexdigest()
                super_summary_dict["node_type"] = 1
                super_summary_dict["metadata"] = super_summary_metadata
                metadata = self.extract_metadata(document_text, bucket, dir, document_id, filename, s3uri, web3uri, ctx_start, ctx_end)
                relationships = self.extract_relationships(document_id, document_text, super_summary_dict, subdocuments, None, metadata)
                doc_gen = self.generate_document(document_text, document_id, super_summary_embedding, metadata, relationships, ctx_start, ctx_end)
                pass
            else:
                Exception("document empty")
        
        self.datastore["doc_index"]["doc_index/data"] = document_index
        savedb = self.save_database(dst, bucket,  dir, self.datastore, **kwargs)
        #database = self.load_database(bucket, dir, **kwargs)
        #embeddings = self.load_embeddings(bucket, dir, **kwargs)
        return self.format(**kwargs)
    
    def generate_summary(self, text, model, **kwargs):
        self.openai = OpenAIAPI(None, meta=meta)
        system = "Summarize with 512 tokens or less the following text:"
        messages = [
            {
                "role": "user",
                "content": text
            }
        ]
        text_tokens = self.openai.tokenize(text + '\n"role": "user"\n"content": text\n'+ system, None, None, **kwargs)
        num_text_tokens = len(text_tokens)
        num_tokens = (16 * 1024) - num_text_tokens
        temperature = 0
        if model != None:
            self.model = model
        else:
            self.model = "gpt-3.5-turbo-16k"
        if self.model == "gpt-3.5-turbo-16k":
            #return 'random summary '
            return self.openai.chat(self.model , messages, system, temperature, num_tokens)["text"]
        else:
            return "model not implemented"

    def extract_metadata(self, chunk, bucket, dir, nodeid, filename, s3uri, web3uri, ctx_start, ctx_end, **kwargs):
        #extract metadata from a file
        metadata = {}
        if bucket == "uscode":
            if dir == "uscode":
                title = chunk.split("§")[0]
                title = title.replace(" U.S.C. ", "")
                number = chunk.split("§")[1]
                number = number.split(" ")[0]
                name = chunk.split("§")[1].split(" ", 1)[1]
                compare = []
                if ";" in name:
                    name1 = name.split(";")[0]
                    if len(name1) > 0:
                        compare.append(name1)
                if ":" in name:
                    name2 = name.split(":")[1]
                    if len(name2) > 0:
                        compare.append(name2)
                if "." in name:
                    name3 = name.split(".")[1]
                    if len(name3) > 0:
                        compare.append(name3)
                if len(compare) > 0:
                    name = min(compare, key=len)

                metadata = {
                    "filename": "./"+filename,
                    "ctx_start": ctx_start,
                    "ctx_end": ctx_end,
                    "title": title,
                    "number": number,
                    "name": name
                }

        elif bucket == "books":
            if dir == "books":
                metadata = {
                    "filename": filename,
                    "ctx_start": ctx_start,
                    "ctx_end": ctx_end
                }

        if s3uri != None:
            metadata["s3uri"] = "s3://"+bucket+"/"+dir+"/"+s3uri,
        if web3uri != None:
            metadata["web3storage"] = "https://" + web3uri + ".ipfs.w3s.link",

        return metadata
        
    def postgres(self, **kwargs):
        #create and prep a database
        return

    def uploads3(self, **kwargs):
        #store a document
        return
    
    def build_index(self, model, data, **kwargs):
        model_id = self.convert_to_uuid(model)
        index = hnswlib.Index(space = 'l2', dim = vector_index_ids[model])
        ids = []
        vectors = list(data.values())
        for i in range(len(vectors)):
            ids.append(i)
        ids = numpy.array(ids)
        vectors = numpy.array(vectors)
        index.add_items(vectors, ids)
        results = pickle.dumps(index)
        return results

    def topk(self, query, data, top_k, model,  bucket, dir, **kwargs):
        #model_embedding = self.HFEmbed.embed(model, None, query, **kwargs)
        #search a document
        index_keys = list(self.indexes.keys())
        if len(index_keys) > 0:
            if bucket in index_keys:
                index_sub_keys = list(self.indexes[bucket].keys())
                if len(index_sub_keys) > 0:
                    if dir in index_sub_keys:
                        if self.indexes[bucket][dir] != None:
                            this_index = self.indexes[bucket][dir]
                            pass
                        else:
                            self.indexes[bucket][dir] = {}
                            self.indexes[bucket][dir][model] = self.build_index(model, data, **kwargs)
                    else:
                            self.indexes[bucket][dir] = {}
                            self.indexes[bucket][dir][model] = self.build_index(model, data, **kwargs)
                else:
                    self.indexes[bucket][dir] = {}
                    self.indexes[bucket][dir][model] = self.build_index(model, data, **kwargs)
            else:
                self.indexes[bucket] = {}
                self.indexes[bucket][dir] = {}
                self.indexes[bucket][dir][model] = self.build_index(model, data, **kwargs)
        else:
            self.indexes[bucket] = {}
            self.indexes[bucket][dir] = {}
            self.indexes[bucket][dir][model] = self.build_index(model, data, **kwargs)

        if self.indexes[bucket][dir][model] != None:
            this_index = pickle.load(self.indexes[bucket][dir][model])
            labels, distances = this_index.knn_query(data, k = top_k)
            node_ids = []
            data_keys = list(data.keys())
            for label in labels:
                node_ids.append(data_keys[label])

            return node_ids , distances
        else:
            return None

    def randomk(self, query, data, top_k, model, **kwargs):
                
        model_embedding = self.generate_embedding(query, model, **kwargs)
        node_ids = list(data.keys())
        random.shuffle(node_ids)
        results = node_ids[:top_k]

        return results
    
    def search(self, query, top_k, model,  bucket, dir, **kwargs):
        ## list web3 documents
        documents = self.web3.list()
        metadata = None
        for document in documents:
            document_name = document["name"]
            if "metadata.json" in document_name:
                metadata = self.web3.download(document["cid"])
        
        if metadata != None:
            metadata = json.loads(metadata.text)
            for key in metadata:
                if key == "vector_index.json":
                    vector_index_cid = metadata[key]
                if key == "vector_store.json":
                    vector_store_cid = metadata[key]
                if key == "doc_index.json":
                    doc_index_cid = metadata[key]
                if key == "doc_store.json":
                    doc_store_cid = metadata[key]
            vector_index_cid = json.loads(self.web3.download(vector_index_cid).text)
            vector_store_cid = json.loads(self.web3.download(vector_store_cid).text)
            doc_index_cid = json.loads(self.web3.download(doc_index_cid).text)
            doc_store_cid = json.loads(self.web3.download(doc_store_cid).text)
            doc_store_node_ids = list(doc_store_cid["doc_store/data"].keys())


            first_model = "text-embedding-ada-002"
            first_model_id = self.convert_to_uuid(first_model)
            if model == None:
                second_model = "bge-large-en-v1.5"
                second_model_id = self.convert_to_uuid(second_model)
            else:
                second_model = model
                second_model_id = self.convert_to_uuid(second_model)
            
            if first_model_id in vector_store_cid["vector_store/data"]:
                first_model_store = vector_store_cid["vector_store/data"][first_model_id]
                first_model_index = vector_index_cid["vector_index/data"][first_model_id]["__data__"]["nodes_dict"]
            if second_model_id in vector_store_cid["vector_store/data"]:
                second_model_store = vector_store_cid["vector_store/data"][second_model_id]
                second_model_index = vector_index_cid["vector_index/data"][second_model_id]["__data__"]["nodes_dict"]
                inverse_second_model_index = {value: key for key, value in second_model_index.items()}
            
            first_search_document_ids = []
            if len(list(first_model_store.keys())) > 0:
                first_search_results = self.topk(query, first_model_store, top_k, first_model,  bucket, dir, **kwargs)
                for result in first_search_results:
                    first_search_document_ids.append(first_model_index[result])  

            if len(first_search_document_ids) > 0:
                second_search_document_ids = []
                first_search_document_children = []
                for first_search_document_id in first_search_document_ids:
                    this_document = doc_store_cid["doc_store/data"][first_search_document_id]
                    this_document_relationships = this_document["__data__"]["relationships"]
                    for relationship in this_document_relationships:
                        this_relationship = this_document_relationships[relationship]
                        if this_relationship["relationship"] == "__child__":
                            first_search_document_children.append(this_relationship["node_id"])
            # swap keys and values for second_model_index
            second_model_vectors = {}
            second_model_id = self.convert_to_uuid(second_model)
            second_document_ids = []
            if len(first_search_document_children) > 0:
                for first_search_document_child in first_search_document_children:
                    this_vector_id = inverse_second_model_index[first_search_document_child]
                    second_model_vectors[this_vector_id] = second_model_store[this_vector_id]
                second_search_results = self.topk(query, second_model_vectors, top_k, second_model,  bucket, dir, **kwargs)
                for result in second_search_results:
                    this_document_id = second_model_index[this_vector_id]
                    second_document_ids.append(this_document_id)
        
            
            final_search_results = []
            if len(second_document_ids) > 0:
                for second_document_id in second_document_ids:
                    this_document = doc_store_cid["doc_store/data"][second_document_id]
                    final_search_results.append(this_document)
                pass
            
            text_excerpts = []
            if len(final_search_results) > 0:
                for search_result in final_search_results:
                    this_search_result = search_result
                    this_result_metadata = this_search_result["__data__"]["metadata"]
                    this_result_web3storage = this_result_metadata["web3storage"]
                    this_result_data = requests.get(this_result_web3storage[0])
                    this_result_tokens = self.openai.tokenize(this_result_data.text, None, None, **kwargs)
                    this_result_ctx_start = this_result_metadata["ctx_start"]
                    this_result_ctx_end = this_result_metadata["ctx_end"]
                    this_result_token_excerpt = this_result_tokens[this_result_ctx_start:this_result_ctx_end]
                    this_result_text = self.openai.detokenize(this_result_token_excerpt, None, **kwargs)
                    text_excerpts.append(this_result_text)

        return text_excerpts, final_search_results
        
    def extract_relationships(self, docid, document, summary, children, parent, metadata, **kwargs):
        #extract relationships from a document
        nodetype = 1
        metadata = metadata
        dochash = hashlib.sha256(document.encode('utf-8')).hexdigest()
        relationships = {}
        relationship_number = 0
        relationships[relationship_number] = {}
        relationships[relationship_number]["relationship"] = "__self__"
        relationships[relationship_number]["node_id"] = docid
        relationships[relationship_number]["node_type"] = nodetype
        relationships[relationship_number]["metadata"] = metadata
        relationships[relationship_number]["hash"] = dochash
        relationship_number = relationship_number + 1
        if (summary is not None) and (len(summary) > 0):
            relationships[relationship_number] = {}
            relationships[relationship_number]["relationship"] = "__summary__"
            relationships[relationship_number]["node_id"] = summary["node_id"]
            relationships[relationship_number]["node_type"] = summary["node_type"]
            relationships[relationship_number]["metadata"] = summary["metadata"]
            relationships[relationship_number]["hash"] = hashlib.sha256(summary["text"].encode('utf-8')).hexdigest()
            relationship_number = relationship_number + 1

        if (parent is not None) and (len(parent) > 0):
            relationships[relationship_number] = {}
            relationships[relationship_number]["relationship"] = "__parent__"
            relationships[relationship_number]["node_id"] = parent["__data__"]["node_id"]
            relationships[relationship_number]["node_type"] = parent["__data__"]["node_type"]
            relationships[relationship_number]["metadata"] = parent["__data__"]["metadata"]
            relationships[relationship_number]["hash"] = parent["__data__"]["hash"]
            relationship_number = relationship_number + 1
        
        if children != None and len(children) > 0:
            for child in children:
                relationships[relationship_number] = {}
                relationships[relationship_number]["relationship"] = "__child__"
                relationships[relationship_number]["node_id"] = child["__data__"]["node_id"]
                relationships[relationship_number]["node_type"] = child["__data__"]["node_type"]
                relationships[relationship_number]["metadata"] = child["__data__"]["metadata"]
                relationships[relationship_number]["hash"] = child["__data__"]["hash"]
                relationship_number = relationship_number + 1
            
        return relationships

    def query(self, bucket, query, k, **kwargs):
        self.bucket = bucket
        self.query = query
        self.k = k
        self.method = 'query'
        return self.format(**kwargs)
    
    def create(self, **kwargs):
        #create and prep a s3 bucket
        pass
    
    def append(self, **kwargs):
        #ingest a file into a s3 bucket
        pass

    def pop(self, **kwargs):
        #pop a file from a s3 bucket
        pass

    def format(self, **kwargs):
        if self.method == 'index':
            results = "indexing not implemented"
            return results
        if self.method == 'query':
            results = "querying will be implemented"
            ## sliding window gzip ##

            return results
        pass

    def generate_embedding(self, text, model, **kwargs):
        resources = {
            "checkpoint": model
        }
        self.HFEmbed = HFEmbed(resources, meta=None)
        if model != None:
            self.model = model
        else:
            self.model = "text-embedding-ada-002"

        if self.model == "text-embedding-ada-002":
            ## make a list with four random floating point vectors ##
            random_vectors = []
            #for i in range(4):
            #    random_vectors.append(random.random()) 
            #return {"data":random_vectors}
            return self.openai.embedding(self.model, text, **kwargs)
        elif self.model == "bge-large-en-v1.5":
            random_vectors = []
            #for i in range(4):
            #    random_vectors.append(random.random())
            #return random_vectors
            return self.HFEmbed.embed("bge-large-en-v1.5", None, text, **kwargs).tolist()
        elif self.model == "bge-base-en":
            return self.HFEmbed.embed("bge-base-en", None, text, **kwargs).tolist()
        elif self.model == "gte-large":
            return self.HFEmbed.embed("gte-large", None, text, **kwargs).tolist()
        elif self.model == "gte-base":
            return self.HFEmbed.embed("gte-base", None, text, **kwargs).tolist()
        elif self.model == "bge-base-en-v1.5":
            return self.HFEmbed.embed("bge-base-en-v1.5", None, text, **kwargs).tolist()
        elif self.model == "instructor":
            return self.HFEmbed.embed(text, "instructor", None, text, **kwargs).tolist()
        elif self.model == "instructor-xl":
            return self.HFEmbed.embed("instructor-xl", None, text, **kwargs).tolist()
        else:
            return "model not implemented"

def main(resources, meta):
    Index = KNN(resources, meta)
    #results = Index.ingest("raw","web3","books","books")
    results = Index.search("text to search", 5, "bge-large-en-v1.5", "books","books" )

if __name__ == '__main__':
    endpoint = "https://object.ord1.coreweave.com"
    access_key = ""
    secret_key = ""
    host_bucket = "%(bucket)s.object.ord1.coreweave.com"
    bucket = "swissknife-knn"
    dir = "test"
    config = {
        "accessKey": access_key,
        "secretKey": secret_key,
        "endpoint": endpoint,
    }
    meta = {}
    meta["config"] = config
    meta['openai_api_key'] = ""
    meta["web3_api_key"] = ""
    main(None, meta=meta)
    pass
