"""
Module aibundle.py - bundled standard codebase.
Copyright (c) 2023 Chris Joakim, MIT License
Timestamp: 2023-10-30 16:27

Usage:  from pysrc.aibundle import Bytes, CogSearchClient, CogSvcsClient, Counter, Env, FS, Mongo, OpenAIClient, Storage, StringUtil, System
"""

import csv
import json
import os
import platform
import socket
import sys
import time
import traceback
import uuid

import certifi
import matplotlib
import openai
import pandas as pd
import psutil
import requests
import tiktoken

from numbers import Number
from typing import Iterator

from Levenshtein import distance
from azure.storage.blob import BlobServiceClient
from bson.objectid import ObjectId
from docopt import docopt
from openai.embeddings_utils import get_embedding
from openai.openai_object import OpenAIObject
from pymongo import MongoClient

# ==============================================================================

class Bytes():
    """
    This class is used to calculate KB, MB, GB, TB, PB, and EB values
    from a given number of bytes.  Also provides as_xxx() translation methods.
    """

    @classmethod
    def kilobyte(cls) -> int:
        """ Return the number of bytes in a kilobyte. """
        return 1024

    @classmethod
    def kilobytes(cls, kilobytes: Number) -> Number:
        """ Return the number of bytes in the given KB value. """
        return Bytes.kilobyte() * abs(float(kilobytes))

    @classmethod
    def megabyte(cls) -> int:
        """ Return the number of bytes in a megabyte."""
        return pow(1024, 2)

    @classmethod
    def megabytes(cls, megabytes: Number) -> Number:
        """ Return the number of bytes in the given MB value. """
        return Bytes.megabyte() * abs(float(megabytes))

    @classmethod
    def gigabyte(cls) -> int:
        """ Return the number of bytes in a gigabyte. """
        return pow(1024, 3)

    @classmethod
    def gigabytes(cls, gigabytes: Number) -> Number:
        """ Return the number of bytes in the given GB value. """
        return Bytes.gigabyte() * abs(float(gigabytes))

    @classmethod
    def terabyte(cls) -> int:
        """ Return the number of bytes in a terabyte. """
        return pow(1024, 4)

    @classmethod
    def terabytes(cls, terabytes: Number) -> Number:
        """ Return the number of bytes in the given TB value. """
        return Bytes.terabyte() * abs(float(terabytes))

    @classmethod
    def petabyte(cls) -> int:
        """ Return the number of bytes in a petabyte."""
        return pow(1024, 5)

    @classmethod
    def petabytes(cls, petabytes: Number) -> Number:
        """ Return the number of bytes in the given PB value. """
        return Bytes.petabyte() * abs(float(petabytes))

    @classmethod
    def exabyte(cls) -> int:
        """ Return the number of bytes in an exabyte. """
        return pow(1024, 6)

    @classmethod
    def exabytes(cls, exabytes: Number) -> Number:
        """ Return the number of bytes in the given EB value. """
        return Bytes.exabyte() * abs(float(exabytes))

    @classmethod
    def as_kilobytes(cls, num_bytes: Number) -> Number:
        """ Return the number of KB for the given number of bytes. """
        return float(abs(num_bytes)) / float(cls.kilobyte())

    @classmethod
    def as_megabytes(cls, num_bytes: Number) -> Number:
        """ Return the number of MB for the given number of bytes. """
        return float(abs(num_bytes)) / float(cls.megabyte())

    @classmethod
    def as_gigabytes(cls, num_bytes: Number) -> Number:
        """ Return the number of GB for the given number of bytes. """
        return float(abs(num_bytes)) / float(cls.gigabyte())

    @classmethod
    def as_terabytes(cls, num_bytes: Number) -> Number:
        """ Return the number of TB for the given number of bytes. """
        return float(abs(num_bytes)) / float(cls.terabyte())

    @classmethod
    def as_petabytes(cls, num_bytes: Number) -> Number:
        """ Return the number of PB for the given number of bytes. """
        return float(abs(num_bytes)) / float(cls.petabyte())
    @classmethod
    def as_exabytes(cls, num_bytes: Number) -> Number:
        """ Return the number of EB for the given number of bytes. """
        return float(abs(num_bytes)) / float(cls.exabyte())
# ==============================================================================

class CogSearchClient():
    """
    This class is used to access an Azure Cognitive Search account
    via its REST API endpoints.
    """
    def __init__(self, opts):
        self.opts = opts
        self.user_agent = {'User-agent': 'Mozilla/5.0'}
        #self.search_api_version = '2021-04-30-Preview'
        self.search_api_version = '2023-07-01-Preview'
        self.verbose = False

        try:
            self.search_name = opts['name']
            self.search_url  = opts['url']
            self.search_admin_key = opts['admin_key']
            self.search_query_key = opts['query_key']
            if self.search_url.endswith('/'):
                self.search_url = self.search_url[:-1]
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())

        self.u = None  # the current url
        self.r = None  # the current requests response object
        self.config = dict()

        self.admin_headers = dict()
        self.admin_headers['Content-Type'] = 'application/json'
        self.admin_headers['api-key'] = self.search_admin_key

        self.query_headers = dict()
        self.query_headers['Content-Type'] = 'application/json'
        self.query_headers['api-key'] = self.search_query_key

    def display_config(self):
        print('search_name:      {}'.format(self.search_name))
        print('search_url:       {}'.format(self.search_url))
        print('search_admin_key: {}'.format(self.search_admin_key))
        print('search_query_key: {}'.format(self.search_query_key))
        print('admin_headers:\n{}'.format(json.dumps(self.admin_headers, sort_keys=False, indent=2)))
        print('query_headers:\n{}'.format(json.dumps(self.query_headers, sort_keys=False, indent=2)))

    # API Invoking methods:

    def list_indexes(self):
        url = self.list_indexes_url()
        self.http_request('list_indexes', 'get', url, self.admin_headers)

    def list_indexers(self):
        url = self.list_indexers_url()
        self.http_request('list_indexers', 'get', url, self.admin_headers)

    def list_datasources(self):
        url = self.list_datasources_url()
        self.http_request('list_datasources', 'get', url, self.admin_headers)

    def get_index(self, name):
        url = self.get_index_url(name)
        self.http_request('get_index', 'get', url, self.admin_headers)

    def get_indexer(self, name):
        url = self.get_indexer_url(name)
        self.http_request('get_indexer', 'get', url, self.admin_headers)

    def get_indexer_status(self, name):
        url = self.get_indexer_status_url(name)
        self.http_request('get_indexer_status', 'get', url, self.admin_headers)

    def get_datasource(self, name):
        url = self.get_datasource_url(name)
        self.http_request('get_datasource', 'get', url, self.admin_headers)

    def create_index(self, name, schema_file):
        self.modify_index('create', name, schema_file)

    def update_index(self, name, schema_file):
        self.modify_index('update', name, schema_file)

    def delete_index(self, name):
        self.modify_index('delete', name, None)

    def modify_index(self, action, name, schema_file):
        if self.verbose:
            print(f'modify_index {action} {name} {schema_file}')
        schema = None
        if action in ['create', 'update']:
            filename = f'schemas/{schema_file}'
            schema = FS.read_json(filename)

        if action == 'create':
            http_method = 'post'
            url = self.create_index_url()
        elif action == 'update':
            http_method = 'put'
            url = self.modify_index_url(name)
        elif action == 'delete':
            http_method = 'delete'
            url = self.modify_index_url(name)

        function = '{}_index_{}'.format(action, name)
        self.http_request(function, http_method, url, self.admin_headers, schema)

    def create_indexer(self, name, schema_file):
        self.modify_indexer('create', name, schema_file)

    def update_indexer(self, name, schema_file):
        self.modify_indexer('update', name, schema_file)

    def delete_indexer(self, name):
        self.modify_indexer('delete', name, None)

    def modify_indexer(self, action, name, schema_file):
        # read the schema json file if necessary
        schema = None
        if action in ['create', 'update']:
            filename = f'schemas/{schema_file}'
            schema = FS.read_json(filename)

        if action == 'create':
            http_method = 'post'
            url = self.create_indexer_url()
        elif action == 'update':
            http_method = 'put'
            url = self.modify_indexer_url(name)
        elif action == 'delete':
            http_method = 'delete'
            url = self.modify_indexer_url(name)

        function = '{}_indexer_{}'.format(action, name)
        self.http_request(function, http_method, url, self.admin_headers, schema)

    def reset_indexer(self, name):
        url = self.reset_indexer_url(name)
        self.http_request('reset_indexer', 'post', url, self.admin_headers)

    def run_indexer(self, name):
        url = self.run_indexer_url(name)
        self.http_request('run_indexer', 'post', url, self.admin_headers)

    def create_cosmos_nosql_datasource(self, acct_envvar, key_envvar, dbname, container):
        acct = os.environ[acct_envvar]
        key  = os.environ[key_envvar]
        conn_str = self.cosmos_nosql_datasource_name_conn_str(acct, key, dbname)
        body = self.cosmosdb_nosql_datasource_post_body()
        body['name'] = self.cosmos_nosql_datasource_name(dbname, container)
        body['credentials']['connectionString'] = conn_str
        body['container']['name'] = container
        body['dataDeletionDetectionPolicy'] = None
        body['encryptionKey'] = None
        body['identity'] = None

        url = self.create_datasource_url()
        function = 'create_cosmos_nosql_datasource_{}_{}'.format(dbname, container)
        self.http_request(function, 'post', url, self.admin_headers, body)

    def delete_datasource(self, name):
        url = self.modify_datasource_url(name)
        function = 'delete_datasource{}'.format(name)
        self.http_request(function, 'delete', url, self.admin_headers, None)

    def create_synmap(self, name, schema_file):
        self.modify_synmap('create', name, schema_file)

    def update_synmap(self, name, schema_file):
        self.modify_synmap('update', name, schema_file)

    def delete_synmap(self, name):
        self.modify_synmap('delete', name, None)

    def modify_synmap(self, action, name, schema_file):
        # read the schema json file if necessary
        schema = None
        if action in ['create', 'update']:
            schema_file = 'schemas/{}.json'.format(schema_file)
            schema = self.load_json_file(schema_file)
            schema['name'] = name

        if action == 'create':
            http_method = 'post'
            url = self.create_synmap_url()
        elif action == 'update':
            http_method = 'put'
            url = self.modify_synmap_url(name)
        elif action == 'delete':
            http_method = 'delete'
            url = self.modify_synmap_url(name)

        function = '{}_synmap_{}'.format(action, name)
        self.http_request(function, http_method, url, self.admin_headers, schema)

    def search_index(self, idx_name, search_name, search_params):
        url = self.search_index_url(idx_name)
        if self.verbose:
            print('---')
            print('search_index: {} {} -> {}'.format(idx_name, search_name, search_params))
            print('search_index url: {}'.format(url))
            print('url:     {}'.format(url))
            print('method:  {}'.format('POST'))
            print('params:  {}'.format(search_params))
            print('headers: {}'.format(self.admin_headers))

        # Invoke the search via the HTTP API
        r = requests.post(url=url, headers=self.admin_headers, json=search_params)
        if self.verbose:
            print('response: {}'.format(r))
        if r.status_code == 200:
            resp_obj = json.loads(r.text)
            outfile  = 'tmp/search_{}.json'.format(search_name)
            self.write_json_file(resp_obj, outfile)
        return r

    def lookup_doc(self, index_name, doc_key):
        if self.verbose:
            print('lookup_doc: {} {}'.format(index_name, doc_key))
        url = self.lookup_doc_url(index_name, doc_key)
        headers = self.query_headers
        function = 'lookup_doc_{}_{}'.format(index_name, doc_key)
        r = self.http_request(function, 'get', url, self.query_headers)

    def http_request(self, function_name, method, url, headers={}, json_body={}):
        """
        This is a generic method which invokes ALL HTTP Requests to
        the Azure Search Service.
        """
        if self.verbose:
            print('===')
            print("http_request: {} {} {}\nheaders: {}\nbody: {}".format(
                function_name, method.upper(), url, headers, json_body))
            print("http_request name/method/url: {} {} {}".format(
                function_name, method.upper(), url))
            print("http_request headers:\n{}".format(json.dumps(
                headers, sort_keys=False, indent=2)))
            print("http_request body:\n{}".format(json.dumps(
                json_body, sort_keys=False, indent=2)))

        if self.no_http():
            return {}
        else:
            r = None
            if method == 'get':
                r = requests.get(url=url, headers=headers)
            elif method == 'post':
                r = requests.post(url=url, headers=headers, json=json_body)
            elif method == 'put':
                r = requests.put(url=url, headers=headers, json=json_body)
            elif method == 'delete':
                r = requests.delete(url=url, headers=headers)
            else:
                print('error; unexpected method value passed to invoke: {}'.format(method))
            if self.verbose:
                print('response: {}'.format(r))
            if r.status_code < 300:
                try:
                    # Save the request and response data as a json file in tmp/
                    outfile  = 'tmp/{}_{}.json'.format(function_name, int(self.epoch()))
                    data = dict()
                    data['function_name'] = function_name
                    data['method'] = method
                    data['url'] = url
                    data['body'] = json_body
                    data['filename'] = outfile
                    data['resp_status_code'] = r.status_code
                    try:
                        data['resp_obj'] = r.json()
                    except:
                        pass # this is expected as some requests don't return a response, like http 204
                    self.write_json_file(data, outfile)
                except Exception as e:
                    print("exception saving http response".format(e))
                    print(traceback.format_exc())
            return r

    # Datasource Name methods:

    def blob_datasource_name(self, container):
        return 'azureblob-{}'.format(container)

    def cosmos_nosql_datasource_name(self, dbname, container):
        return 'cosmosdb-nosql-{}-{}'.format(dbname, container)

    def cosmos_nosql_datasource_name_conn_str(self, acct, key, dbname):
        # acct = os.environ['AZURE_COSMOSDB_NOSQL_ACCT']
        # key  = os.environ['AZURE_COSMOSDB_NOSQL_RO_KEY1']
        return 'AccountEndpoint=https://{}.documents.azure.com;AccountKey={};Database={}'.format(
            acct, key, dbname)

    def cosmos_nosql_datasource_name(self, dbname, container):
        return 'cosmosdb-nosql-{}-{}'.format(dbname, container)

    # URL methods below:

    def list_indexes_url(self):
        return '{}/indexes?api-version={}'.format(self.search_url, self.search_api_version)

    def list_indexers_url(self):
        return '{}/indexers?api-version={}'.format(self.search_url, self.search_api_version)

    def list_datasources_url(self):
        return '{}/datasources?api-version={}'.format(self.search_url, self.search_api_version)

    def list_skillsets_url(self):
        return '{}/skillsets?api-version={}'.format(self.search_url, self.search_api_version)

    def get_index_url(self, name):
        return '{}/indexes/{}?api-version={}'.format(self.search_url, name, self.search_api_version)

    def get_indexer_url(self, name):
        return '{}/indexers/{}?api-version={}'.format(self.search_url, name, self.search_api_version)

    def get_indexer_status_url(self, name):
        return '{}/indexers/{}/status?api-version={}'.format(self.search_url, name, self.search_api_version)

    def get_datasource_url(self, name):
        return '{}/datasources/{}?api-version={}'.format(self.search_url, name, self.search_api_version)

    def get_skillset_url(self, name):
        return '{}/skillsets/{}?api-version={}'.format(self.search_url, name, self.search_api_version)

    def create_index_url(self):
        return '{}/indexes?api-version={}'.format(self.search_url, self.search_api_version)

    def modify_index_url(self, name):
        return '{}/indexes/{}?api-version={}'.format(self.search_url, name, self.search_api_version)

    def create_indexer_url(self):
        return '{}/indexers?api-version={}'.format(self.search_url, self.search_api_version)

    def modify_indexer_url(self, name):
        return '{}/indexers/{}?api-version={}'.format(self.search_url, name, self.search_api_version)

    def reset_indexer_url(self, name):
        return '{}/indexers/{}/reset?api-version={}'.format(self.search_url, name, self.search_api_version)

    def run_indexer_url(self, name):
        return '{}/indexers/{}/run?api-version={}'.format(self.search_url, name, self.search_api_version)

    def create_datasource_url(self):
        return '{}/datasources?api-version={}'.format(self.search_url, self.search_api_version)

    def modify_datasource_url(self, name):
        return '{}/datasources/{}?api-version={}'.format(self.search_url, name, self.search_api_version)

    def create_synmap_url(self):
        return '{}/synonymmaps?api-version={}'.format(self.search_url, self.search_api_version)

    def modify_synmap_url(self, name):
        return '{}/synonymmaps/{}?api-version={}'.format(self.search_url, name, self.search_api_version)

    def create_skillset_url(self):
        return '{}/skillsets?api-version={}'.format(self.search_url, self.search_api_version)

    def modify_skillset_url(self, name):
        return '{}/skillsets/{}?api-version={}'.format(self.search_url, name, self.search_api_version)

    def search_index_url(self, idx_name):
        return '{}/indexes/{}/docs/search?api-version={}'.format(self.search_url, idx_name, self.search_api_version)

    def lookup_doc_url(self, index_name, doc_key):
        return '{}/indexes/{}/docs/{}?api-version={}'.format(self.search_url, index_name, doc_key, self.search_api_version)

    # Schema methods below:

    def blob_datasource_post_body(self):
        body = {
            "name" : "... populate me ...",
            "type" : "azureblob",
            "credentials" : {
                "connectionString" : "... populate me ..." },
                "container" :
                    { "name" : "... populate me ..." }
        }
        return body

    def cosmosdb_nosql_datasource_post_body(self):
        schema = {
            "name": "... populate me ...",
            "type": "cosmosdb",
            "credentials": {
                "connectionString": "... populate me ..."
            },
            "container": {
                "name": "... populate me ...",
                "query": None
            },
            "dataChangeDetectionPolicy": {
                "@odata.type": "#Microsoft.Azure.Search.HighWaterMarkChangeDetectionPolicy",
                "highWaterMarkColumnName": "_ts"
            }
        }
        return schema

    def cosmosdb_nosql_datasource_post_body(self):
        schema = {
            "name": "... populate me ...",
            "type": "cosmosdb",
            "credentials": {
                "connectionString": "... populate me ..."
            },
            "container": {
                "name": "... populate me ...",
                "query": None
            },
            "dataChangeDetectionPolicy": {
                "@odata.type": "#Microsoft.Azure.Search.HighWaterMarkChangeDetectionPolicy",
                "highWaterMarkColumnName": "_ts"
            },
            "dataDeletionDetectionPolicy": "null",
            "encryptionKey": "null",
            "identity": "null"
        }
        return schema

    def indexer_schema(self, indexer_name, index_name, datasource_name):
        schema = {}
        schema['name'] = indexer_name
        schema['dataSourceName'] = datasource_name
        schema['targetIndexName'] = index_name
        schema['schedule'] = { "interval" : "PT2H" }
        return schema

    # Other methods

    def epoch(self):
        return time.time()

    def no_http(self):
        for arg in sys.argv:
            if arg == '--no-http':
                return True
        return False

    def load_json_file(self, infile):
        with open(infile, 'rt') as json_file:
            return json.loads(str(json_file.read()))

    def write_json_file(self, obj, outfile):
        with open(outfile, 'wt') as f:
            f.write(json.dumps(obj, sort_keys=False, indent=2))
            print('file written: {}'.format(outfile))
# ==============================================================================

class CogSvcsClient():
    """
    This class is used to access an Azure Cognitive Services account
    via REST API endpoints.
    """
    def __init__(self, opts):
        self.opts = opts

    # TextAnalytics methods:
    # See https://learn.microsoft.com/en-us/rest/api/language/

    def text_analytics_sentiment(self, text_lines, language):
        url = self.get_cogsvcs_target_url('text/analytics/v3.0/sentiment')
        headers = self.get_cogsvcs_headers()
        body = {'documents': []}
        for line in text_lines:
            body['documents'].append({'id': str(uuid.uuid4()), 'language': language, 'text': str(line).strip()})
        return requests.post(url, headers=headers, data=json.dumps(body))

    def text_analytics_key_phrases(self, text_lines, language):
        url = self.get_cogsvcs_target_url('text/analytics/v3.0/keyPhrases')
        headers = self.get_cogsvcs_headers()
        text = ' '.join(text_lines)
        body = {'documents': []}
        body['documents'].append({'id': str(uuid.uuid4()), 'language': language, 'text': str(text).strip()})
        return requests.post(url, headers=headers, data=json.dumps(body))

    def text_analytics_entities(self, text_lines, language):
        url = self.get_cogsvcs_target_url('text/analytics/v3.0/entities/recognition/general')
        headers = self.get_cogsvcs_headers()
        text = ' '.join(text_lines)
        body = {'documents': []}
        body['documents'].append({'id': str(uuid.uuid4()), 'language': language, 'text': str(text).strip()})
        return requests.post(url, headers=headers, data=json.dumps(body))

    # TextTranslation methods:

    def text_translate_formats(self):
        url = self.get_cogsvcs_target_url('translator/text/batch/v1.0/documents/formats')
        headers = self.get_cogsvcs_headers()
        return requests.get(url, headers=headers)

    def text_translate_languages(self):
        # https://learn.microsoft.com/en-us/azure/cognitive-services/translator/reference/v3-0-languages
        url = 'https://api.cognitive.microsofttranslator.com/languages?api-version=3.0'
        return requests.get(url, headers={})

    def text_translate(self, text_lines, to_lang):
        # https://learn.microsoft.com/en-us/azure/cognitive-services/translator/reference/v3-0-translate
        url = f'https://api.cognitive.microsofttranslator.com/translate?api-version=3.0&to={to_lang}'
        headers, body = self.get_texttranslator_headers(), []
        for line in text_lines:
            body.append({'Text': str(line).strip()})
        return requests.post(url, headers=headers, data=json.dumps(body))

    # ComputerVision methods:

    def image_analyze(self, image_url):
        # See https://learn.microsoft.com/en-us/rest/api/computervision/3.1/analyze-image/analyze-image?tabs=HTTP
        url = self.get_cogsvcs_target_url('vision/v3.1/analyze?visualFeatures=Categories,Adult,Tags,Description,Faces,Color,ImageType,Objects,Brands&details=Landmarks&language=en')
        headers = self.get_cogsvcs_headers()
        body = {'url': image_url}
        return requests.post(url, headers=headers, data=json.dumps(body))

    def image_describe(self, image_url):
        # See https://learn.microsoft.com/en-us/rest/api/computervision/3.1/describe-image/describe-image?tabs=HTTP
        url = self.get_cogsvcs_target_url('vision/v3.1/describe?maxCandidates=2&language=en')
        headers = self.get_cogsvcs_headers()
        body = {'url': image_url}
        return requests.post(url, headers=headers, data=json.dumps(body))

    def image_tag(self, image_url):
        # See https://learn.microsoft.com/en-us/rest/api/computervision/3.1/tag-image/tag-image?tabs=HTTP
        url = self.get_cogsvcs_target_url('vision/v3.1/tag?language=en')
        headers = self.get_cogsvcs_headers()
        body = {'url': image_url}
        return requests.post(url, headers=headers, data=json.dumps(body))

    def image_read(self, image_url, callback_sleep_secs=3):
        try:
            # See https://learn.microsoft.com/en-us/rest/api/computervision/3.1/read/read?tabs=HTTP
            # This is a two-step process - first submit the image for analysis, then retrieve the results.
            url = self.get_cogsvcs_target_url('vision/v3.1/read/analyze?language=en')
            headers = self.get_cogsvcs_headers()
            body = {'url': image_url}
            # print(json.dumps(body))
            resp = requests.post(url, headers=headers, data=json.dumps(body))
            # print(resp)
            # print(resp.headers)
            callback_url = resp.headers['Operation-Location']
            print(f'callback_url: {callback_url}')
            time.sleep(callback_sleep_secs)
            return requests.get(callback_url, headers=headers)
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())
        return None

    # Face methods:

    def face_detect(self, image_url):
        # See https://learn.microsoft.com/en-us/rest/api/faceapi/face/detect-with-url?tabs=HTTP
        url = self.get_cogsvcs_target_url('face/v1.0/detect?returnFaceId=false&returnFaceLandmarks=true')
        headers = self.get_face_headers()
        body = {'url': image_url}
        return requests.post(url, headers=headers, data=json.dumps(body))

    # "private" methods below

    def get_cogsvcs_rest_endpoint(self, alt_env_var_name=None):
        if alt_env_var_name == None:
            return Env.var('AZURE_COGSVCS_ALLIN1_URL')
        else:
            return Env.var(alt_env_var_name)

    def get_cogsvcs_target_url(self, path):
        if self.get_cogsvcs_rest_endpoint().endswith('/'):
            return '{}{}'.format(self.get_cogsvcs_rest_endpoint(), path)
        else:
            return '{}/{}'.format(self.get_cogsvcs_rest_endpoint(), path)

    def get_face_rest_endpoint(self, alt_env_var_name=None):
        if alt_env_var_name == None:
            return Env.var('AZURE_COGSVCS_FACE_URL')
        else:
            return Env.var(alt_env_var_name)

    def get_face_target_url(self, path):
        if self.get_cogsvcs_rest_endpoint().endswith('/'):
            return '{}{}'.format(self.get_face_rest_endpoint(), path)
        else:
            return '{}/{}'.format(self.get_face_rest_endpoint(), path)

    def get_cogsvcs_headers(self, alt_env_var_name=None):
        key = None
        if alt_env_var_name == None:
            key = Env.var('AZURE_COGSVCS_ALLIN1_KEY')
        else:
            key = Env.var(alt_env_var_name)
        headers = {
            'Ocp-Apim-Subscription-Key': key,
            'Content-Type': 'application/json',
        }
        return headers

    def get_texttranslator_headers(self, alt_env_var_name=None):
        key = None
        if alt_env_var_name == None:
            key = Env.var('AZURE_COGSVCS_TEXTTRAN_KEY')
        else:
            key = Env.var(alt_env_var_name)
        headers = {
            'Ocp-Apim-Subscription-Key': key,
            'Ocp-Apim-Subscription-Region': 'eastus',
            'Content-Type': 'application/json',
        }
        return headers

    def get_face_headers(self, alt_env_var_name=None):
        key = None
        if alt_env_var_name == None:
            key = Env.var('AZURE_COGSVCS_FACE_KEY')
        else:
            key = Env.var(alt_env_var_name)
        headers = {
            'Ocp-Apim-Subscription-Key': key,
            'Ocp-Apim-Subscription-Region': 'eastus',
            'Content-Type': 'application/json',
        }
        return headers

    def verbose(self):
        for arg in sys.argv:
            if arg == '--verbose':
                return True
        return False
# ==============================================================================

class Counter():
    """
    This class implements a simple int counter with an underlying dict object.
    """
    def __init__(self):
        self.data = {}

    def increment(self, key: str) -> None:
        """ Increment the given key by 1. """
        keys = self.data.keys()
        if key in keys:
            self.data[key] = self.data[key] + 1
        else:
            self.data[key] = 1

    def decrement(self, key: str) -> None:
        """ Decrement the given key by 1. """
        keys = self.data.keys()
        if key in keys:
            self.data[key] = self.data[key] - 1
        else:
            self.data[key] = -1

    def get_value(self, key: str) -> int:
        """ Get the int value of the given key. """
        keys = self.data.keys()
        if key in keys:
            return self.data[key]
        return 0

    def get_data(self) -> dict:
        """ Return the underlying dict object. """
        return self.data
# ==============================================================================

class Env():
    """
    This class is used to read the host environment, such as username and
    environment variables.  It also has methods for command-line flag
    argument processing.
    """

    @classmethod
    def var(cls, name: str, default=None) -> str | None:
        """ Return the value of the given environment variable name, or None. """
        if name in os.environ:
            return os.environ[name]
        return default

    @classmethod
    def username(cls) -> str | None:
        """ Return the USERNAME (Windows) or USER (macOS/Linux) value. """
        usr = cls.var('USERNAME')
        if usr is None:
            usr = cls.var('USER')
        return usr

    @classmethod
    def epoch(cls) -> float:
        """ Return the current epoch time, as time.time() """
        return time.time()

    @classmethod
    def verbose(cls) -> bool:
        """ Return a boolean indicating if --verbose or -v is in the command-line. """
        flags = [ '--verbose', '-v' ]
        for arg in sys.argv:
            for flag in flags:
                if arg == flag:
                    return True
        return False

    @classmethod
    def boolean_arg(cls, flag: str) -> bool:
        """ Return a boolean indicating if the given arg is in the command-line. """
        for arg in sys.argv:
            if arg == flag:
                return True
        return False
# ==============================================================================

class FS():
    """
    This class is used to do IO operations vs the local File System.
    """
    @classmethod
    def as_unix_filename(cls, filename: str) -> str:
        """ Return the given filename with unix slashes. """
        if filename.upper().startswith("C:"):
            return filename[2:].replace("\\", "/")
        return filename

    @classmethod
    def read(cls, infile: str) -> str | None:
        """ Read the given file, return the file contents str or None. """
        if os.path.isfile(infile):
            with open(file=infile, encoding='utf-8', mode='rt') as file:
                return file.read()
        return None

    @classmethod
    def readr(cls, infile: str) -> str | None:
        """ Read the given file with mode 'r', return the file contents str or None. """
        if os.path.isfile(infile):
            with open(file=infile, encoding='utf-8', mode='r') as file:
                return file.read()
        return None

    @classmethod
    def read_binary(cls, infile: str) -> str | None:
        """ Read the given binary file with mode 'rb', return the bytes or None. """
        if os.path.isfile(infile):
            with open(file=infile, mode='rb') as file:
                return file.read()
        return None

    @classmethod
    def read_lines(cls, infile: str) -> list[str] | None:
        """ Read the given file, return an array of lines(strings) or None """
        if os.path.isfile(infile):
            lines = []
            with open(file=infile, encoding='utf-8', mode='rt') as file:
                for line in file:
                    lines.append(line)
            return lines
        return None

    @classmethod
    def read_single_line(cls, infile: str) -> str | None:
        """ Read the given file, return the first line or None """
        lines = cls.read_lines(infile)
        if lines is not None:
            if len(lines) > 0:
                return lines[0].strip()
        return None

    @classmethod
    def read_encoded_lines(cls, infile: str, encoding='cp1252') -> list[str] | None:
        """
        Read the given file, with the given encoding.
        Return an array of lines(strings) or None.
        """
        if os.path.isfile(infile):
            lines = []
            with open(file=infile, encoding=encoding, mode='rt') as file:
                for line in file:
                    lines.append(line)
            return lines
        return None

    @classmethod
    def read_win_cp1252(cls, infile: str) -> str | None:
        """
        Read the given file with Windows encoding cp1252.
        Return an array of lines(strings) or None.
        """
        if os.path.isfile(infile):
            with open(file=os.path.join(infile), encoding='cp1252', mode='r') as file:
                return file.read()
        return None

    @classmethod
    def read_csv_as_dicts(cls, infile: str, delim=',', dialect='excel') -> list[str] | None:
        """
        Read the given csv filename, return an array of dicts or None.
        """
        if os.path.isfile(infile):
            rows = []
            with open(file=infile, encoding='utf-8', mode='rt') as csvfile:
                rdr = csv.DictReader(csvfile, dialect=dialect, delimiter=delim)
                for row in rdr:
                    rows.append(row)
            return rows
        return None

    @classmethod
    def read_csv_as_rows(cls, infile: str, delim=',', skip=0) -> list[str] | None:
        """
        Read the given csv filename, return an array of csv rows or None.
        """
        if os.path.isfile(infile):
            rows = []
            with open(file=infile, encoding='utf-8', mode='rt') as csvfile:
                rdr = csv.reader(csvfile, delimiter=delim)
                for idx, row in enumerate(rdr):
                    if idx >= skip:
                        rows.append(row)
            return rows
        return None

    @classmethod
    def read_json(cls, infile: str, encoding='utf-8') -> dict | list | None:
        """ Read the given JSON file, return either a list, a dict, or None. """
        if os.path.isfile(infile):
            with open(file=infile, encoding=encoding, mode='rt') as file:
                return json.loads(file.read())
        return None

    @classmethod
    def write_json(cls, obj: object, outfile: str, pretty=True, verbose=True) -> None:
        """ Write the given object to the given file as JSON. """
        if obj is not None:
            jstr = None
            if pretty is True:
                jstr = json.dumps(obj, sort_keys=False, indent=2)
            else:
                jstr = json.dumps(obj)

            with open(file=outfile, encoding='utf-8', mode='w') as file:
                file.write(jstr)
                if verbose is True:
                    print(f'file written: {outfile}')

    @classmethod
    def write_lines(cls, lines: list[str], outfile: str, verbose=True) -> None:
        """ Write the given str lines to the given file. """
        if lines is not None:
            with open(file=outfile, encoding='utf-8', mode='w') as file:
                for line in lines:
                    file.write(line + "\n") # os.linesep)  # \n works on Windows
                if verbose is True:
                    print(f'file written: {outfile}')

    @classmethod
    def text_file_iterator(cls, infile: str) -> Iterator[str] | None:
        """ Return a line generator that can be iterated with iterate() """
        if os.path.isfile(infile):
            with open(file=infile, encoding='utf-8', mode='rt') as file:
                for line in file:
                    yield line.strip()

    @classmethod
    def write(cls, outfile: str, string_value: str, verbose=True) -> None:
        """ Write the given string to the given file. """
        if outfile is not None:
            if string_value is not None:
                with open(file=outfile, encoding='utf-8', mode='w') as file:
                    file.write(string_value)
                    if verbose is True:
                        print(f'file written: {outfile}')

    @classmethod
    def list_directories_in_dir(cls, basedir: str) -> list[str] | None:
        """ Return a list of directories in the given directory, or None. """
        if os.path.isdir(basedir):
            files = []
            for file in os.listdir(basedir):
                dir_or_file = os.path.join(basedir, file)
                if os.path.isdir(dir_or_file):
                    files.append(file)
            return files
        return None

    @classmethod
    def list_files_in_dir(cls, basedir: str) -> list[str] | None:
        """ Return a list of files in the given directory, or None. """
        if os.path.isdir(basedir):
            files = []
            for file in os.listdir(basedir):
                dir_or_file = os.path.join(basedir, file)
                if os.path.isdir(dir_or_file):
                    pass
                else:
                    files.append(file)
            return files
        return None

    @classmethod
    def walk(cls, directory: str) -> list[dict] | None:
        """ Return a list of dicts for each file in the given directory, or None. """
        if os.path.isdir(directory):
            files = []
            for dir_name, _, base_names in os.walk(directory):
                for base_name in base_names:
                    full_name = f"{dir_name}/{base_name}"
                    entry = {}
                    entry['base'] = base_name
                    entry['dir'] = dir_name
                    entry['full'] = full_name
                    entry['abspath'] = os.path.abspath(full_name)
                    files.append(entry)
            return files
        return None

    @classmethod
    def read_csvfile_into_rows(cls, infile: str, delim=',') -> list | None:
        """ Read the given csv filename, return an array of csv rows or None. """
        if os.path.isfile(infile):
            rows = []  # return a list of csv rows
            with open(file=infile, encoding='utf-8', mode='rt') as csvfile:
                reader = csv.reader(csvfile, delimiter=delim)
                for row in reader:
                    rows.append(row)
            return rows
        return None

    @classmethod
    def read_csvfile_into_objects(cls, infile: str, delim=',') -> list[object] | None:
        """ Read the given csv filename, return an array of objects or None. """
        if os.path.isfile(infile):
            objects = []
            with open(file=infile, encoding='utf-8', mode='rt') as csvfile:
                reader = csv.reader(csvfile, delimiter=delim)
                headers = None
                for idx, row in enumerate(reader):
                    if idx == 0:
                        headers = row
                    else:
                        if len(row) == len(headers):
                            obj = {}
                            for field_idx, field_name in enumerate(headers):
                                key = field_name.strip().lower()
                                obj[key] = row[field_idx].strip()
                            objects.append(obj)
            return objects
        return None
# ==============================================================================

class Mongo():
    """
    This class is used to access a MongoDB database, including the CosmosDB
    Mongo API - RU model or vCore.
    """
    def __init__(self, opts: dict):
        self._opts = opts
        self._db = None
        self._coll = None
        if 'conn_string' in self._opts.keys():
            if 'cosmos.azure.com' in opts['conn_string']:
                self._env = 'cosmos'
            else:
                self._env = 'mongo'
            self._client = MongoClient(opts['conn_string'], tlsCAFile=certifi.where())
        else:
            if 'cosmos.azure.com' in opts['host']:
                self._env = 'cosmos'
            else:
                self._env = 'mongo'
            self._client = MongoClient(opts['host'], opts['port'], tlsCAFile=certifi.where())

        if self.is_verbose():
            print(json.dumps(self._opts, sort_keys=False, indent=2))

    def is_verbose(self) -> bool:
        """ Return True if the verbose option is set. """
        if 'verbose' in self._opts.keys():
            return self._opts['verbose']
        return False

    def list_databases(self) -> list[str]:
        """ Return the list of database names in the account. """
        try:
            return sorted(self._client.list_database_names())
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())
            return None

    def create_database(self, dbname):
        """ Create a database with the given name."""
        return self._client[dbname]

    def delete_database(self, dbname):
        """ Delete a database with the given name."""
        if dbname in 'admin,local,config'.split(','):
            return
        self._client.drop_database(dbname)

    def delete_container(self, cname):
        """ Delete a container with the given name."""
        self._db.drop_collection(cname)

    def list_collections(self):
        """ Return the list of collection names in the current database. """
        return self._db.list_collection_names(filter={'type': 'collection'})

    def set_db(self, dbname):
        """ Set the current database to the given name. """
        self._db = self._client[dbname]
        return self._db

    def set_coll(self, collname):
        """ Set the current collection to the given name. """
        try:
            self._coll = self._db[collname]
            return self._coll
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())
            return None

    def command_db_stats(self):
        """ Execute the 'dbstats' command and return the results."""
        return self._db.command({'dbstats': 1})

    def command_coll_stats(self, cname):
        """ Execute the 'collStats' command and return the results."""
        return self._db.command("collStats", cname)

    def command_list_commands(self):
        """ Execute the 'listCommands' command and return the results."""
        return self._db.command('listCommands')

    def command_sharding_status(self):
        """ Execute the 'printShardingStatus' command and return the results."""
        return self._db.command('printShardingStatus')

    def get_shards(self):
        """ Return the list of shards in the cluster per the config database. """
        self.set_db('config')
        return self._db.shards.find()

    def extension_command_get_database(self):
        """ Execute the 'getDatabase' command and return the results."""
        command = {}
        command['customAction'] = 'GetDatabase'
        return self._db.command(command)

    def get_shard_info(self) -> dict:
        """ Return a dict of shard info. """
        shard_dict = {}
        for shard in self._client.config.shards.find():
            shard_name = shard.get("_id")
            shard_dict[shard_name] = shard
        return shard_dict

    def create_coll(self, cname):
        """ Create a collection with the given name in the current database. """
        return self._db[cname]

    def get_coll_indexes(self, collname) -> list | None:
        """ Return the list of indexes for the given collection. """
        try:
            self.set_coll(collname)
            return self._coll.index_information()
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())
            return None

    # crud methods below, metadata methods above

    def insert_doc(self, doc):
        """ Insert a document into the current collection and return the result. """
        return self._coll.insert_one(doc)

    def find_one(self, query_spec):
        """
        Execute a find_one query in the current collection and return the result.
        """
        return self._coll.find_one(query_spec)

    def find(self, query_spec):
        """
        Execute a find query in the current collection and return the results.
        """
        return self._coll.find(query_spec)

    def find_by_id(self, id_str: str):
        """
        Execute a find_one query in the current collection, with the given id
        as a string, and return the results.
        """
        return self._coll.find_one({'_id': ObjectId(id_str)})

    def aggregate(self, pipeline):
        """ Execute an aggregation pipeline in the current collection and return the results. """
        # https://pymongo.readthedocs.io/en/stable/examples/aggregation.html
        # https://learn.microsoft.com/en-us/azure/cosmos-db/mongodb/vcore/vector-search
        return self._coll.aggregate(pipeline)

    def delete_by_id(self, id_str: str):
        """ Delete a document from the current collection by id and return the result. """
        return self._coll.delete_one({'_id': ObjectId(id_str)})

    def delete_one(self, query_spec):
        """ Delete a document from the current collection and return the result."""
        return self._coll.delete_one(query_spec)

    def delete_many(self, query_spec):
        """ Delete documents from the current collection and return the result."""
        return self._coll.delete_many(query_spec)

    def update_one(self, filter, update, upsert):
        """ Update a document in the current collection and return the result."""
        return self._coll.update_one(filter, update, upsert)

    def update_many(self, filter, update, upsert):
        """ Update documents in the current collection and return the result."""
        return self._coll.update_many(filter, update, upsert)

    def count_docs(self, query_spec):
        """
        Return the number of documents in the current collection
        that match the query spec.
        """
        return self._coll.count_documents(query_spec)

    def last_request_stats(self):
        """ Return the last request statistics (Cosmos DB)."""
        return self._db.command({'getLastRequestStatistics': 1})

    def last_request_request_charge(self):
        """ Return the last request charge in RUs (Cosmos DB)."""
        stats = self.last_request_stats()
        if stats is None:
            return -1
        return stats['RequestCharge']

    def client(self):
        """ Return the pymonto client object. """
        return self._client
# ==============================================================================

class OpenAIClient(object):
    """ This class is a REST and/or SDK client to Azure OpenAI. """

    def __init__(self, opts):
        self.opts = opts
        self.type = 'azure'  # default
        try:
            if 'type' in opts.keys():
                self.type = opts['type'].strip().lower()  # azure or openai
            if self.type == 'azure':
                openai.api_base    = opts['url']
                openai.api_key     = opts['key']
                openai.api_type    = 'azure'
                openai.api_version = '2023-05-15'  # '2022-06-01-preview' '2023-05-15'
                if 'api_version' in opts.keys():
                    openai.api_version = opts['api_version']
            else:
                openai.api_key = opts['key']
        except Exception as e:
            print(str(e))
            print(traceback.format_exc())

        self.headers = dict()
        self.headers['Content-Type'] = 'application/json'
        self.headers['api-key'] = openai.api_key

        # default embedding and encoding values

        self.encoding_model = 'cl100k_base'
        self.embedding_model = 'text-embedding-ada-002'
        self.embeddings_sleep_seconds = 0.20
        self.embeddings_pause_seconds = 150.0
        self.retry_count = 5

        # override default embedding and encoding values

        if 'encoding_model' in opts.keys():
            self.encoding_model = opts['encoding_model']
        if 'embedding_model' in opts.keys():
            self.embedding_model = opts['embedding_model']
        if 'embeddings_sleep_seconds' in opts.keys():
            self.embeddings_sleep_seconds = float(opts['embeddings_sleep_seconds'])
        if 'embeddings_pause_seconds' in opts.keys():
            self.embeddings_pause_seconds = float(opts['embeddings_pause_seconds'])
        if 'retry_count' in opts.keys():
            self.retry_count = int(opts['retry_count'])

        self.encoding = tiktoken.get_encoding(self.encoding_model)

    def get_config(self) -> dict:
        """ return a dict containing the config values for this client """
        config = {}
        config['type'] = self.type
        config['openai.api_base'] = openai.api_base
        config['openai.api_key']  = openai.api_key
        config['openai.api_version'] = openai.api_version
        config['openai.api_version'] = openai.api_version
        config['headers'] = self.headers
        config['opts'] = self.opts
        config['encoding_model'] = self.encoding_model
        config['embedding_model'] = self.embedding_model
        config['embeddings_sleep_seconds'] = self.embeddings_sleep_seconds
        config['embeddings_pause_seconds'] = self.embeddings_pause_seconds
        config['retry_count'] = self.retry_count
        return config

    def list_deployments(self) -> None:
        # https://learn.microsoft.com/en-us/rest/api/cognitiveservices/azureopenaipreview/deployments/list?tabs=HTTP
        if openai.api_base.endswith('/'):
            url = '{}/openai/deployments?api-version=2022-06-01-preview'.format(openai.api_base)
        else:
            url = '{}/openai/deployments?api-version=2022-06-01-preview'.format(openai.api_base)
        print(url)

        r = self.http_request('list_deployments', 'get', url, headers=self.headers, json_body={})
        deployments_list = r.json()['data']
        for deployment in deployments_list:
            model = deployment['model']
            id = deployment['id']
            created_at = deployment['created_at']
            print('model: {}, id: {}, created_at: {}'.format(model, id, created_at))

        # Output
        # model: code-davinci-002, id: code-davinci-002, created_at: 1687276047
        # model: gpt-35-turbo, id: gpt-35-turbo, created_at: 1687276076
        # model: text-ada-001, id: text-ada-001, created_at: 1687276112
        # model: text-davinci-003, id: text-davinci-003, created_at: 1687276138
        # model: text-embedding-ada-002, id: text-embedding-ada-002, created_at: 1687460972

    def get_embedding(self, text):
        # This method implements limited retry logic with linear backoff
        # to handle possible OpenAI API rate limiting.
        for n in range(self.retry_count):
            #print('get_embedding for text: {}'.format(text))
            e = self.try_get_embedding(text)
            if e != None:
                return e
        raise Exception('unable to get embedding for text: {}'.format(text))

    def try_get_embedding(self, text):
        # See https://platform.openai.com/docs/guides/embeddings/what-are-embeddings
        #   "We recommend using text-embedding-ada-002 for nearly all use cases. It’s better, cheaper, and simpler to use"
        #   MODEL NAME              TOKENIZER    MAX INPUT TOKENS   OUTPUT DIMENSIONS
        #   text-embedding-ada-002  cl100k_base  8191               1536
        try:
            time.sleep(self.embeddings_sleep_seconds)
            text = text.replace("\n", ' ')
            e = openai.Embedding.create(input=[text], engine=self.embedding_model)
            return e['data'][0]['embedding']  # returns a list of 1536 floats!
        except Exception as e:
            print("try_get_embedding exception: {}".format(str(e)))
            traceback.print_exc()
            self.embeddings_sleep_seconds = (self.embeddings_sleep_seconds) * 1.5
            print('new embeddings_sleep_seconds is {}'.format(self.embeddings_sleep_seconds))
            print('pausing for {} seconds'.format(self.embeddings_pause_seconds))
            time.sleep(self.embeddings_pause_seconds)
            return None

    def get_token_count(self, text: str) -> int:
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            print("exception: {}".format(str(e)))
            traceback.print_exc()
            return -1

    def generate(self, deployment_name: str, prompt: str, max_tokens: int) -> object | None:
        try:
            return self.get_openai_response(deployment_name, prompt, int(max_tokens))
        except Exception as e:
            print("exception: {}".format(str(e)))
            traceback.print_exc()

    def get_openai_response(self, deployment_name: str, prompt: str, max_tokens) -> OpenAIObject:
        #text = response['choices'][0]['text'].replace('\n', '').replace(' .', '.').strip()
        data = {}
        try:
            data['deployment_name'] = deployment_name
            data['prompt'] = prompt
            data['max_tokens'] = max_tokens
            data['response'] = openai.Completion.create(
                engine=deployment_name, prompt=prompt, max_tokens=max_tokens)
        except Exception as e:
            data['exception'] = str(e)
            traceback.print_exc()
        return data

    def http_request(self, function_name, method, url, headers={}, json_body={}):
        # This is a generic method which invokes ALL HTTP Requests to the Azure Search Service
        print('===')
        print("http_request: {} {} {}\nheaders: {}\nbody: {}".format(
            function_name, method.upper(), url, headers, json_body))

        print("http_request name/method/url: {} {} {}".format(
            function_name, method.upper(), url))
        print("http_request headers:\n{}".format(json.dumps(
            headers, sort_keys=False, indent=2)))
        print("http_request body:\n{}".format(json.dumps(
            json_body, sort_keys=False, indent=2)))

        print('---')
        r = None
        if method == 'get':
            r = requests.get(url=url, headers=headers)
        elif method == 'post':
            r = requests.post(url=url, headers=headers, json=json_body)
        elif method == 'put':
            r = requests.put(url=url, headers=headers, json=json_body)
        elif method == 'delete':
            r = requests.delete(url=url, headers=headers)
        else:
            print('error; unexpected method value passed to invoke: {}'.format(method))

        print('response.status_code: {}'.format(r.status_code))

        if r.status_code < 300:
            try:
                # Save the request and response data as a json file in tmp/
                outfile  = 'tmp/{}_{}.json'.format(function_name, int(self.epoch()))
                data = dict()
                data['function_name'] = function_name
                data['method'] = method
                data['url'] = url
                data['body'] = json_body
                data['filename'] = outfile
                data['resp_status_code'] = r.status_code
                try:
                    data['resp_obj'] = r.json()
                except:
                    pass # this is expected as some requests don't return a response, like http 204
                self.write_json_file(data, outfile)
            except Exception as e:
                print("exception saving http response".format(e))
                print(traceback.format_exc())
        else:
            print(r.text)
        return r

    def epoch(self):
        return int(time.time())

    def write_json_file(self, obj, outfile):
        with open(outfile, 'wt') as f:
            f.write(json.dumps(obj, sort_keys=False, indent=2))
            print('file written: {}'.format(outfile))
# ==============================================================================

class Storage():
    """
    This class is used to access an Azure Storage account.
    """
    def __init__(self, opts):
        acct_name = opts['acct']
        acct_key  = opts['key']
        acct_url  = f'https://{acct_name}.blob.core.windows.net/'
        self.blob_service_client = BlobServiceClient(
            account_url=acct_url, credential=acct_key)

    def account_info(self):
        """ Return the account information. """
        return self.blob_service_client.get_account_information()

    def list_containers(self):
        """ Return the list of container names in the account. """
        clist = []
        try:
            containers = self.blob_service_client.list_containers(include_metadata=True)
            for container in containers:
                clist.append(container)
            return clist
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())
            return None

    def create_container(self, cname):
        """ Create a container in the account. """
        try:
            container_client = self.blob_service_client.get_container_client(cname)
            container_client.create_container()
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())

    def delete_container(self, cname):
        """ Delete a container in the account. """
        try:
            container_client = self.blob_service_client.get_container_client(cname)
            container_client.delete_container()
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())

    def list_container(self, cname):
        """ Return the list of blobs in the container. """
        try:
            container_client = self.blob_service_client.get_container_client(cname)
            return container_client.list_blobs()
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())
            return None

    def upload_blob_from_file(self, local_file_path, cname, blob_name, overwrite=True):
        """ Upload a blob from a local file. """
        try:
            blob_client = self.blob_service_client.get_blob_client(container=cname, blob=blob_name)
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=overwrite)
            return True
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())
            return False

    def upload_blob_from_string(self, string_data, cname, blob_name, overwrite=True):
        """ Upload a blob from a string. """
        try:
            blob_client = self.blob_service_client.get_blob_client(container=cname, blob=blob_name)
            blob_client.upload_blob(string_data, overwrite=overwrite)
            return True
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())
            return False

    def download_blob(self, cname, blob_name, local_file_path):
        """ Download a blob to a local file. """
        try:
            blob_client = self.blob_service_client.get_blob_client(container=cname, blob=blob_name)
            with open(local_file_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())

    def download_blob_to_string(self, cname: str, blob_name: str) -> str:
        """ Download a blob to a string. """
        blob_client = self.blob_service_client.get_blob_client(container=cname, blob=blob_name)
        downloader = blob_client.download_blob(max_concurrency=1, encoding='UTF-8')
        return downloader.readall()
# ==============================================================================

class StringUtil():
    """
    This class is used to do common string operations.
    """
    @classmethod
    def levenshtein_distance(cls, str1: str, str2: str) -> int:
        """ Return the levenshtein distance between the two given strings. """
        return distance(str(str1), str(str2))
# ==============================================================================

class System():
    """
    This class is an interface to system information such as memory usage.
    """

    @classmethod
    def command_line_args(cls) -> list[str]:
        """ Return sys.argv """
        return sys.argv

    @classmethod
    def platform(cls) -> str:
        """ Return the platform.system() string. """
        return platform.system()

    @classmethod
    def is_windows(cls) -> bool:
        """ Return True if the platform is Windows, else False. """
        return 'win' in cls.platform().lower()

    @classmethod
    def is_mac(cls) -> bool:
        """ Return True if the platform is Apple macOS, else False. """
        return 'darwin' in cls.platform().lower()

    @classmethod
    def pid(cls) -> int:
        """ Return the current process id int. """
        return os.getpid()

    @classmethod
    def process_name(cls) -> str:
        """ Return the current process name. """
        return psutil.Process().name()

    @classmethod
    def user(cls) -> str:
        """ Return the current user name; os.getlogin(). """
        return os.getlogin()

    @classmethod
    def hostname(cls) -> str:
        """ Return the current hostname; socket.gethostname(). """
        try:
            return socket.gethostname()
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())
            return 'unknown'

    @classmethod
    def cwd(cls) -> str:
        """ Return the current working directory; Process.cwd(). """
        return psutil.Process().cwd()

    @classmethod
    def pwd(cls) -> str:
        """ Return the current working directory; os.getcwd() """
        return os.getcwd()

    @classmethod
    def platform_info(cls) -> str:
        """ Return a string with the platform info including processor. """
        return f'{platform.platform()} : {platform.processor()}'

    @classmethod
    def cpu_count(cls) -> int:
        """ Return the number of CPUs on the system. """
        return psutil.cpu_count(logical=False)

    @classmethod
    def memory_info(cls):
        """ Return the memory info for the current process. """
        return psutil.Process().memory_info()

    @classmethod
    def virtual_memory(cls):
        """ Return the virtual memory info for the current process. """
        return psutil.virtual_memory()

    @classmethod
    def epoch(cls) -> float:
        """ Return the current epoch time in seconds, as a float """
        return time.time()

    @classmethod
    def sleep(cls, seconds=1.0) -> None:
        """ Sleep for the given number of float seconds. """
        time.sleep(seconds)
