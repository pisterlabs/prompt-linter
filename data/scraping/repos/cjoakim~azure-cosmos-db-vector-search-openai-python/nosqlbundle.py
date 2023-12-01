"""
Module nosqlbundle.py - bundled standard codebase.
Copyright (c) 2023 Chris Joakim, MIT License
Timestamp: 2023-07-30 13:36

Usage:  from pysrc.nosqlbundle import Bytes, Cosmos, Counter, Env, FS, Mongo, OpenAIClient, RCache, Storage, System, Template
"""

import csv
import json
import os
import platform
import socket
import sys
import time
import traceback

import certifi
import jinja2
import matplotlib
import openai
import pandas as pd
import psutil
import redis
import requests
import tiktoken

from numbers import Number
from typing import Iterator

from azure.cosmos import cosmos_client
from azure.cosmos import diagnostics
from azure.cosmos import exceptions
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

class Cosmos():
    """
    This class is used to access a Cosmos DB NoSQL API account.
    """
    def __init__(self, opts):
        self._dbname = None
        self._dbproxy = None
        self._ctrproxy = None
        self._cname = None
        self.reset_record_diagnostics()
        url = opts['url']
        key = opts['key']
        if 'enable_query_metrics' in opts.keys():
            self._query_metrics = True
        else:
            self._query_metrics = False
        self._client = cosmos_client.CosmosClient(url, {'masterKey': key})

    def list_databases(self):
        """ Return the list of database names in the account. """
        self.reset_record_diagnostics()
        return list(self._client.list_databases())

    def set_db(self, dbname):
        """ Set the current database to the given dbname. """
        try:
            self.reset_record_diagnostics()
            self._dbname = dbname
            self._dbproxy = self._client.get_database_client(database=dbname)
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())

        return self._dbproxy
        # <class 'azure.cosmos.database.DatabaseProxy'>

    def list_containers(self):
        """ Return the list of container names in the current database. """
        self.reset_record_diagnostics()
        return list(self._dbproxy.list_containers())

    def create_container(self, cname, partition_key, throughput):
        """ Create a container in the current database. """
        try:
            self.reset_record_diagnostics()
            self._ctrproxy = self._dbproxy.create_container(
                id=cname,
                partition_key=partition_key.PartitionKey(path=partition_key),
                offer_throughput=throughput,
                populate_query_metrics=self._query_metrics,
                response_hook=self._record_diagnostics)
            return self._ctrproxy
            # <class 'azure.cosmos.container.ContainerProxy'>
        except exceptions.CosmosResourceExistsError as excp:
            print(str(excp))
            print(traceback.format_exc())
            return self.set_container(cname)
        except Exception as excp2:
            print(str(excp2))
            print(traceback.format_exc())
            return None

    def set_container(self, cname):
        """ Set the current container in the current database to the given cname. """
        self.reset_record_diagnostics()
        self._ctrproxy = self._dbproxy.get_container_client(cname)
        # <class 'azure.cosmos.container.ContainerProxy'>
        return self._ctrproxy

    def update_container_throughput(self, cname, throughput):
        """ Update the throughput of the given container. """
        self.reset_record_diagnostics()
        self.set_container(cname)
        offer = self._ctrproxy.replace_throughput(
            throughput=int(throughput),
            response_hook=self._record_diagnostics)
        return offer

    def get_container_offer(self, cname):
        """ Get the current offer (throughput) for the given container. """
        self.reset_record_diagnostics()
        self.set_container(cname)
        offer = self._ctrproxy.read_offer(
            response_hook=self._record_diagnostics)
        # <class 'azure.cosmos.offer.Offer'>
        return offer

    def delete_container(self, cname):
        """ Delete the given container name in the current database. """
        try:
            self.reset_record_diagnostics()
            return self._dbproxy.delete_container(
                cname,
                populate_query_metrics=self._query_metrics,
                response_hook=self._record_diagnostics)
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())
            return None

    def upsert_doc(self, doc):
        """ Upsert the given document in the current container. """
        try:
            self.reset_record_diagnostics()
            return self._ctrproxy.upsert_item(
                doc,
                populate_query_metrics=self._query_metrics,
                response_hook=self._record_diagnostics)
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())
            return None

    def delete_doc(self, doc, doc_pk):
        """ Delete the given document in the current container. """
        try:
            self.reset_record_diagnostics()
            return self._ctrproxy.delete_item(
                doc,
                partition_key=doc_pk,
                populate_query_metrics=self._query_metrics,
                response_hook=self._record_diagnostics)
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())
            return None

    def read_doc(self, cname, doc_id, doc_pk):
        """ Execute a point-read for container, document id, and partition key. """
        try:
            self.set_container(cname)
            self.reset_record_diagnostics()
            return self._ctrproxy.read_item(
                doc_id,
                partition_key=doc_pk,
                populate_query_metrics=self._query_metrics,
                response_hook=self._record_diagnostics)
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())
            return None

    def query_container(self, cname, sql, xpartition, max_count):
        """ Execute a given SQL query of the given container name. """
        try:
            self.set_container(cname)
            self.reset_record_diagnostics()
            return self._ctrproxy.query_items(
                query=sql,
                enable_cross_partition_query=xpartition,
                max_item_count=max_count,
                populate_query_metrics=self._query_metrics,
                response_hook=self._record_diagnostics)
        except Exception as excp:
            print(str(excp))
            print(traceback.format_exc())
            return excp

    # Metrics and Diagnostics

    def enable_query_metrics(self):
        """ Return a boolean indicating whether query metrics are enabled. """
        self._query_metrics = True

    def disable_query_metrics(self):
        """ Set query metrics to False. """
        self._query_metrics = False

    def reset_record_diagnostics(self):
        """ Reset the record diagnostics in this object. """
        self._record_diagnostics = diagnostics.RecordDiagnostics()

    def print_record_diagnostics(self):
        """ Print the record diagnostics. """
        print(f'record_diagnostics: {self._record_diagnostics.headers}')
        print(str(type(self._record_diagnostics.headers)))
        keys = self._record_diagnostics.headers.keys()
        print(str(type(keys)))
        print(keys)
        for header in self._record_diagnostics.headers.items():
            print(header)
            print(str(type(header)))

    def record_diagnostics_headers_dict(self):
        """ Read and return the record diagnostics headers as a dictionary. """
        data = {}
        for header in self._record_diagnostics.headers.items():
            key, val = header  # unpack the header 2-tuple
            data[key] = val
        return data

    def print_last_request_charge(self):
        """ Print the last request charge and activity id. """
        charge = self.last_request_charge()
        activity = self.last_activity_id()
        print(f'last_request_charge: {charge} activity: {activity}')

    def last_request_charge(self):
        """ Return the last request charge in RUs, default to -1. """
        header = 'x-ms-request-charge'
        if header in self._record_diagnostics.headers:
            return self._record_diagnostics.headers[header]
        return -1

    def last_activity_id(self):
        """ Return the last diagnostics activity id, default to None. """
        header = 'x-ms-activity-id'
        if header in self._record_diagnostics.headers:
            return self._record_diagnostics.headers[header]
        return None
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
                openai.api_version = '2022-06-01-preview'  # '2023-05-15'
                if 'version' in opts.keys():
                    openai.api_version = opts['version']
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

class RCache():
    """
    This class is used to access either a local Redis server, or Azure Cache
    for Redis.
    """
    def __init__(self, host, port):
        self.redis_client = redis.Redis(host=host, port=port)

    def set(self, key:str, value):
        """ Set the given cache key to the given value. """
        return self.redis_client.set(key, value)

    def get(self, key: str):
        """ Get the cache value for the given cache key. """
        return self.redis_client.get(key)

    def client(self):
        """ Return the redis.Redis client object. """
        return self.redis_client
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
# ==============================================================================

class Template():
    """
    This class is used to create text content using jinja2 templates.
    """
    @classmethod
    def get_template(cls, root_dir: str, name):
        """
        Return a jinja2 template object for the given filename
        in the templates/ directory. """
        filename = f'templates/{name}'
        return cls._get_jinja2_env(root_dir).get_template(filename)

    @classmethod
    def render(cls, template, values: dict) -> str:
        """ Render the given template object with the given dict of values. """
        return template.render(values)

    @classmethod
    def _get_jinja2_env(cls, root_dir: str):
        """
        Private method to return a jinja2 Environment object for the
        given root_dir.
        """
        return jinja2.Environment(
            loader = jinja2.FileSystemLoader(
                root_dir), autoescape=True)
