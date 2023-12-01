from pymongo import MongoClient
import configparser
import re
import subprocess
import os
import requests
import sys
import toml 
from langchain.prompts import PromptTemplate
import json
from flask import Flask, request, jsonify


API_URL = os.getenv("PYPIAPI")
MONGO_URI = os.getenv("MONGO")
AITOKEN = os.getenv("OPENAI_API_KEY")

# Create a MongoDB client
client = MongoClient(MONGO_URI)
db = client['pypi-packages']


def registerpypipackage(name, version, dependency=False):
  package_data = {
            'name': name,
            'version': version,
            'status': "initial", 
            'dependency': dependency,
            'hotfix': { 'filename': "", 'content': ""},
            'setuppy': False,
            'setuppyhotfix': False,
            'setupcfg': False,
            'prettysetuppy': False,
            'pyprojecttoml': False,
            'sourcedownloaded': False,
            'sourceunpacked': False,
            'specfile': "",
            'specfilecreated': False,
            'rpmbuild': False,
            'rpmfilepublished': False,
            'debbuild': False
        }
  response = requests.post(API_URL, json=package_data)
  if response.status_code == 200:
    print(f'Registered package: {name}')
  else:
    print(f'Error registering package {name}: {response.text}')

def filenames(mypath):
  file_names = []
  files = os.listdir(mypath)
  for file in files:
    file_names.append(file)
  return file_names

def diflist(list1 , list2):
  list_dif = [i for i in list1 + list2 if i not in list1 or i not in list2]
  return list_dif

def unpack_gz_file(filename):
  download_folder = os.getenv('DOWNLOAD_FOLDER', '/tmp')
  runme = subprocess.call(["tar", "-xzf", download_folder + filename, '-C' , download_folder])
  if runme == 0:
    return True
  else:
    print("File not unpacked")
    return False

def unpack_zip_file(filename):
  download_folder = os.getenv('DOWNLOAD_FOLDER', '/tmp')
  runme = subprocess.call(["unzip", download_folder + filename, '-d' , download_folder])
  if runme == 0:
    return True
  else:
    print("File not unpacked")
    return False


def unpack_whl_file(filename):
  download_folder = os.getenv('DOWNLOAD_FOLDER', '/tmp')
  runme = subprocess.call(["wheel", "unpack", filename, '--dest' , download_folder])
  if runme == 0:
    return True
  else:
    print("File not unpacked")
    return False
      

if __name__ == "__main__":
  query = {'sourcedownloaded': True, 'sourceunpacked': False}
  packages = db['pypi_packages']
  download_folder = os.getenv('DOWNLOAD_FOLDER', '/tmp')
  before = filenames(download_folder)
  for file in filenames(download_folder):
    name = file.replace('.tar.gz', '')[::-1].split('-', 1)[-1][::-1]
    version = file.replace('.tar.gz', '')[::-1].split('-', 1)[0][::-1]
    query = {'name': name, 'version': version}
    package = packages.find_one(query)
    if package:
      print("Package already exists")
      print(package['sourceunpacked'])
      if package['sourceunpacked'] == False:
        print("Unpacking source file")
        if file.endswith('.gz'):  
          if unpack_gz_file(file):
            print("Unpacked gz file")
            query = {'name': name, 'version': version}
            update = {'$set': {'status': 'sourceunpacked', 'sourceunpacked': True}}
            packages.update_one(query, update)
            after = filenames(download_folder)
            dif = diflist(before, after)
            filesadded = []
            for file in dif:
              filesadded.append(file)
            update = {'$set': {'status': 'sourceunpacked', 'sourceunpacked': True, 'sourcepath': filesadded}}
            packages.update_one(query, update)



        if file.endswith('.zip'):
          if unpack_zip_file(file):
            print("Unpacked zip file")
            query = {'name': name, 'version': version}
            update = {'$set': {'status': 'sourceunpacked', 'sourceunpacked': True}}
            packages.update_one(query, update)
        continue
    else:
      print("Registering package")
      registerpypipackage(name, version, dependency=True)
      print("Unpacking source file")

      if file.endswith('.gz'):  
        if unpack_gz_file(file):
          print("Unpacked gz file")
          query = {'name': name, 'version': version}
          update = {'$set': {'status': 'sourceunpacked', 'sourceunpacked': True }}
          packages.update_one(query, update)

      if file.endswith('.zip'):
        if unpack_zip_file(file):
          print("Unpacked zip file")
          query = {'name': name, 'version': version}
          update = {'$set': {'status': 'sourceunpacked', 'sourceunpacked': True }}
          packages.update_one(query, update)

 
