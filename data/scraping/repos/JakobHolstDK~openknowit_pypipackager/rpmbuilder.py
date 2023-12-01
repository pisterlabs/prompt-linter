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


def fix_indentation(filename):
    # Use the filename variable for further processing
    with open(filename, 'r') as file:
        content = file.read()
    fixed_content = ""
    for line in content.splitlines():
        heading = True
        spacecount = 0
        newline = ''
        for char in line:
            if char == ' ' and heading:
                spacecount = spacecount + 1
            else:   
                heading = False
                print("Found non-space")
                newline = newline + char
        fixed_content = fixed_content + newline + '\n'
    with open(filename, 'w') as file:
        file.write(fixed_content)

def create_default_pyproject_toml(name, version):
    pyproject = {
        'tool': {
            'poetry': {
                'name': name,
                'version': version,
                'description': '',
                'authors': [],
                'license': 'MIT',
                'dependencies': {},
                'dev-dependencies': {}
            }
        }
    }
    return (pyproject)








def hotfixmysource(name, version):
    pypi_packages = db['pypi_packages']
    query = {'name': name, 'version': version}
    if os.path.exists('setup.py.hotfixed'):
      with open('setup.py.hotfixed', "r") as f:
            content = f.read()
      hotfix = {'filename': 'setup.py', 'content': content}
      open('setup.py', 'w').write(content)
      pypi_packages.update_one(query, {'$set': {'setuppyhotfix': True, 'hotfix': hotfix}})
      return True
    else:
        query = {'name': name, 'version': version, 'setuppyhotfix': True  }
        if pypi_packages.find_one(query):
          package = pypi_packages.find_one(query)
          hotfix = package['hotfix']
          open(hotfix['filename'], 'w').write(hotfix['content'])
          return True
        else:
          return False  
        

    

def createsetuppyfrompyprojecttoml(name, version):
  download_folder = os.getenv('DOWNLOAD_FOLDER', '/tmp')
  source_folder = download_folder + name + '-' + version
  pyprojecttoml_file = source_folder + '/' + 'pyproject.toml'
  setuppy_file = source_folder + '/' + 'setup.py'
  if os.path.exists(pyprojecttoml_file):
    with open(pyprojecttoml_file, "r") as f:
      config = toml.load(f)
    try: 
      if config["tool"]["poetry"]["name"] == name:
        poetry = True
    except:
        poetry = False

    if  poetry:
      try:
        project_name = config["tool"]["poetry"]["name"]
      except:
        project_name = name
      try:
        project_version = config["tool"]["poetry"]["version"]
      except:
        project_version = version
      try:

        project_description = config["tool"]["poetry"]["description"]
      except:
        project_description = ''
      try:
        project_license = config["tool"]["poetry"]["license"]
      except:
        project_license = ''
      try:
        project_url = config["tool"]["poetry"]["homepage"]
      except:
        project_url = ''
      try:
        project_authors = config["tool"]["poetry"]["authors"] 
      except:
        project_authors = ''
    try: 
      if config['project']['name'] == name:
        project = True
    except:
        project = False

    if  project:
        project_name = config['project']['name']
        if 'version' in config['project']:
          project_version = config['project']['version']
        else:
          project_version = version

        if 'description' in config['project']:
          project_description = config['project']['description']
        else:
          project_description = ''

        if 'license' in config['project']:
          project_license = config['project']['license']
        else:
          project_license = ''

        if 'project_url' in config['project']:
           project_url = config['project']['project_url']
        else:
           project_url = ''

        if 'project_authors' in config['project']:
          project_authors = config['project']['project_authors']
        else:
          project_authors = ''
        project_authors = config['project']['authors']

    if not project and not poetry:
      print("The name in the pyproject.toml has no project or poetry entry with the same name as the package")
      return False
    

    # Generate the setup.py file
    with open(setuppy_file, "w") as f:
      f.write(f"""

from setuptools import setup, find_packages
  setup(
    name="{project_name}",
    version="{project_version}",
    description="{project_description}",
    license="{project_license}",
    url="{project_url}",
    author="{project_authors}",
    packages=find_packages(),
  )
    """)

def convert_setup_cfg_to_setup_py(input_file, output_file):
    with open(input_file, 'r') as cfg_file:
        cfg_content = cfg_file.read()

    py_content = '''
from setuptools import setup

setup(
''' + cfg_content + ')'

    with open(output_file, 'w') as py_file:
        py_file.write(py_content)



def replace_setupcfg_with_pyprojecttoml(setupcfg_file, pyprojecttoml_file, name, version):
# Read the contents of setup.cfg
  config = configparser.ConfigParser()
  config.read(setupcfg_file)
# Extract the relevant fields from setup.cfg
  pyproject = create_default_pyproject_toml(name, version)
  try:
    description = config["metadata"]["description"]
  except:
    description = ''
  try:
    url = config["metadata"]["url"]
  except:
    url = ''
  try:
    author = config["metadata"]["author"]
  except:
    author = ''

  try:
    author_email = config["metadata"]["author_email"]
  except:
    author_email = ''

  try:  
    license = config["metadata"]["license"]
  except:
    license = ''
  try:
    classifiers = config["metadata"]["classifiers"].split("\n")
  except:
    classifiers = ''

  try:  
    requires_python = config["metadata"]["requires_python"]
  except:
    requires_python = ''

  if os.path.exists(pyprojecttoml_file):
    with open(pyprojecttoml_file, "r") as f:
      pyproject = toml.load(f)
  else:
    pyproject = create_default_pyproject_toml(name, version)
  pyproject['tool'] = {}
  pyproject['tool']['poetry'] = {}
  pyproject['tool']['poetry']["name"] = name
  try: 
    pyproject['tool']['poetry']["version"] = version
  except:
    pyproject['tool']['poetry']["Version"] = version
  try:
    pyproject['tool']['poetry']["description"] = description
  except:
    pyproject['tool']['poetry']["description"] = ''
  try:
    pyproject['tool']['poetry']["homepage"] = url
  except:
    pyproject['tool']['poetry']["homepage"] = ''
  try:  
    pyproject['tool']['poetry']["authors"] = [f"{author} <{author_email}>"]
  except:
    pyproject['tool']['poetry']["authors"] = ''
  try:
    pyproject['tool']['poetry']["license"] = license
  except:
    pyproject['tool']['poetry']["license"] = ''
  try:
    pyproject['tool']['poetry']["classifiers"] = classifiers
  except:
    pyproject['tool']['poetry']["classifiers"] = ''
  with open(pyprojecttoml_file, "w") as f:
    toml.dump(pyproject, f)


def prettymysetuppy(name, version):
  download_folder = os.getenv('DOWNLOAD_FOLDER', '/tmp')
  source_folder = download_folder + name + '-' + version
  setuppy_file = source_folder + '/' + 'setup.py'
  setupcfg_file = source_folder + '/' + 'setup.cfg'
  prettysetuppy_file = source_folder + '/' + 'pretty.setup.py'
  pyprojecttoml_file = source_folder + '/' + 'pyproject.toml'

  if not os.path.exists(setuppy_file):
     if os.path.exists(pyprojecttoml_file):
       createsetuppyfrompyprojecttoml(name, version)
   
  setupfile = ""
  if os.path.exists(setuppy_file):
    with open(setuppy_file, 'r') as file:
      data = file.readlines()
      for line in data:
        if line == "   from":
          line = line.replace("   from", "from")
        line = line.replace("  setup(", "setup(")
        line = line.replace("   setup(", "setup(")
        setupfile += line

    prompt = PromptTemplate(
      input_variables=["setupfile"],
      template="Pretty this python setup-py file. the file has to have name : " + name + " and a version : " + version +"  : {setupfile}",
    )
    response = prompt.format(setupfile=setupfile)
    response = response.replace("Pretty this python setup-py file. the file has to have name : " + name + " and a version : " + version + "  : ", "")
    open(prettysetuppy_file, 'w').write(response)

    print(prettysetuppy_file)
    fix_indentation(prettysetuppy_file)


def registerpypipackage(name, version):
  package_data = {
            'name': name,
            'version': version,
            'rpmbuild': False,
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

def create_spec_file(name, version):
  download_folder = os.getenv('DOWNLOAD_FOLDER', '/tmp')
  source_folder = download_folder + name + '-' + version
  setuppyhotfix_file = source_folder + '/' + 'setup.py.hotfixed'
  if os.path.exists(setuppyhotfix_file):
    setup_file  = setuppyhotfix_file
    try:
      subprocess.call(["python3", "setup.py.hotfixed", "bdist_rpm", "--spec-only"], cwd=source_folder)
      return True
    except:
        print("Error creating spec file")
        return False


  spec_file = source_folder + '/' + name + '.spec'
  #setup.py bdist_rpm --spec-only
  if os.path.exists(spec_file):
      try:
        subprocess.call(["python3", "setup.py", "bdist_rpm", "--spec-only"], cwd=source_folder)
        return True
      except:
        print("Error creating spec file")

  else:
    print("No setup.py file found in source folder")
    setup_file  = download_folder + name + '-' + version + '/pretty.setup.py'
    if os.path.exists(setup_file):
      try:
        subprocess.call(["python3", "pretty.setup.py", "bdist_rpm", "--spec-only"], cwd=source_folder)
      except:
        print("Error creating spec file")
    else:
      print("No pretty.setup.py file found in source folder")
  


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
   print("HELLLLLP")

def unpack_whl_file(filename):
  download_folder = os.getenv('DOWNLOAD_FOLDER', '/tmp')
  subprocess.call(["wheel", "unpack", filename, '--dest' , download_folder])

def downloadpypipackage(name, version):
  download_folder = os.getenv('DOWNLOAD_FOLDER', '/tmp')
  package_name = name + '==' + version
  #subprocess.call(["pip", "download", "-d", download_folder, package_name])
  before = filenames(download_folder)
  subprocess.call(["pip", "download", '--no-binary' , ':all:',  "-d", download_folder, package_name])
  after = filenames(download_folder)
  diff = diflist(before, after)
  for i in diff:
    newpackage = i[::-1].split('-', 1)[1][::-1]
    newversion = i[::-1].split('-', 1)[0][::-1].replace('.tar.gz', '').replace('.whl', '').replace('.zip', '')

  for filename in os.listdir(download_folder):
        if filename.startswith(name + '-') and filename.endswith('.whl'):
            return os.path.join(download_folder, filename)
  for filename in os.listdir(download_folder):
        name = name.replace('-','_')
        if filename.startswith(name + '-') and filename.endswith('.whl'):
            return os.path.join(download_folder, filename)
  return None

query = {'sourcedownloaded': False}
packages = db['pypi_packages']
for package in packages.find(query):
    filename = downloadpypipackage(package['name'], package['version'])
    query = {'name': package['name'], 'version': package['version']}
    update = {'$set': {'sourcefile': filename}, '$set': {'sourcedownloaded': True}}
    packages.update_one(query, update)

download_folder = os.getenv('DOWNLOAD_FOLDER', '/tmp')
for file in filenames(download_folder):
  if file.endswith('.gz'):  
    if unpack_gz_file(file):
      query = {'sourceunpacked': True}
      update = {'$set': {'sourceunpacked': True}}
      packages.update_one(query, update)

  if file.endswith('.zip'):
    if unpack_zip_file(file):
      print("Unpacked zip file")
 
#setup.py does not exist but we have a setup.cfg file



query = {'prettysetuppy': False}
packages = db['pypi_packages']

for package in packages.find(query):
    hotfixmysource(package['name'],  package['version'])
    setupcfg_file = download_folder + package['name'] + '-' + package['version'] + '/setup.cfg'
    setuppy_file = download_folder + package['name'] + '-' + package['version'] + '/setup.py'
    pyprojecttoml_file = download_folder + package['name'] + '-' + package['version'] + '/pyproject.toml'
    if os.path.exists(setupcfg_file):
      if (not os.path.exists(setuppy_file)) and (not os.path.exists(pyprojecttoml_file)):
        convert_setup_cfg_to_setup_py(setupcfg_file, setuppy_file)

    #  convert_setup_cfg_to_setup_py(setupcfg_file, setuppy_file)


    prettymysetuppy(package['name'],  package['version'])
    if os.path.exists(download_folder + package['name'] + '-' + package['version'] + '/pretty.setup.py'):
      query = {'name': package['name'], 'version': package['version']}
      update = {'$set': {'prettysetuppy': True}}
      packages.update_one(query, update)

query = {'specfilecreated': False}
packages = db['pypi_packages']
for package in packages.find(query):
    create_spec_file(package['name'],  package['version'])
    if os.path.exists(download_folder + package['name'] + '-' + package['version'] + '/dist/' + package['name'] + '.spec'):
      specfile = open(download_folder + package['name'] + '-' + package['version'] + '/dist/' + package['name'] + '.spec', 'r').read()
      query = {'name': package['name'], 'version': package['version']}
      update = {'$set': {'specfile': specfile, 'specfilecreated': True}}
      packages.update_one(query, update)


    #query = {'name': package['name'], 'version': package['version']}
    #update = {'$set': {'rpmbuild': True}}
    #packages.update_one(query, update)
    #registerpypipackage(package['name'],  package['version'])

print("-----------------------------------------------------------------------------------------------------")    
query = {'specfilecreated': True}
packages = db['pypi_packages']
for package in packages.find(query):
  print(package['name'] + '-' + package['version'] + ' has a spec file')
print("-----------------------------------------------------------------------------------------------------")    
query = {'specfilecreated': False}
packages = db['pypi_packages']
for package in packages.find(query):
  print(package['name'] + '-' + package['version'] + ' has no spec file')
print("-----------------------------------------------------------------------------------------------------")    
