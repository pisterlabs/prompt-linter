# Driver program to create prototypes using ChatGPT4 LLM from defined unit tests

import os
import re
import subprocess
import json
import importlib.util
import openai
import hashlib
import logging
import sqlalchemy

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Text, ForeignKey

# set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

SETTINGS_PATH = "..\\settings.json"

def main():

    # read settings
    with open(SETTINGS_PATH, "r") as f:
        settings = json.load(f)
    
    # authenticate with openai
    logging.debug("Authenticating with OpenAI")
    openai_api_key = settings["openai_api_key"]
    openai.api_key = openai_api_key

    # setup database
    logging.debug("Setting up database")
    db_connection_str = settings['sqlalchemy_database_uri']
    setup_db(db_connection_str)

    # open database connection using sqlalchemy
    engine = sqlalchemy.create_engine(db_connection_str)
    metadata_obj = MetaData()
    conn = engine.connect()
    prototypes = Table('prototypes', metadata_obj, autoload_with=engine)

    # get directory of this file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    test_dir = os.path.join(dir_path, "unit_tests")

    # recursively walk all files in test_dir
    files = []
    for r, d, f in os.walk(test_dir):
        for file in f:
            if file[-7:] == 'test.py':
                files.append(os.path.join(r, file))

    # process each file
    for file in files:
        logging.info("Processing {}".format(file))

        # step 1: get the prompt and programming language from the file
        user_prompt = import_var_from_module(file, "PROMPT")
        language = import_var_from_module(file, "LANGUAGE")
        code_name = import_var_from_module(file, "CODE_NAME")

        # step 2: run the prompt through the LLM
        system_prompt = f""" 
            You are a computer programmer. 
            Only return code in the {language} programming language. Do not include any text outside of the code block.
            All code will be compiled and run on a Windows operating system using the gcc compiler.
            Do not use d_type anywhere in your code as 'struct dirent' has no member named 'd_type' in Windows.
            One or more functions along with required imports and global variables should be returned depending on the user input. 
            """
        system_prompt = " ".join(system_prompt.split())
      
        logging.info("Running prompt through LLM")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            n = 20,
            messages=[
                { "role": "system", "content": system_prompt },
                {"role": "user", "content": user_prompt}
            ]
        )

        # step 3: store the returned code in the database
        for choice in response['choices']:
            
            content  = choice['message']['content']

            # look for the code block
            pattern = r'```\S*(.*?)```'
            matches = re.findall(pattern, content, re.DOTALL)
            if len(matches) > 0:
                content = matches[0].strip()

            # find the sha256 hash of the content
            hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
                       
            # insert into database using sqlalchemy
            query = prototypes.insert().values(hash=hash, name=code_name, prompt=user_prompt, language=language, code=content, status=0, num_errors=0)
            conn.execute(query)
            conn.commit()
            

            # step 4: write the code to a file and compile it using subprocess popen
            with open("proto.c", "w") as f:
                f.write(content)

            # compile the code            
            p = subprocess.Popen(['gcc.exe', '-shared', '-o', 'proto.dll', '-fPIC', 'proto.c'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            (out, err) = p.communicate()
            if p.returncode != 0:
                logging.error(f"Error compiling code in ")
                out = out.decode("utf-8")
                err = err.decode("utf-8")
                logging.error(out)
                logging.error(err)

                # update database with error code (1)
                query = prototypes.update().where(prototypes.c.hash==hash).values(status=1)
                conn.execute(query)
                conn.commit()

                continue

            # step 5: run the unit tests
            logging.info("Running unit tests in {}".format(file))
            p = subprocess.Popen(['python', file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            (out, _) = p.communicate()
            
            # check for errors in subprocess, if so, log filename and continue
            if p.returncode != 0:
                logging.error(f"Error in unit test {file} - {hash}")
                logging.error(out)
                # update database with error code (1)
                query = prototypes.update().where(prototypes.c.hash==hash).values(status=1)
                conn.execute(query)
                conn.commit()
                continue

            # Decode output as json
            try: 
                out = json.loads(out)
                num_failures = out["num_failures"]
                num_errors = out["num_errors"]
                num_tests = out["num_tests"]
            except:
                logging.error(f"Error decoding json in unit test {file} - {hash}")
                # update database with error code (1)
                query = prototypes.update().where(prototypes.c.hash==hash).values(status=1)
                conn.execute(query)
                conn.commit()
                continue

            # log output
            logging.info("num_failures: {}".format(num_failures))
            logging.info("num_errors: {}".format(num_errors))
            logging.info("num_tests: {}".format(num_tests))

            # step 6: update the database with the number of failures and errors
            total_errors = num_failures + num_errors
            query = prototypes.update().where(prototypes.c.hash==hash).values(num_errors=total_errors, status=2 if total_errors > 0 else 3)
            conn.execute(query)
            conn.commit()
           
    # close database connection
    conn.close()
    engine.dispose()

def setup_db(db_connection_str):
    """
    Creates the database if it does not exist
    """
    # open database connection using sqlalchemy
    engine = sqlalchemy.create_engine(db_connection_str)
    # Create the Metadata Object 
    metadata_obj = MetaData() 
    
    # Define the profile table 
    
    # database name 
    prototypes = Table( 
        'prototypes',                                         
        metadata_obj,                                     
        Column('hash', String, primary_key=True),   
        Column('name', String),
        Column('prompt', Text),
        Column('language', String),
        Column('code', Text),
        Column('status', Integer),
        Column('num_errors', Integer),               
    ) 
  
    # Create the profile table 
    metadata_obj.create_all(engine) 

    engine.dispose()

def import_var_from_module(path, var):
    """
    Imports a variable from a module given the path to the module and the variable name
    """
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, var, None)

if __name__ == "__main__":
    main()
