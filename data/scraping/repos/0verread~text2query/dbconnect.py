import os
import uuid
import json
import openai
import datetime
import string, random

import MySQLdb as mysqldb
import psycopg2
import bcrypt

from google.cloud import storage


from dotenv import load_dotenv

load_dotenv()

storage_client = storage.Client()
BUCKET = storage_client.bucket('orgateai.appspot.com')

# Make API call to OPENAI
def makeit(table_schema, prompt):
  final_prompt= r"### Postgres SQL tables, with their properties:\n#\n# " + table_schema + r"\n#\n### " +  prompt + r" \nSELECT"
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt = final_prompt,
    temperature=0.5,
    max_tokens=100,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=["#", ";"]
  )
  return response.choices[0].text


# Create prompt
def get_prompt(schema):
    table_strings = []
    table_schema = schema.get('schema')
    for table_name, table_info in table_schema.items():
        columns = [f"{col_name}" for col_name, col_type in table_info.items()]
        table_string = f"{table_name}({', '.join(columns)})"
        table_strings.append(table_string)
    schema_part = r'\n# '.join(table_strings)
    return schema_part
# --------------------------------------------------------------------------------------- #
""" 
All this functions are designed to create a connection with Customer DB.
Right now, we have support for PostgresQL and MySQL database.
"""

# Support of postgresql
def psqldb_connnection(user, password, dbname, host=None, port=None):
  conn = psycopg2.connect(database=dbname,
                          host=host,
                          user=user,
                          password=password,
                          port=port)

  return conn


# supoort of mysql DB
def mysql_connection(user, password, dbname, host=None, port=None):
    conn = mysqldb.connect(user=user, password=password, database=dbname, host=host)
    return conn

# Customer DB connector controller 
def connect_cust_db(host, dbname, dbuser, dbpassword, dbtype):
    if dbtype.casefold() == "mysql":
        return mysql_connection(dbuser, dbpassword, dbname, host)
    elif dbtype.casefold == "postgressql":
        return psqldb_connnection(dbuser, dbpassword, dbname, host)
    return None

# --------------------------------------------------------------------------- #

def getConfig(db_config):
    config = list(db_config)
    return config[0], config[1], config[2]

def db_config_by_apikey(api_key):
    db_instance = connect_db()
    curr = db_instance.cursor()
    if api_key:
        curr.execute('SELECT dbconfig FROM customers WHERE apikey = %s', [api_key])
        return curr.fetchall()
    return None


def get_dbconn_by_apikey(db_type, api_key):
    db_config = list(zip(db_config_by_apikey(api_key)))
    db_config_str = db_config[0][0][0]
    db_config_dict = json.loads(db_config_str)

    # Get DB config
    dbname = list(db_config_dict.keys())[0]
    config = list(db_config_dict.values())[0]

    host, user, password = getConfig(config.values()) 

    # Check DB type and make conn accordingly
    if db_type.casefold() == "mysql":
        db = mysql_connection(user, password, dbname, host)
    elif db_type.casefold() == "postgresql":
        # For localhost DB
        db  = psqldb_connnection(user, password, dbname, host)
    # making DB connection: test postgres
    return db


# -------------------------------------------------------------------------------

# plannetscale DB : Our prod DB
def connect_db():
    host = os.getenv("HOST")
    user = os.getenv("DBUSER")
    passwd = os.getenv("PASSWORD")
    database = os.getenv("DATABASE")
    ssl_mode = "VERIFY_IDENTITY"
    ssl = {
        "ca": "/etc/pki/tls/certs/ca-bundle.crt"
    }

    db = mysqldb.connect(host=host, user=user, password=passwd, database=database)
    return db

def get_dbname_by_apikey(api_key):
    db_config = list(zip(db_config_by_apikey(api_key)))
    db_config_str = db_config[0][0][0]
    db_config_dict = json.loads(db_config_str)

    # Get DB config
    dbname = list(db_config_dict.keys())[0]
    return dbname


def create_file(json_schema, fileName):
 blob = BUCKET.blob(fileName)
 blob.upload_from_string(
     data = json.dumps(json_schema),
     content_type='application/json'
 )
 res = fileName + ' upload complete'
 return {"response": res}

def read_schema_file(fileName):
    blob = BUCKET.blob(fileName)
    file_data = json.loads(blob.download_as_string())
    return file_data

def save_schema_file(api_key, schema):
    # file_name = None

    # db_config = list(zip(db_config_by_apikey(api_key)))
    # db_config_str = db_config[0][0][0]
    # db_config_dict = json.loads(db_config_str)

    # Get DB Name
    dbname = schema.get('dbname')

    # Save it as a JSON file to Google CDN
    file_name = get_file_name(api_key, dbname)
    res = create_file(schema, file_name)
    return file_name

def get_dbconfig_from_dict(dbconfig):
    host = dbconfig.get('host')
    dbname = dbconfig.get('dbname')
    dbuser = dbconfig.get('dbuser')
    dbpassword = dbconfig.get('dbpassword')
    dbtype = dbconfig.get('db_type')
    return host, dbname, dbuser, dbpassword, dbtype

def get_dbconn_by_dbconfig(db_config, api_key):
    db_ins = connect_db()
    curr = db_ins.cursor()# check for right api key    
    if api_key:
        curr.execute('SELECT name FROM customers WHERE apikey = %s', [api_key])
        name = curr.fetchall()
        if name is None:
            return {"error": "Wrong API key", "status": "400"}
        host, dbname, dbuser, dbpassword, dbtype = get_dbconfig_from_dict(db_config)
        user_db_ins = connect_cust_db(host, dbname, dbuser, dbpassword, dbtype)
        return user_db_ins
    return {"error": "API key not provided", "status": "400"} 

# Get table schema
def get_table_schema(db_config, api_key, tables):
    conn = get_dbconn_by_dbconfig(db_config, api_key)
    # conn = get_dbconn_by_apikey(db_type, api_key)
    # dbname = get_dbname_by_apikey(api_key)

    host, dbname, dbuser, dbpassword, dbtype = get_dbconfig_from_dict(db_config)
    curr = conn.cursor()
    table_schemas = {}
    for table in tables:
        curr.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{table}'")
        columns = curr.fetchall()
        schema = {column[0]: column[1] for column in columns}
        table_schemas[table] = schema
    return {"dbname":dbname, "dbtype": dbtype, "schema": table_schemas}

##################################################################################################

"""
This part is for friday-cli: get and execute SQL queries based on schema provided
"""
def check_api_key(api_key):
    if api_key:
        db_ins = connect_db()
        curr = db_ins.cursor()
        curr.execute('SELECT name FROM customers WHERE apikey = %s', [api_key])
        name = curr.fetchall()
        if name:
           return True
    return False
        
def get_sql_query_by_db_schema(query, db_schema):
    final_sql_query = None
    if query and db_schema:
        prompt = "{}{}".format('A query to get', query)
        sql_q = makeit(db_schema, prompt)
        final_sql_query = sql_q.replace("\\n", " ").replace("\n", " ")
    return final_sql_query

def get_sql_query(api_key, query, db_schema, db_config=None):
    final_sql_query = None
    if api_key:
        is_right_api_key = check_api_key(api_key)
        if is_right_api_key is False:
            return {"error": "Something wrong with API Key. Make sure it is correct.", "status": "400"}
        final_sql_query = get_sql_query_by_db_schema(query, db_schema)
        if final_sql_query is None:
            return {"error": "Couldn't create SQL query. Make sure db_schema is correct.", "status": "400"}
        return 'select' + final_sql_query
    return {"error": "Empty API key.", "status": "400"} 


####################################################################################################

def exe_query(api_key, db_config, query):
    host, dbname, dbuser, dbpassword, dbtype = get_dbconfig_from_dict(db_config) 
    config_file_name = get_file_name(api_key, dbname)
    json_schema = read_schema_file(config_file_name) 
    conn = connect_cust_db(host, dbname, dbuser, dbpassword, dbtype)
    cur = conn.cursor()

    # Get the query ready using OpenAI api
    final_prompt = "{}{}".format('A query to get ', query)

    table_schemas_str = get_prompt(json_schema) + ''
    sql_stmt = makeit(table_schemas_str, final_prompt)
    final_sql_q = sql_stmt.replace("\\n", " ").replace("\n", " ")
    cur.execute('select ' + final_sql_q)
    return cur.fetchall()

def get_columns(db, table):
    ins = db.cursor()
    getColNamesStmt = "describe " + table
    ins.execute(getColNamesStmt)
    print(ins.fetchall())

def getApiKey(name):
    api_key = None
    # TODO: Shoudn't be connecting everytime making api call
    db_instance = connect_db()
    id = getId('org')
    if db_instance is not None:
        # TODO: check if config is right before creating API key
        api_key = get_api_key()
        curr = db_instance.cursor()
        # dbconfig = get_dbconfig(dbname, dbuser, dbpassword, host)
        now = datetime.datetime.now()
        curr.execute('INSERT INTO customers (id, name, apikey, totalapicall, created_at, dbconfig) values (%s, %s, %s, %s, %s, %s)', 
                     (id, name, api_key, 0, now, {}))
        db_instance.commit()
    return api_key



