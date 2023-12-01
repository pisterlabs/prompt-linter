# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC # <font size="35"> Bronze table</font>
# MAGIC *The following script creates the Multi-plex Bronze table that will be used in the subsequent pipelines*

# COMMAND ----------

import os
from subprocess import Popen, PIPE
import json
import tika
from tika import parser

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, regexp_replace, col, current_timestamp, date_format

from pyspark.sql.types import StringType, StructType, StructField, IntegerType, TimestampType, LongType, BinaryType, ArrayType
import pdfplumber
from io import BytesIO
import tiktoken
import hashlib

from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain


# COMMAND ----------

# Function to run CLI command and retrieve the output
def run_cli_command(command):
    try:
        proc = Popen(command, stdout=PIPE, stderr=PIPE, shell=True)
        output, error = proc.communicate()
        print("login into azure CLI")
        return output.decode('utf-8')
    except Exception as e:
        return f"An error occurred: {str(e)}"
    

def mount_adls_folder_if_not_mounted(
    storage_account_name,
    metadata_dict,
    folder
):
    mounts = dbutils.fs.mounts()
    mount_point = f"/mnt/{storage_account_name}/{folder}"
    is_mounted = any(mount.mountPoint == mount_point for mount in mounts)

    AppID  = metadata_dict["client_id"]
    appSecret = metadata_dict['client_secret']
    tenant_id = metadata_dict['tenant_id']

  
    
    if not is_mounted:
        configs =  {
                "fs.azure.account.auth.type": "OAuth",
            "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
            "fs.azure.account.oauth2.client.id": AppID,
            "fs.azure.account.oauth2.client.secret": appSecret,
            "fs.azure.account.oauth2.client.endpoint": f"https://login.microsoftonline.com/{tenant_id}/oauth2/token" 
            }

        dbutils.fs.mount(
            source = f"abfss://raw@{storage_account_name}.dfs.core.windows.net/",
            mount_point = f"/mnt/{storage_account_name}/{folder}",
            extra_configs = configs
            
            )
        
        print(f"Mounted folder '{folder}' to '{mount_point}'")
    else:
        print(f"Folder '{folder}' is already mounted at '{mount_point}'")



# Function to log in to Azure CLI
def login_to_azure_cli(client_id ,
                        tenant_id , 
                        client_secret ):
    login_command = f"az login --service-principal -u {client_id} -p {client_secret} --tenant {tenant_id}"
    run_cli_command(login_command)


def get_dir_content(ls_path):
    dir_paths = dbutils.fs.ls(ls_path)
    file_paths = []

    for p in dir_paths:
        if p.isDir() and p.path != ls_path:
            subfolder_files = get_dir_content(p.path)
            file_paths.extend(subfolder_files)
        else:
            # Check if the file has a .pdf extension before adding it to the list
            if p.path.endswith(".pdf"):
                file_paths.append(p.path)

    return file_paths



@udf(returnType=StringType())
def extract_text_from_pdf_udf(pdf_bytes):
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
    if len(text) < 50:
        parsed = parser.from_buffer(pdf_bytes)
        text = parsed["content"]
    return text

@udf(returnType=StructType([
            StructField("item", StringType(), True),
            StructField("category", StringType(), True),
            StructField("author", StringType(), True),
            StructField("title", StringType(), True),
            StructField("filetype", StringType(), True)
        ]))
def extract_file_info_udf(file_path):
    # Extract information from the file path
    file_info = {}
    file_path_parts = file_path.split("/")

    file_info["item"] = file_path_parts[-4]
    file_info["category"] = file_path_parts[-3]
    file_info["author"] = file_path_parts[-2]
    file_info["title"] = file_path_parts[-1].split(".")[0]
    file_info["filetype"] = file_path_parts[-1].split(".")[1]   

    return file_info

@udf(returnType=IntegerType())
def num_tokens_from_string_udf(string, encoding_name="cl100k_base") :
    try:
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
    except:
        num_tokens = 0
    return num_tokens

@udf(returnType = StringType())
def create_id_udf(folder, typeofDoc, subject, author, title):
   
    # create a string to hash
    my_string = f"{folder}{typeofDoc}{subject}{author}{title}"
    # create a hash object using the SHA-256 algorithm
    hash_object = hashlib.sha256()
    # update the hash object with the string to be hashed
    hash_object.update(my_string.encode())
    # get the hexadecimal representation of the hash
    hex_dig = hash_object.hexdigest()
    return hex_dig



@udf(returnType = StringType())
def summarize_text_udf(text):
    """"""
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")    

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    
    docs = [Document(page_content=t) for t in chunks]
    chain.run(docs)

    return chain.run(docs)


# summarize_text_udf = udf(summarize_text, StringType())
# num_tokens_udf = udf(num_tokens_from_string, IntegerType())
# extract_text_udf = udf(extract_text_from_pdf, StringType())
# schema = StructType([
#     StructField("item", StringType(), True),
#     StructField("category", StringType(), True),
#     StructField("author", StringType(), True),
#     StructField("title", StringType(), True),
#     StructField("filetype", StringType(), True)
# ])

# extract_file_info_udf = udf(extract_file_info, schema)
# create_id_udf = udf(create_id, StringType())

# COMMAND ----------

class bams_summarizer:

    #initialize the class
    def __init__(self, model_name="gpt-3.5-turbo", temperature=0):

        # Initialize ChatOpenAI for summarization and chat
        self.llm_summarize = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.llm_chat = ChatOpenAI(model_name=model_name, temperature=temperature)

        self.embeddings = OpenAIEmbeddings()

        # Generate a summary of the loaded document
        def summarize(self, chain_type="map_reduce"):
            if not self.docs:
                raise ValueError("No document loaded. Please load a document first using `load_document` method.")
            
            # Load the summarization chain and run it on the loaded documents
            chain = load_summarize_chain(self.llm_summarize, chain_type=chain_type)
            return chain.run(self.docs)
        




# COMMAND ----------



# COMMAND ----------

#set secrets metadata
metadata_dict ={
        "client_id" : dbutils.secrets.get("bams-secrets" , "client_id"),
        "tenant_id" : dbutils.secrets.get("bams-secrets" , "tenant_id"),
        "client_secret": dbutils.secrets.get("bams-secrets" , "client_secret"),
        "service_principal_name" : dbutils.secrets.get("bams-secrets" , "service_principal_name"),
        "openai_api": dbutils.secrets.get("bams-secrets" , "openAIAPI")

}

#login to azure cli
login_to_azure_cli(client_id  = metadata_dict["client_id"], 
                   tenant_id  = metadata_dict['tenant_id'], 
                   client_secret = metadata_dict['client_secret']
                   )

#mount 
mount_adls_folder_if_not_mounted(
    storage_account_name = "adlsbamsllmappv2",
    metadata_dict = metadata_dict,
    folder = "raw"

)

mount_adls_folder_if_not_mounted(
    storage_account_name = "adlsbamsllmappv2",
    metadata_dict = metadata_dict,
    folder = "process"

)

os.environ['OPENAI_API_KEY'] = metadata_dict['openai_api']

# COMMAND ----------


# Read the binary file into a DataFrame
pdf_path = "/mnt/adlsbamsllmappv2/raw/"
# Define the schema of the binary file
binary_schema = StructType([
    StructField("path", StringType(), True),
    StructField("modificationTime", TimestampType(), True),
    StructField("length", LongType(), True),
    StructField("content", BinaryType(), True)
])

binary_df = spark.read.\
    format("binaryFile").\
        schema(binary_schema).\
            load(get_dir_content("/mnt/adlsbamsllmappv2/raw/")).\
                withColumn("timestamp", current_timestamp()).\
                withColumn("year_month", date_format("timestamp", "yyyy-MM")).\
            write.format("delta").\
                mode("overwrite").\
                    option("overwriteSchema", "true").\
                    option("path", "/mnt/adlsbamsllmappv2/process/").\
                        saveAsTable("bronze_raw")


# COMMAND ----------

get_dir_content('/mnt/adlsbamsllmappv2/raw/')

# COMMAND ----------

display(spark.read.\
    format("binaryFile").\
        schema(binary_schema).\
            load(get_dir_content("/mnt/adlsbamsllmappv2/raw/")))

# COMMAND ----------

def process_bronze():



    query= spark.\
            readStream.\
                format("delta").\
                    table("bronze_raw").\
            writeStream.\
                option("checkpointLocation", "/mnt/adlsbamsllmappv2/process/checkpoints").\
                option("mergeSchema", True).\
                partitionBy("year_month").\
                    trigger(availableNow=True).\
                            table("bronze")

    query.awaitTermination()


# COMMAND ----------

process_bronze()

# COMMAND ----------

display(spark.readStream\
      .table("bronze")\
      .createOrReplaceTempView("bronze_tmp"))

# COMMAND ----------

# MAGIC %sql
# MAGIC select*
# MAGIC from bronze

# COMMAND ----------

batch_df = spark.table("bronze")

a = (batch_df.\
            filter("path = 'dbfs:/mnt/adlsbamsllmappv2/raw/AcademicJournal/anthropology/Price/Descent, Clans and Territorial Organization in the Tikar Chiefdom of Ngambe, Cameroon.pdf'").\
                dropDuplicates(["path","length","content"]).\
                   withColumn("text", extract_text_udf(col("content"))).\
                       withColumn('summary', summarize_text_udf(col('text')))
)

# COMMAND ----------

def summarize_text_udf(text):
    """"""
    llm = OpenAI(temperature=0)
    chain = load_summarize_chain(llm, chain_type="map_reduce")    

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
    chunks = text_splitter.split_text(text)
    
    docs = [Document(page_content=t) for t in chunks]
    chain.run(docs)

    return str(chain.run(docs))

# COMMAND ----------




@udf(returnType= StringType())
def goog_summarize_udf(text_col):

    text_str = str(text_col)
    chunks = textwrap.wrap(text_str, 6000)
    text_result = []
    chunk_list = [summarize_chunk_with_goog(t) for t in chunks]

    print(chunk_list)
    print(len(chunk_list))

    return '\n '.join(chunk_list)
