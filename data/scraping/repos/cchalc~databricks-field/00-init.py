# Databricks notebook source
# MAGIC %md 
# MAGIC # init notebook setting up the backend. 
# MAGIC
# MAGIC Do not edit the notebook, it contains import and helpers for the demo
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
# MAGIC <img width="1px" src="https://www.google-analytics.com/collect?v=1&gtm=GTM-NKQ8TT7&tid=UA-163989034-1&cid=555&aip=1&t=event&ec=field_demos&ea=display&dp=%2F42_field_demos%2Ffeatures%2Fdolly-chatbot%2Finit&dt=ML">

# COMMAND ----------

dbutils.widgets.text("catalog", "hive_metastore", "Catalog")
dbutils.widgets.text("db", "dbdemos_llm", "Database")
dbutils.widgets.text("reset_all_data", "false", "Reset Data")
reset_all_data = dbutils.widgets.get("reset_all_data") == "true"

catalog = dbutils.widgets.get("catalog")
db = dbutils.widgets.get("db")
db_name = db

import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf, length, pandas_udf


# COMMAND ----------

# MAGIC %run ./00-global-setup $reset_all_data=$reset_all_data $catalog=$catalog $db=$db

# COMMAND ----------

import gc
from pyspark.sql.functions import pandas_udf
import pandas as pd
from typing import Iterator
import torch

folder =  "/dbdemos/product/llm/databricks-doc"

# Cache our model to dbfs to avoid loading them everytime
hugging_face_cache = "/dbfs"+folder+"/cache/hf"

import os
os.environ['TRANSFORMERS_CACHE'] = hugging_face_cache

# COMMAND ----------

if reset_all_data or is_folder_empty(folder):
  if reset_all_data and folder.startswith("/dbdemos/product/llm"):
    dbutils.fs.rm(folder, True)
  download_file_from_git('/dbfs'+folder, "databricks-demos", "dbdemos-dataset", "/llm/databricks-documentation")
  download_file_from_git('/dbfs'+folder+'/pdf_files', "databricks-demos", "dbdemos-dataset", "/llm/databricks-pdf-documentation")

else:
  print("data already existing. Run with reset_all_data=true to force a data cleanup for your local demo.")

# COMMAND ----------

#install Ghostscript and tesseract ocr on the cluster (should be done by init scripts)
def install_ocr_on_nodes():
    """
    install Ghostscript on the cluster (should be done by init scripts)
    """
    # from pyspark.sql import SparkSession
    import subprocess
    num_workers = max(1,int(spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers")))
    command = "sudo rm -r /var/cache/apt/archives/* /var/lib/apt/lists/* && sudo apt-get clean && sudo apt-get update && sudo apt-get install -y ghostscript tesseract-ocr" 
    subprocess.check_output(command, shell=True)

    def run_command(iterator):
        for x in iterator:
            yield subprocess.check_output(command, shell=True)

    # spark = SparkSession.builder.getOrCreate()
    data = spark.sparkContext.parallelize(range(num_workers), num_workers) 
    # Use mapPartitions to run command in each partition (worker)
    output = data.mapPartitions(run_command)
    try:
        output.collect();
        return True
    except Exception as e:
        print(f"Couldn't install on all node: {e}")
        return False

#Create custom class to load the PDF files from memory
#The complete new feature parsing and loading PDF files using PyMuPDF from bytes buffers is created and is in the process to be added to Langchain
#Here is the PR link: https://github.com/langchain-ai/langchain/pull/12066
# The constructor is wrapped in a function to avoid import errors.
def InMemoryPDFLoader(doc):
    from langchain.document_loaders.base import BaseLoader
    from langchain.schema import Document
    import fitz
    class InMemoryPDFLoader(BaseLoader):
        def __init__(self, stream=None, **kargs):
            super().__init__(**kargs)
            self.doc = fitz.open(stream=stream)

        def lazy_load(self):
            for page_num in range(self.doc.page_count):
                page = self.doc[page_num]
                metadata = {"page":page_num,
                            "total_pages":self.doc.page_count}
                yield Document(page_content=page.get_text(), metadata=metadata)

        def load(self):
            return list(self.lazy_load())
    return InMemoryPDFLoader(doc) 

# COMMAND ----------

import requests
import time
import re

#Helper to send REST queries. This will try to use an existing warehouse or create a new one.
class SQLStatementAPI:
    def __init__(self, warehouse_name = "dbdemos-shared-endpoint", catalog = "dbdemos", schema = "openai_demo"):
        self.base_url =dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
        self.token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
        username = re.sub("[^A-Za-z0-9]", '_', username)
        warehouse = self.get_or_create_endpoint(username, warehouse_name)
        #Try to create it
        if warehouse is None:
          raise Exception(f"Couldn't find or create a warehouse named {warehouse_name}. If you don't have warehouse creation permission, please change the name to an existing one or ask an admin to create the warehouse with this name.")
        self.warehouse_id = warehouse['warehouse_id']
        self.catalog = catalog
        self.schema = schema
        self.wait_timeout = "50s"

    def get_or_create_endpoint(self, username, endpoint_name):
        ds = self.get_demo_datasource(endpoint_name)
        if ds is not None:
            return ds
        def get_definition(serverless, name):
            return {
                "name": name,
                "cluster_size": "Small",
                "min_num_clusters": 1,
                "max_num_clusters": 1,
                "tags": {
                    "project": "dbdemos"
                },
                "warehouse_type": "PRO",
                "spot_instance_policy": "COST_OPTIMIZED",
                "enable_photon": "true",
                "enable_serverless_compute": serverless,
                "channel": { "name": "CHANNEL_NAME_CURRENT" }
            }
        def try_create_endpoint(serverless):
            w = self._post("api/2.0/sql/warehouses", get_definition(serverless, endpoint_name))
            if "message" in w and "already exists" in w['message']:
                w = self._post("api/2.0/sql/warehouses", get_definition(serverless, endpoint_name+"-"+username))
            if "id" in w:
                return w
            print(f"WARN: Couldn't create endpoint with serverless = {endpoint_name} and endpoint name: {endpoint_name} and {endpoint_name}-{username}. Creation response: {w}")
            return None

        if try_create_endpoint(True) is None:
            #Try to fallback with classic endpoint?
            try_create_endpoint(False)
        ds = self.get_demo_datasource(endpoint_name)
        if ds is not None:
            return ds
        print(f"ERROR: Couldn't create endpoint.")
        return None      
      
    def get_demo_datasource(self, datasource_name):
        data_sources = self._get("api/2.0/preview/sql/data_sources")
        for source in data_sources:
            if source['name'] == datasource_name:
                return source
        """
        #Try to fallback to an existing shared endpoint.
        for source in data_sources:
            if datasource_name in source['name'].lower():
                return source
        for source in data_sources:
            if "dbdemos-shared-endpoint" in source['name'].lower():
                return source
        for source in data_sources:
            if "shared-sql-endpoint" in source['name'].lower():
                return source
        for source in data_sources:
            if "shared" in source['name'].lower():
                return source"""
        return None
      
    def execute_sql(self, sql):
      x = self._post("api/2.0/sql/statements", {"statement": sql, "warehouse_id": self.warehouse_id, "catalog": self.catalog, "schema": self.schema, "wait_timeout": self.wait_timeout})
      return self.result_as_df(x, sql)
    
    def wait_for_statement(self, results, timeout = 600):
      sleep_time = 3
      i = 0
      while i < timeout:
        if results['status']['state'] not in ['PENDING', 'RUNNING']:
          return results
        time.sleep(sleep_time)
        i += sleep_time
        results = self._get(f"api/2.0/sql/statements/{results['statement_id']}")
      self._post(f"api/2.0/sql/statements/{results['statement_id']}/cancel")
      return self._get(f"api/2.0/sql/statements/{results['statement_id']}")
        
      
    def result_as_df(self, results, sql):
      results = self.wait_for_statement(results)
      if results['status']['state'] != 'SUCCEEDED':
        print(f"Query error: {results}")
        return pd.DataFrame([[results['status']['state'],{results['status']['error']['message']}, results]], columns = ['state', 'message', 'results'])
      if results["manifest"]['schema']['column_count'] == 0:
        return pd.DataFrame([[results['status']['state'], sql]], columns = ['state', 'sql'])
      cols = [c['name'] for c in results["manifest"]['schema']['columns']]
      results = results["result"]["data_array"] if "data_array" in results["result"] else []
      return pd.DataFrame(results, columns = cols)

    def _get(self, uri, data = {}, allow_error = False):
        r = requests.get(f"{self.base_url}/{uri}", params=data, headers=self.headers)
        return self._process(r, allow_error)

    def _post(self, uri, data = {}, allow_error = False):
        return self._process(requests.post(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _put(self, uri, data = {}, allow_error = False):
        return self._process(requests.put(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _delete(self, uri, data = {}, allow_error = False):
        return self._process(requests.delete(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _process(self, r, allow_error = False):
      if r.status_code == 500 or r.status_code == 403 or not allow_error:
        r.raise_for_status()
      return r.json()
    
#sql_api = SQLStatementAPI(warehouse_name = "dbdemos-shared-endpoint-test", catalog = "dbdemos", schema = "openai_demo")
#sql_api.execute_sql("select 'test'")

# COMMAND ----------

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore', SyntaxWarning)
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('ignore', UserWarning)

# COMMAND ----------

import urllib
import json
import mlflow
import time

class EndpointApiClient:
    def __init__(self):
        self.base_url =dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
        self.token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    def create_inference_endpoint(self, endpoint_name, served_models):
        data = {"name": endpoint_name, "config": {"served_models": served_models}}
        return self._post("api/2.0/serving-endpoints", data)

    def get_inference_endpoint(self, endpoint_name):
        return self._get(f"api/2.0/serving-endpoints/{endpoint_name}", allow_error=True)
      
      
    def inference_endpoint_exists(self, endpoint_name):
      ep = self.get_inference_endpoint(endpoint_name)
      if 'error_code' in ep and ep['error_code'] == 'RESOURCE_DOES_NOT_EXIST':
          return False
      if 'error_code' in ep and ep['error_code'] != 'RESOURCE_DOES_NOT_EXIST':
          raise Exception(f"endpoint exists ? {ep}")
      return True

    def create_endpoint_if_not_exists(self, endpoint_name, model_name, model_version, workload_size, scale_to_zero_enabled=True, wait_start=True, environment_vars = {}):
      models = [{
            "model_name": model_name,
            "model_version": model_version,
            "workload_size": workload_size,
            "scale_to_zero_enabled": scale_to_zero_enabled,
            "environment_vars": environment_vars
      }]
      if not self.inference_endpoint_exists(endpoint_name):
        r = self.create_inference_endpoint(endpoint_name, models)
      #Make sure we have the proper version deployed
      else:
        ep = self.get_inference_endpoint(endpoint_name)
        if 'pending_config' in ep:
            self.wait_endpoint_start(endpoint_name)
            ep = self.get_inference_endpoint(endpoint_name)
        if 'pending_config' in ep:
            model_deployed = ep['pending_config']['served_models'][0]
            print(f"Error with the model deployed: {model_deployed} - state {ep['state']}")
        else:
            model_deployed = ep['config']['served_models'][0]
        if model_deployed['model_version'] != model_version:
          print(f"Current model is version {model_deployed['model_version']}. Updating to {model_version}...")
          u = self.update_model_endpoint(endpoint_name, {"served_models": models})
      if wait_start:
        self.wait_endpoint_start(endpoint_name)
      
      
    def list_inference_endpoints(self):
        return self._get("api/2.0/serving-endpoints")

    def update_model_endpoint(self, endpoint_name, conf):
        return self._put(f"api/2.0/serving-endpoints/{endpoint_name}/config", conf)

    def delete_inference_endpoint(self, endpoint_name):
        return self._delete(f"api/2.0/serving-endpoints/{endpoint_name}")

    def wait_endpoint_start(self, endpoint_name):
      i = 0
      while self.get_inference_endpoint(endpoint_name)['state']['config_update'] == "IN_PROGRESS" and i < 500:
        if i % 10 == 0:
          print("waiting for endpoint to build model image and start...")
        time.sleep(10)
        i += 1
      ep = self.get_inference_endpoint(endpoint_name)
      if ep['state'].get("ready", None) != "READY":
        print(f"Error creating the endpoint: {ep}")
        
      
    # Making predictions

    def query_inference_endpoint(self, endpoint_name, data):
        return self._post(f"realtime-inference/{endpoint_name}/invocations", data)

    # Debugging

    def get_served_model_build_logs(self, endpoint_name, served_model_name):
        return self._get(
            f"api/2.0/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/build-logs"
        )

    def get_served_model_server_logs(self, endpoint_name, served_model_name):
        return self._get(
            f"api/2.0/serving-endpoints/{endpoint_name}/served-models/{served_model_name}/logs"
        )

    def get_inference_endpoint_events(self, endpoint_name):
        return self._get(f"api/2.0/serving-endpoints/{endpoint_name}/events")

    def _get(self, uri, data = {}, allow_error = False):
        r = requests.get(f"{self.base_url}/{uri}", params=data, headers=self.headers)
        return self._process(r, allow_error)

    def _post(self, uri, data = {}, allow_error = False):
        return self._process(requests.post(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _put(self, uri, data = {}, allow_error = False):
        return self._process(requests.put(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _delete(self, uri, data = {}, allow_error = False):
        return self._process(requests.delete(f"{self.base_url}/{uri}", json=data, headers=self.headers), allow_error)

    def _process(self, r, allow_error = False):
      if r.status_code == 500 or r.status_code == 403 or not allow_error:
        print(r.text)
        r.raise_for_status()
      return r.json()

# COMMAND ----------

def display_answer(question, answer):
  prompt = answer[0]["prompt"].replace('\n', '<br/>')
  answer = answer[0]["answer"].replace('\n', '<br/>').replace('Answer: ', '')
  #Tune the message with the user running the notebook. In real workd example we'd have a table with the customer details. 
  displayHTML(f"""
              <div style="float: right; width: 45%;">
                <h3>Debugging:</h3>
                <div style="border-radius: 10px; background-color: #ebebeb; padding: 10px; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; color: #363636"><strong>Prompt sent to the model:</strong><br/><i>{prompt}</i></div>
              </div>
              <h3>Chatbot:</h3>
              <div style="border-radius: 10px; background-color: #e3f6fc; padding: 10px; width: 45%; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; margin-left: 40px; font-size: 14px">
                <img style="float: left; width:40px; margin: -10px 5px 0px -10px" src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/robot.png?raw=true"/>Hey! I'm your Databricks assistant. How can I help?
              </div>
              <div style="border-radius: 10px; background-color: #c2efff; padding: 10px; width: 45%; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; font-size: 14px">{question}</div>
                <div style="border-radius: 10px; background-color: #e3f6fc; padding: 10px;  width: 45%; box-shadow: 2px 2px 2px #F7f7f7; margin-bottom: 10px; margin-left: 40px; font-size: 14px">
                <img style="float: left; width:40px; margin: -10px 5px 0px -10px" src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/robot.png?raw=true"/> {answer}
                </div>
        """)

# COMMAND ----------

# DBTITLE 1,Optional: Allowing Model Serving IPs
#If your workspace has ip access list, you need to allow your model serving endpoint to hit your AI gateway. Based on your region, IPs might change. Please reach out your Databrics Account team for more details.

# def allow_serverless_ip():
#   base_url =dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get(),
#   headers = {"Authorization": f"Bearer {<Your PAT Token>}", "Content-Type": "application/json"}
#   return requests.post(f"{base_url}/api/2.0/ip-access-lists", json={"label": "serverless-model-serving", "list_type": "ALLOW", "ip_addresses": ["<IP RANGE>"], "enabled": "true"}, headers = headers).json()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## Helpers to get catalog and index status:

# COMMAND ----------

# DBTITLE 1,endpoint
def vs_endpoint_exists(vsc, endpoint_name):
    try:
        vsc.get_endpoint(endpoint_name)
        return True
    except Exception as e:
        if 'Not Found' in str(e):
            print(f'Unexpected error describing the endpoint. Try deleting it? vsc.delete_endpoint({endpoint_name}) and rerun the previous cell')
            raise e
        return False
    
def wait_for_vs_endpoint_to_be_ready(vsc, vs_endpoint_name):
  for i in range(180):
    endpoint = vsc.get_endpoint(vs_endpoint_name)
    status = endpoint.get("endpoint_status", endpoint.get("status"))["state"].upper()
    print(endpoint)
    if "ONLINE" in status:
      return endpoint
    elif "PROVISIONING" in status:
      if i % 20 == 0: 
        print(f"Waiting for endpoint to be ready, this can take a few min... {endpoint}")
      time.sleep(10)
    else:
      raise Exception(f'''Error with the endpoint {vs_endpoint_name}. - this shouldn't happen: {endpoint}.\n Please delete it and re-run the previous cell: vsc.delete_endpoint("{vs_endpoint_name}")''')
  raise Exception(f"Timeout, your endpoint isn't ready yet: {vsc.get_endpoint(vs_endpoint_name)}")

# COMMAND ----------

# DBTITLE 1,index
def index_exists(vsc, index_full_name, endpoint_name):
    try:
        vsc.get_index(index_full_name, endpoint_name).describe()
        return True
    except Exception as e:
        if 'RESOURCE_DOES_NOT_EXIST' not in str(e):
            print(f'Unexpected error describing the index. This could be a permission issue. Try deleting it? vsc.delete_index({index_full_name})')
            raise e
        return False
    
def wait_for_index_to_be_ready(vsc, index_name, vs_endpoint_name):
  for i in range(180):
    idx = vsc.get_index(index_name, vs_endpoint_name).describe()
    index_status = idx.get('status', idx.get('index_status', {}))
    status = index_status.get('status', 'UNKOWN').upper()
    url = index_status.get('index_url', index_status.get('url', 'UNKOWN'))
    if "ONLINE" in status:
      return idx
    if "UNKOWN" in status:
      print(f"Can't get the status - will assume index is ready {idx} - url: {url}")
      return idx
    elif "PROVISIONING" in status:
      if i % 20 == 0: print(f"Waiting for index to be ready, this can take a few min... {index_status} - pipeline url:{url}")
      time.sleep(10)
    else:
        raise Exception(f'''Error with the index - this shouldn't happen. DLT pipeline might have been killed.\n Please delete it and re-run the previous cell: vsc.delete_index("{index_name}, {vs_endpoint_name}") \nIndex details: {idx}''')
  raise Exception(f"Timeout, your index isn't ready yet: {vsc.get_index(index_name, vs_endpoint_name)}")

# COMMAND ----------

import requests
import pandas as pd
import concurrent.futures
from bs4 import BeautifulSoup
import re

# Function to fetch HTML content for a given URL
def fetch_html(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error fetching {url}: {response}")
        return None
    return response.content

# Function to process a URL and extract text from the specified div
def download_web_page(url):
    html_content = fetch_html(url)
    if html_content:
        soup = BeautifulSoup(html_content, "html.parser")
        article_div = soup.find("div", itemprop="articleBody")
        if article_div:
            article_text = str(article_div)
            return {"url": url, "text": article_text.strip()}
    return None


def download_documentation_articles_from_urls(urls):
    # Download the web page (see _resource for the download_web_page function)
    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        results = list(executor.map(download_web_page, urls))
    
    # Filter out None values (URLs that couldn't be fetched or didn't have the specified div)
    valid_results = [result for result in results if result is not None]
    return spark.createDataFrame(valid_results)

# COMMAND ----------

# DBTITLE 1,Create and deploy an embedding proxy (temporary)

def launch_embedding_model_proxy_endpoint(proxy_model_full_name, proxy_endpoint_name, ai_gateway_route_name, sp_secret_scope = "dbdemos", sp_secret_key = "ai_gateway_service_principal"):
    import mlflow.pyfunc
    import json
    import pandas as pd
    from mlflow.gateway import MlflowGatewayClient
    import cloudpickle, pydantic, psutil
    #url used to send the request to the gateway from the serverless endpoint
    url = "https://"+dbutils.notebook.entry_point.getDbutils().notebook().getContext().browserHostName().get()


    serving_client = EndpointApiClient()
    if 'error_code' not in serving_client.get_inference_endpoint(proxy_endpoint_name):
        print(f'Proxy endpoint {proxy_endpoint_name} exists')
    else:
        print(f"Can't find proxy model serving endpoint named {proxy_endpoint_name} - start creating a new one, this will take a few minutes...")
        class LLMProxyModel(mlflow.pyfunc.PythonModel):
            import numpy as np 
            import pandas as pd 
            from mlflow import gateway

            def __init__(self):
                self.client = MlflowGatewayClient("databricks")

            def load_context(self, context):
                #Make sure you set os.environ['DATABRICKS_TOKEN'] in your endpoint deployment
                os.environ['DATABRICKS_HOST'] = url
            
            def predict(self, context, model_input):
                # If the input is a DataFrame or numpy array,
                # convert the first column to a list of strings.
                if isinstance(model_input, pd.DataFrame):
                    model_input = model_input.iloc[:, 0].tolist()
                elif isinstance(model_input, np.ndarray):
                    model_input = model_input[:, 0].tolist()
                elif isinstance(model_input, str):
                    model_input = [model_input]
                r = self.client.query(route=ai_gateway_route_name, data={"text": model_input})
                # AI gateway doesn't return an array for single entry, we want our model to be consistent
                if len(model_input) == 1 and len(r['embeddings']) > 1:
                    return np.array([r['embeddings']])
                return np.array(r['embeddings'])
            
        from mlflow.models import infer_signature
        #let's try our model:
        proxy_model = LLMProxyModel()
        input_example = ["How can I track billing usage on my workspaces?"]
        results = proxy_model.predict(None, input_example)

        with mlflow.start_run(run_name="chatbot_rag_embeddings") as run:
            #Let's try our model calling our Gateway API: 
            signature = infer_signature(input_example, results)
            pip_requirements = ["mlflow=="+mlflow.__version__, "databricks-vectorsearch", "cloudpickle=="+cloudpickle.__version__, "pydantic=="+pydantic.__version__, "psutil=="+psutil.__version__]
            mlflow.pyfunc.log_model("model", python_model=proxy_model, signature=signature, pip_requirements=pip_requirements, input_example=input_example)

        #Enable Unity Catalog with MLflow registry
        mlflow.set_registry_uri('databricks-uc')
        client = MlflowClient()
        #Add model within our catalog
        latest_model = mlflow.register_model(f'runs:/{run.info.run_id}/model', proxy_model_full_name)
        client.set_registered_model_alias(name=proxy_model_full_name, alias="prod", version=latest_model.version)

        #Make sure all other users can access the model for our demo(see _resource/00-init for details)
        set_model_permission(proxy_model_full_name, "ALL_PRIVILEGES", "account users")
        
        # Deploy model
        #Helper for the endpoint rest api, see details in _resources/00-init
        serving_client = EndpointApiClient()
        #Start the endpoint using the REST API (you can do it using the UI directly)
        serving_client.create_endpoint_if_not_exists(proxy_endpoint_name, 
                                                    model_name=proxy_model_full_name, 
                                                    model_version = latest_model.version, 
                                                    workload_size="Small",
                                                    scale_to_zero_enabled=True, 
                                                    wait_start = True, 
                                                    environment_vars={"DATABRICKS_TOKEN": f"{{{{secrets/{sp_secret_scope}/{sp_secret_key}}}}}"})
        #Make sure all users can access our endpoint for this demo
        # set_model_endpoint_permission(proxy_endpoint_name, "CAN_MANAGE", "users")

