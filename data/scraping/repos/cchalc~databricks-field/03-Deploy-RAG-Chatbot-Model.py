# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # 3/ Creating the Chat bot with Retrieval Augmented Generation (RAG)
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-inference.png?raw=true" style="float: right; margin-left: 10px"  width="900px;">
# MAGIC
# MAGIC
# MAGIC Our Vector Search Index is now ready!
# MAGIC
# MAGIC Let's now create and deploy a new Model Serving Endpoint to perform RAG.
# MAGIC
# MAGIC The flow will be the following:
# MAGIC
# MAGIC - A user asks a question
# MAGIC - The question is sent to our serverless Chatbot RAG endpoint
# MAGIC - The endpoint searches for docs similar to the question, leveraging Vector Search on our Documentation table
# MAGIC - The endpoint creates a prompt enriched with the doc
# MAGIC - The prompt is sent to the AI Gateway, ensuring security, stability and governance
# MAGIC - The gateway sends the prompt to a MosaicML LLM Endpoint (currently LLama 2 70B)
# MAGIC - MosaicML returns the result
# MAGIC - We display the output to our customer!

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch mlflow==2.8.0 databricks-sdk==0.12.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/00-init $catalog=dbdemos $db=chatbot $reset_all_data=false

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Deploying the AI gateway for llama2-70B completions
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-inference-1.png?raw=true" style="float: right; margin-left: 10px"  width="600px;">
# MAGIC
# MAGIC
# MAGIC As previously, the first step is to create an AI gateway for our MosaicML API. 
# MAGIC
# MAGIC We will be using llama2-70B as model.
# MAGIC
# MAGIC Deploying a massive LLM such as `llama2-70B` is complex and expensive, requiring multiple GPUs.
# MAGIC
# MAGIC To simplify this task, Databricks offers a `llama2-70B` endpoint through MosaicML, providing low-latencies answers at low cost.

# COMMAND ----------

#init MLflow experiment
import mlflow
from mlflow import gateway
init_experiment_for_batch("llm-chatbot-rag", "rag-model")

gateway.set_gateway_uri(gateway_uri="databricks")

mosaic_route_name = "mosaicml-llama2-70b-chat"

try:
    route = gateway.get_route(mosaic_route_name)
except:
    # Create a route for embeddings with MosaicML
    print(f"Creating the route {mosaic_route_name}")
    print(gateway.create_route(
        name=mosaic_route_name,
        route_type="llm/v1/chat",
        model={
            "name": "llama2-70b-chat",
            "provider": "mosaicml",
            "mosaicml_config": {
                "mosaicml_api_key": dbutils.secrets.get(scope="dbdemos", key="mosaic_ml_api_key")
            }
        }
    ))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's try our AI Gateway:
# MAGIC
# MAGIC AI Gateway accepts Databricks tokens as its authentication mechanism. 
# MAGIC
# MAGIC Let's send a simple REST call to our gateway. Note that we don't specify the LLM key nor the model details, only the gateway route.
# MAGIC
# MAGIC Remember each LLM might be trained with different prompt format. As example, llama2 is expecting the following [prompt](https://huggingface.co/blog/llama2#how-to-prompt-llama-2):
# MAGIC
# MAGIC ```
# MAGIC   <s>[INST]<<SYS>>
# MAGIC       {{ system_prompt }}
# MAGIC       <</SYS>>
# MAGIC       {{ user_msg_1 }} [/INST] {{ model_answer_1 }} </s>
# MAGIC   <s>[INST] {{ user_msg_2 }} [/INST] 
# MAGIC   ```
# MAGIC   
# MAGIC   This is not trivial to setup and easy to get wrong, leading to poor results. When setup with a `llm/v1/chat` type, the AI gateway handles this for you (you will have to craft the prompt if you use `llm/v1/completions`).

# COMMAND ----------

print(f"Calling AI gateway {gateway.get_route(mosaic_route_name).route_url}...")

messages = [{"role": "system", "content": "Your are a Big Data assistant. If you don't know the answer, say so."},
            {"role": "user", "content": "What is Apache Spark?"}]
    
r = gateway.query(route=mosaic_route_name, data = {"messages": messages})

print(r['candidates'][0]['message'])

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Creating an endpoint for the RAG chatbot, using the gateway we deployed
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-inference-1.png?raw=true" style="float: right; margin-left: 10px"  width="600px;">
# MAGIC
# MAGIC Our gateway is ready, and our different model deployments can now securely use the MosaicML route to query our LLM.
# MAGIC
# MAGIC We are now going to build our Chatbot RAG model and deploy it as an endpoint for realtime Q&A!
# MAGIC
# MAGIC #### A note on prompt engineering
# MAGIC
# MAGIC The usual prompt engineering method applies for this chatbot. Make sure you're prompting your model with proper parameters and matching the model prompt format if any.
# MAGIC
# MAGIC For a production-grade example, you'd typically use `langchain` and potentially send the entire chat history to your endpoint to support "follow-up" style questions *(this version doesn't support follow-up question but it's only a question of prompt engineering)*.
# MAGIC
# MAGIC More advanced chatbot behavior can be added here, including [Chain of Thought](https://cobusgreyling.medium.com/chain-of-thought-prompting-in-llms-1077164edf97), history summarization etc.
# MAGIC
# MAGIC Here is an example with `langchain`:
# MAGIC
# MAGIC ```
# MAGIC from langchain.llms import MlflowAIGateway
# MAGIC
# MAGIC gateway = MlflowAIGateway(
# MAGIC     gateway_uri="databricks",
# MAGIC     route="mosaicml-llama2-70b-completions",
# MAGIC     params={"temperature": 0.7, "top_p": 0.95,}
# MAGIC   )
# MAGIC prompt = PromptTemplate(input_variables=['context', 'question'], template="your template as string")
# MAGIC ```
# MAGIC
# MAGIC To keep our demo super simple and not getting confused with `langchain`, we will create a plain text template. 

# COMMAND ----------

vs_endpoint_name="dbdemos_vs_endpoint"

from mlflow.pyfunc import PythonModel

import os
#Service principal Databricks PAT token we'll use to access our AI Gateway
os.environ['DATABRICKS_TOKEN'] = dbutils.secrets.get("dbdemos", "ai_gateway_service_principal")

workspace_url = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()

from mlflow.gateway import MlflowGatewayClient
import cloudpickle, pydantic, psutil

class ChatbotRAG(mlflow.pyfunc.PythonModel):
    from mlflow import gateway
    
    def load_context(self, context):
        os.environ['DATABRICKS_HOST'] = workspace_url
        self.client = MlflowGatewayClient("databricks")
        from databricks.vector_search.client import VectorSearchClient
        self.vsc = VectorSearchClient(personal_access_token=os.environ['DATABRICKS_TOKEN'], workspace_url=workspace_url)
        
    #Send a request to our Vector Search Index to retrieve similar content.
    def find_relevant_doc(self, question, num_results = 1, relevant_threshold = 0.66):
        results = self.vsc.get_index(f"{catalog}.{db}.databricks_documentation_vs_index", vs_endpoint_name).similarity_search(
          query_text=question,
          columns=["url", "content"],
          num_results=num_results)
        docs = results.get('result', {}).get('data_array', [])
        #Filter on the relevancy score. Below 0.6 means we don't have good relevant content
        if len(docs) > 0 and docs[0][-1] > relevant_threshold :
          return {"url": docs[0][0], "content": docs[0][1]}
        return None

    def predict(self, context, model_input):
        # If the input is a DataFrame or numpy array,
        # convert the first column to a list of strings.
        if isinstance(model_input, pd.DataFrame):
            model_input = model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, np.ndarray):
            model_input = model_input[:, 0].tolist()
        elif isinstance(model_input, str):
            model_input = [model_input]
        answers = []
        for question in model_input:
          #Build the prompt
          system_prompt = "You are an assistant for Databricks users. You are answering python, coding, SQL, data engineering, spark, data science, DW and platform, API or infrastructure administration question related to Databricks. If the question is not related to one of these topics, kindly decline to answer."
          doc = self.find_relevant_doc(question)
          #Add docs from our knowledge base to the prompt
          if doc is not None:
            system_prompt += f"\n\n Here is a documentation page which might help you answer: \n\n{doc['content']}"
            
          user_prompt = f"Answer the following user question. If you don't know or the question isn't relevant to Databricks or professional, don't answer. Only give a detailed answer. Don't add note or comment.\n\n  Question: {question}"
        
          #Note the DATABRICKS_TOKEN environement variable is set in the endpoint and will be use for the auth to our mlflow gateway
          messages = [{"role": "system", "content": system_prompt},
                      {"role": "user",   "content": user_prompt}]
          response = self.client.query(route=mosaic_route_name, data={"messages": messages})
        
          answer = response['candidates'][0]['message']['content']
          if doc is not None:
            answer += f"""\nFor more details, <a href="{doc['url']}">open the documentation</a>  """
          answers.append({"answer": answer.replace('\n', '<br/>'), "prompt": json.dumps(messages)})
        return answers

# COMMAND ----------

# DBTITLE 1,Let's try our chatbot in the notebook directly:
proxy_model = ChatbotRAG()
proxy_model.load_context(None)
results = proxy_model.predict(None, ["How can I track billing usage on my workspaces?"])
print(results[0]["answer"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Saving our chatbot model in Unity Catalog

# COMMAND ----------

from mlflow.models import infer_signature
with mlflow.start_run(run_name="chatbot_rag") as run:
    python_model = ChatbotRAG()
    #Let's try our model calling our Gateway API: 
    signature = infer_signature(["some", "data"], results)
    pip_requirements = ["mlflow=="+mlflow.__version__, "databricks-vectorsearch", "cloudpickle=="+cloudpickle.__version__, "pydantic=="+pydantic.__version__, "psutil=="+psutil.__version__]
    mlflow.pyfunc.log_model("model", python_model=python_model, signature=signature, pip_requirements=pip_requirements)

# COMMAND ----------

# DBTITLE 1,Register our model to MLFlow
#Enable Unity Catalog with MLflow registry
mlflow.set_registry_uri('databricks-uc')
model_name = f"{catalog}.{db}.dbdemos_chatbot_model"

client = MlflowClient()
try:
  #Get the model if it is already registered to avoid re-deploying the endpoint
  latest_model = client.get_model_version_by_alias(model_name, "prod")
  print(f"Our model is already deployed on UC: {model_name}")
except:  
  #Add model within our catalog
  latest_model = mlflow.register_model(f'runs:/{run.info.run_id}/model', model_name)
  client.set_registered_model_alias(name=model_name, alias="prod", version=latest_model.version)

  #Make sure all other users can access the model for our demo(see _resource/00-init for details)
  set_model_permission(model_name, "ALL_PRIVILEGES", "account users")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Let's now deploy our realtime model endpoint
# MAGIC
# MAGIC Let's leverage Databricks Secrets to load the credentials to hit our Vector Search. Note that this is done using a Service Principal, a technical user that will be allowed to query the gateway and the index. See the [documentation](https://docs.databricks.com/en/machine-learning/model-serving/store-env-variable-model-serving.html) for more details.

# COMMAND ----------

chatbot_model_serving_endpoint = "dbdemos_chatbot_rag"
#Helper for the endpoint rest api, see details in _resources/00-init
serving_client = EndpointApiClient()
#Start the endpoint using the REST API (you can do it using the UI directly)
serving_client.create_endpoint_if_not_exists(chatbot_model_serving_endpoint, 
                                            model_name=f"{catalog}.{db}.dbdemos_chatbot_model", 
                                            model_version = latest_model.version, 
                                            workload_size="Small",
                                            scale_to_zero_enabled=True, 
                                            wait_start = True, 
                                            environment_vars={"DATABRICKS_TOKEN": "{{secrets/dbdemos/ai_gateway_service_principal}}"})

#Make sure all users can access our endpoint for this demo
set_model_endpoint_permission(chatbot_model_serving_endpoint, "CAN_MANAGE", "users")

# COMMAND ----------

# MAGIC %md
# MAGIC Our endpoint is now deployed! You can directly [open it from the UI](#/mlflow/endpoints/dbdemos_chatbot_rag) and visualize its performance!
# MAGIC
# MAGIC Let's run a REST query to try it in Python. As you can see, we send the `test sentence` doc and it returns an embedding representing our document.

# COMMAND ----------

# DBTITLE 1,Let's try to send a query to our chatbot
import timeit
question = "How can I track billing usage on my workspaces?"

answer = requests.post(f"{serving_client.base_url}/serving-endpoints/{chatbot_model_serving_endpoint}/invocations", 
                       json={"dataframe_split": {'data': [question]}}, 
                       headers=serving_client.headers).json()

display_answer(question, answer['predictions'])

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Congratulations! You have deployed your first GenAI RAG model!
# MAGIC
# MAGIC You're now ready to deploy the same logic for your internal knowledge base leveraging Lakehouse AI.
# MAGIC
# MAGIC We've seen how the Lakehouse AI is uniquely positioned to help you solve your GenAI challenge:
# MAGIC
# MAGIC - Simplify Data Ingestion and preparation with Databricks Engineering Capabilities
# MAGIC - Accelerate Vector Search  deployment with fully managed indexes
# MAGIC - Simplify, secure and control your LLM access with AI gateway
# MAGIC - Access MosaicML's LLama 2 endpoint
# MAGIC - Deploy realtime model endpoint to perform RAG 
# MAGIC
# MAGIC Lakehouse AI is uniquely positioned to accelerate your GenAI deployment.
# MAGIC
# MAGIC Want to go further? Discover how to deploy a complete more chatbot, including:
# MAGIC
# MAGIC * Add a step to ensure non-relevant question are ignored (only answer Databricks-related content)
# MAGIC * Scale your RAG context to many documents
# MAGIC * Add Langchain 
# MAGIC
# MAGIC Open then [04-Advanced-Chatbot-Chain]($./04-Advanced-Chatbot-Chain) notebook 
# MAGIC
# MAGIC Interested in fine tuning and deploying your own models? Reach out to your account team!
