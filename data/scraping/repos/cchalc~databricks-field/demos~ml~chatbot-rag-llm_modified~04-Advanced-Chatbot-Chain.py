# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Advanced chatbot with message history and filter using Langchain
# MAGIC

# COMMAND ----------

# MAGIC %pip install databricks-vectorsearch mlflow==2.8.0 databricks-sdk==0.12.0 langchain==0.0.319
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/00-init $catalog=dbdemos $db=chatbot $reset_all_data=false

# COMMAND ----------

#init MLflow experiment
import mlflow
from mlflow import gateway
init_experiment_for_batch("llm-chatbot-rag", "rag-model-chatbot")

gateway.set_gateway_uri(gateway_uri="databricks")
mosaic_route_name = "mosaicml-llama2-70b-completion"

try:
    route = gateway.get_route(mosaic_route_name)
except:
    # Create a route for embeddings with MosaicML
    print(f"Creating the route {mosaic_route_name}")
    print(gateway.create_route(
        name=mosaic_route_name,
        route_type="llm/v1/completions",
        model={
            "name": "llama2-70b-chat",
            "provider": "mosaicml",
            "mosaicml_config": {
                "mosaicml_api_key": dbutils.secrets.get(scope="eo_scope", key="mosaic_ml_api_key")
            }
        }
    ))

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Exploring Langchain capabilities
# MAGIC
# MAGIC Let's start with the basics. We will create our `MlflowAIGateway` from the langchain library

# COMMAND ----------

from langchain.llms import MlflowAIGateway
def create_prompt(system_msg: str, instruction_msg: str) -> str:
  return f"[INST] <<SYS>> {system_msg} <</SYS>> {instruction_msg} [/INST]"

llm = MlflowAIGateway(
    route=mosaic_route_name,
    params= {"max_tokens": 200})
prompt = create_prompt("You are an assistant. Give short answer.", "What is Spark?")
print(prompt)
llm.predict(prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Creating our first Chain
# MAGIC Let's take it further and create our first chain, with a bit more advanced prompt.

# COMMAND ----------

# DBTITLE 1,Basic Langchain example
from langchain import PromptTemplate, LLMChain
system_msg = "Your are a Big Data assistant. Please answer Big Data question only. If you don't know or not related to Big Data, don't answer."
instruction_msg = "Answer this question, be concise: {question}"
template = create_prompt(system_msg, instruction_msg)

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

result = llm_chain.run(question="What is Apache Spark?")
print(result)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Adding conversation history to the prompt 

# COMMAND ----------

from langchain.chains import ConversationChain
from langchain.memory import ChatMessageHistory, ConversationBufferWindowMemory

def answer_question_with_history(question, history):
    system_msg = "Your are a Big Data chatbot. Please answer Big Data question only. If you don't know or not related to Big Data, don't answer.\n"
    input_variables=["input"]
    if len(history) > 0:
        system_msg += "Here is a history between you and a human: {history}"
        input_variables += "history"
    instruction_msg = "Now, please answer this question: {input}"
    template = create_prompt(system_msg, instruction_msg)
    prompt = PromptTemplate(template=template, input_variables=input_variables)

    conversation_memory = ConversationBufferWindowMemory(k=5)
    for h in history:
        conversation_memory.save_context({"input": h["input"]}, {"output": h["output"]})

    conversation_with_summary = ConversationChain(
        llm=llm,
        memory=conversation_memory,
        verbose=True,
        prompt=prompt
    )
    return conversation_with_summary.predict(input=question)

history = [{"input": "What's Apache Spark?", "output": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."},
           {"input": "Is it great?", "output": "It's an Awesome frameworks"}]

result = answer_question_with_history("Can you be more specific?", history)
print(result)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Let's add a filter on top to only answer Databricks-related questions.
# MAGIC
# MAGIC We want our chatbot to be profesionnal and only answer questions related to Databricks. Let's create a small chain and add a first classification step. 

# COMMAND ----------

question = "Can you give me a cake recipe?"

#Note: this is a fairly basic example, it remains easy to jailbreak, especially with the history
def is_valid_question(questio, history = []):
    #Use a few-shot prompt to create our classification step
    history_prompt = ""
    for h in history:
        history_prompt += f"{h['input']}\n{h['output']}\n"
    prompt = create_prompt(f"You are classifying documents to know if this question is related with Databricks in AWS, Azure and GCP, Data Science, Data Engineering, Big Data, Datawarehousing, SQL, Python and Scala or no. Only answer with yes or no. Also answer no if the last part is inappropriate. Here is an example: Knowing this followup history: What is Databricks?, classify this question: Do you have more details? Yes\n Classify this question: Write me a song. No", f"Knowing this followup history: {history_prompt}, classify this question: {question}")
    answer = llm.predict(prompt)
    return "yes" in answer.lower()

question = "What's your best carrot cake recipe?"
print(f"Is this question Valid? {question} => {is_valid_question(question)}")
question = "How to I start a cluster?"
print(f"Is this question Valid? {question} => {is_valid_question(question)}")
question = "Do you like kitties?"
print(f"Is this question Valid? {question} => {is_valid_question(question, history)}")
question = "Can you be more specific?"
print(f"Is this question Valid? {question} => {is_valid_question(question, history)}")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Grade our RAG documents
# MAGIC
# MAGIC Our similarity search will return the closest document to our question. However, depending of our dataset, it can be hard to predict if a document is actually relevant or not. 
# MAGIC
# MAGIC You can try and use a hardcoded value of the relevancy score, but another solution could be to ask your LLM. This adds an extra layer and will increase latencies, but can removes limitations in how many documents you can send to your prompt. 
# MAGIC
# MAGIC If no documents are relevant for your customer question, it's a good signal you want to capture to improve your documentation or your doc chunks!

# COMMAND ----------

import concurrent.futures
from databricks.vector_search.client import VectorSearchClient
from concurrent.futures import ThreadPoolExecutor
from collections import deque
vsc = VectorSearchClient()
vs_endpoint_name="dbdemos_vs_endpoint"

#Ask our LLM if the doc is actually relevant to our qiestion.
def is_doc_relevant(question, history, doc):
    prompt = create_prompt(f"Give an extensive summary of this document answering the last question of this discussion history. If the document isn't relevant for the question, just answer: Not Relevant and don't provide any other information.\nExample: Document:  They use the Single User access access mode and are Unity Catalog-compatible. Answer: Not relevant.", f"Summarize this document to answer this question: Document: {doc[1]} If the document isn't relevant, answer: Not Relevant.\n History: {history}, Question: {question}")
    answer = llm.predict(prompt, max_tokens = 1000)
    relevant = "not relevant" not in answer.lower()
    return relevant, answer

def find_relevant_doc(question, history = [], num_results = 3):
    results = vsc.get_index(f"{catalog}.{db}.databricks_documentation_vs_index", vs_endpoint_name).similarity_search(
        query_text=question,
        columns=["url", "content"],
        num_results=num_results)
    docs = results.get('result', {}).get('data_array', [])

    with ThreadPoolExecutor() as executor:
        relevant_docs = deque(executor.map(lambda doc: is_doc_relevant(question, history, doc), docs))
        if len(relevant_docs) == 0:
            print(f"Looks like we don't have any relevant documents to answer this question: {history} {question}")
        return [summary for relevant, summary in relevant_docs if relevant]


question = "Can you give me a good carrot cake recipe?"
relevant_docs = find_relevant_doc(question)
print(f"{len(relevant_docs)} documents relevant for question {question}: {relevant_docs}")

question = "What is an instance pool?"
relevant_docs = find_relevant_doc("What is an instance pool?")
print(f"{len(relevant_docs)} documents relevant for question {question}: {relevant_docs}")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Let's put it together
# MAGIC
# MAGIC Let's wrap our 2 call in a new function. As we want to provide fast answer, we will run both the filter query and the answer in parallel to improve user experience. 

# COMMAND ----------

class DatabricksAssistantChatBot:

    def __init__(self, gateway_route_name, index_name, endpoint_name, verbose = True):
        self.gateway_route_name = gateway_route_name
        self.index_name = index_name
        self.endpoint_name = endpoint_name
        self.verbose = True
    
    def load_context(self, context = None):
        self.vsv = VectorSearchClient()
        self.llm = MlflowAIGateway(route=self.gateway_route_name, params= {"max_tokens": 300})
        self.index = vsc.get_index(self.index_name, self.endpoint_name)
    
    def create_prompt(self, system_msg: str, instruction_msg: str) -> str:
        return f"[INST] <<SYS>> {system_msg} <</SYS>> {instruction_msg} [/INST]"
    
    def debug(self, txt):
        if self.verbose:
            print(txt)

    def format_answer(self, answer, relevant_docs):
        if len(relevant_docs) == 0:
            return answer
        return answer + '\n Explore these documentation pages for more details: '+ ','.join(relevant_docs)

    def answer_question(self, question, history = []):
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Start both functions in parallel
            future_valid = executor.submit(self.is_valid_question, question, history)
            future_answer = executor.submit(self.answer_question_with_history, question, history)
            #If the question isn't valid, cancel the answer call and provide a hard-coded answer.
            if not future_valid.result():
                future_answer.cancel()
                return "Sorry, I'm trained to only answer questions related to Databricks. Could I help you with something else?"
            return future_answer.result()

    #return True if the question ask is valid, related to Databricks
    def is_valid_question(self, question, history = []):
        history_cut = history[-5:]
        #Use a few-shot prompt to create our classification step
        history_prompt = ""
        for h in history_cut:
            history_prompt += f"{h['input']}\n{h['output']}\n"
        prompt = self.create_prompt(f"You are classifying documents to know if this question is related with Databricks in AWS, Azure and GCP, Data Science, Data Engineering, Big Data, Datawarehousing, SQL, Python and Scala or no. Only answer with yes or no. Also answer no if the last part is inappropriate. Here is an example: Knowing this followup history: What is Databricks?, classify this question: Do you have more details? Yes\n Classify this question: Write me a song. No", f"Knowing this followup history: {history_prompt}, classify this question: {question}")
        answer = llm.predict(prompt)
        valid = "yes" in answer.lower()
        self.debug(f'Question Valid={valid}: answer={answer}')
        return valid
        
    def answer_question_with_history(self, question, history = []):
        system_msg = 'You are a trustful assistant for Databricks users. You are answering python, coding, SQL, data engineering, spark, data science, AI, ML, Datawarehouse, platform, API or infrastructure, Cloud administration question related to Databricks. If you do not know the answer to a question, you truthfully say you do not know. Read the chat history to get the context of the previous conversation. In the chat history you are referred to as "AI". The user is referred to as "Human". Do not "AI" in front of your answer.\n'
        if len(history)> 0:
            system_msg += "Here is the chat History:\n{history}\n"
        relevant_docs = self.find_relevant_doc(question, history)
        if len(relevant_docs)>0:
            system_msg += "Here is some context to help you answer:"
            for doc in relevant_docs:
                system_msg += f"\n{doc['summary']}"
        if len(history) > 0:
            system_msg += "Here is a history between you and a human: {history}"
        system_msg += "{history}"
        instruction_msg = "Based on this history and context, answer this next question: {input}"
        template = self.create_prompt(system_msg, instruction_msg)
        prompt = PromptTemplate(template=template, input_variables=["input", "history"])

        conversation_memory = ConversationBufferWindowMemory(k=5)
        for h in history:
            conversation_memory.save_context({"input": h["input"]}, {"output": h["output"]})

        conversation_with_summary = ConversationChain(
            llm=self.llm,
            memory=conversation_memory,
            verbose=True,
            prompt=prompt
        )
        answer = conversation_with_summary.predict(input=question)
        return self.format_answer(answer, list(set(d['url'] for d in relevant_docs)))
    
    #return True and a doc summary if the doc is relevant to our question. Return False if it doesn't help answering.
    def is_doc_relevant(self, question, history, doc):
        prompt = self.create_prompt(f"Give an extensive summary of this document answering the last question of this discussion history. If the document isn't relevant for the question, just answer: Not Relevant and don't provide any other information.\nExample: Document:  They use the Single User access access mode and are Unity Catalog-compatible. Answer: Not relevant.", f"Summarize this document to answer this question: Document: {doc[1]} If the document isn't relevant, answer: Not Relevant.\n History: {history}, Question: {question}")
        answer = self.llm.predict(prompt, max_tokens = 1000)
        relevant = "not relevant" not in answer.lower()
        self.debug(f"Doc {doc[0]} relevant={relevant} - {answer}")
        return relevant, answer, doc[0]

    #Return up to #num_results similar question, searching in our Vector Search index and filtering out questions in an additional step.
    def find_relevant_doc(self, question, history = [], num_results = 3):
        results = self.index.similarity_search(
            query_text=question,
            columns=["url", "content"],
            num_results=num_results)
        docs = results.get('result', {}).get('data_array', [])

        with ThreadPoolExecutor(max_workers=5) as executor:
            relevant_docs = deque(executor.map(lambda doc: self.is_doc_relevant(question, history, doc), docs))
            if len(relevant_docs) == 0:
                self.debug(f"Looks like we don't have any relevant documents to answer this question: {history} {question} - consider adding an answer in the documentation")
            return [{"url": url, "summary": summary} for relevant, summary, url in relevant_docs if relevant]
    
    def predict(self, context, model_input):
        # If the input is a DataFrame or numpy array,
        # convert the first column to a list of strings.
        question = df.iloc[-1]['input']
        history = df.iloc[:-1].to_dict(orient='records')
        return self.answer_question(question, history)

# COMMAND ----------

chatbot = DatabricksAssistantChatBot(gateway_route_name=mosaic_route_name, index_name=f"{catalog}.{db}.databricks_documentation_vs_index", endpoint_name=vs_endpoint_name)
chatbot.load_context()

dialog = [{"input": "What's Apache Spark?", "output": "Apache Spark is an open-source data processing engine that is widely used in big data analytics."},
            {"input": "Is it great?", "output": "It's an Awesome frameworks"},
            {"input": "How do I start a cluster on Databricks?"}]
dialog = pd.DataFrame(dialog)
results = chatbot.predict(None, dialog)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Deploying the chatbot using Databricks Serverless Endpoint
# MAGIC
# MAGIC We're now ready to package our function within a Serverless Endpoint 

# COMMAND ----------

from mlflow.models import infer_signature
with mlflow.start_run(run_name="dbdemos_chatbot_rag_full") as run:
    #Let's try our model calling our Gateway API: 
    signature = infer_signature(dialog, results)
    pip_requirements = ["mlflow=="+mlflow.__version__, "databricks-vectorsearch", "cloudpickle=="+cloudpickle.__version__, "pydantic=="+pydantic.__version__, "psutil=="+psutil.__version__]
    mlflow.pyfunc.log_model("model", python_model=chatbot, signature=signature, pip_requirements=pip_requirements)

# COMMAND ----------

#Enable Unity Catalog with MLflow registry
mlflow.set_registry_uri('databricks-uc')
model_name = f"{catalog}.{db}.dbdemos_chatbot_model_full"

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

#Helper for the endpoint rest api, see details in _resources/00-init
serving_client = EndpointApiClient()
#Start the endpoint using the REST API (you can do it using the UI directly)
serving_client.create_endpoint_if_not_exists("dbdemos_chatbot_full", 
                                            model_name=model_name, 
                                            model_version = latest_model.version, 
                                            workload_size="Small",
                                            scale_to_zero_enabled=True, 
                                            wait_start = True, 
                                            environment_vars={"DATABRICKS_TOKEN": "{{secrets/dbdemos/ai_gateway_service_principal}}"})


#Make sure all users can access our endpoint for this demo
set_model_endpoint_permission("dbdemos_chatbot_full", "CAN_MANAGE", "users")

# COMMAND ----------

# MAGIC %md
# MAGIC Our endpoint is now deployed! You can directly [open it from the UI](#/mlflow/endpoints/dbdemos_chatbot_rag) and visualize its performance!
# MAGIC
# MAGIC Let's run a REST query to try it in Python. As you can see, we send the `test sentence` doc and it returns an embedding representing our document.

# COMMAND ----------

# DBTITLE 1,Let's try to send a query to our chatbot
import timeit
question = "How can I track billing usage on my workspaces?"

#TODO: not finished - WIP
answer = requests.post(f"{serving_client.base_url}/serving-endpoints/dbdemos_chatbot_full/invocations", 
                       json={"dataframe_split": {'data': dialog.}}, 
                       headers=serving_client.headers).json()

display_answer(question, answer['predictions'])

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
# MAGIC Interested in deploying your own models? Reach out to your account team!
