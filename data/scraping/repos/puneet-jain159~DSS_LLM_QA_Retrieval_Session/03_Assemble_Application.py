# Databricks notebook source
# MAGIC %md The purpose of this notebook is to define and persist the model to be used by the QA Bot accelerator.  This notebook is inspired from https://github.com/databricks-industry-solutions/diy-llm-qa-bot.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC With our documents indexed, we can now focus our attention on assembling the core application logic.  This logic will have us retrieve a document from our vector store based on a user-provided question.  That question along with the document, added to provide context, will then be used to assemble a prompt which will then be sent to a model in order to generate a response. </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/bot_application.png' width=900>
# MAGIC
# MAGIC </p>
# MAGIC In this notebook, we'll first walk through these steps one at a time so that we can wrap our head around what all is taking place.  We will then repackage the logic as a class object which will allow us to more easily encapsulate our work.  We will persist that object as a model within MLflow which will assist us in deploying the model in the last notebook associated with this accelerator.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %run "./util/install-prep-libraries"

# COMMAND ----------

# DBTITLE 1,Optional : Load Ray Dashboard to show cluster Utilisation
# MAGIC %run "./util/install_ray"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import re
import time
import pandas as pd
import mlflow
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.schema import BaseRetriever
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.base_language import BaseLanguageModel

from langchain import LLMChain
from util.mptbot import HuggingFacePipelineLocal,TGILocalPipeline

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./util/notebook-config"

# COMMAND ----------

# MAGIC %md ##Step 1: Explore Answer Generation
# MAGIC
# MAGIC To get started, let's explore how we will derive an answer in response to a user provide question.  We'll start by defining that question here:

# COMMAND ----------

# DBTITLE 1,Specify Question
question =   """ What is the name of policy holder?"""

# COMMAND ----------

# MAGIC %md Using our vector store, assembled in the prior notebook, we will retrieve document chunks relevant to the question: 
# MAGIC
# MAGIC **NOTE** The OpenAI API key used by the OpenAIEmbeddings object is specified in an environment variable set during the earlier `%run` call to get configuration variables.

# COMMAND ----------

# DBTITLE 1,Retrieve Relevant Documents
# open vector store to access embeddings
try:
  embeddings
except:
  print("*** Load Embedding Model ***")
  if config['model_id'] == 'openai' :
    embeddings = OpenAIEmbeddings(model=config['embedding_model'])
  else:
    if "instructor" in config['embedding_model']:
      embeddings = HuggingFaceInstructEmbeddings(model_name= config['embedding_model'])
    else:
      embeddings = HuggingFaceEmbeddings(model_name= config['embedding_model'])

# load the documents in the vector store
vector_store = FAISS.load_local(embeddings=embeddings, folder_path=config['vector_store_path'])

# configure document retrieval 
n_documents = 10 # number of documents to retrieve 
retriever = vector_store.as_retriever(search_kwargs={'k': n_documents}) # configure retrieval mechanism

prepend_query ="Represent this sentence for searching relevant passages: \n "
# get relevant documents
docs = retriever.get_relevant_documents(prepend_query+ question)
for doc in docs: 
  print(doc.page_content,'\n','*'*50) 

# COMMAND ----------

# MAGIC %md We can now turn our attention to the prompt that we will send to the model.  This prompt needs to include placeholders for the *question* the user will submit and the document that we believe will provide the *context* for answering it.
# MAGIC
# MAGIC Please note that the prompt consists multiple prompt elements, defined using [prompt templates](https://python.langchain.com/en/latest/modules/prompts/chat_prompt_template.html).  In a nutshell, prompt templates allow us to define the basic structure of a prompt and more easily substitute variable data into them to trigger a response.  The system message prompt shown here provides instruction to the model about how we want it to respond.  The human message template provides the details about the user-initiated request.
# MAGIC
# MAGIC The prompts along with the details about the model that will respond to the prompt are encapsulated within an [LLMChain object](https://python.langchain.com/en/latest/modules/chains/generic/llm_chain.html).  This object simply defines the basic structure for resolving a query and returning a reponse:

# COMMAND ----------

config['template'] = """<s><<SYS>>
  You are a assistant built to answer policy related questions based on the context provided, the context is a document and use no other information.If the context does not provide enough relevant information to determine the answer, just say I don't know. If the context is irrelevant to the question, just say I don't know. If the query doesn't form a complete question, just say I don't know.Only answer the question asked and do not repeat the question
  <</SYS>>[INST] Given the context: {context}.  Answer the question {question} ?\n [/INST]
""".strip()

# COMMAND ----------

# DBTITLE 1,Define Chain to Generate Responses
# define system-level instructions
system_message_prompt = SystemMessagePromptTemplate.from_template(config['template'])
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])


if config['model_id']  == 'openai':

  # define model to respond to prompt
  llm = ChatOpenAI(model_name=config['openai_chat_model'], temperature=config['temperature'])

else:
  # define model to respond to prompt
  llm = TGILocalPipeline.from_model_id(
    model_id=config['model_id'],
    model_kwargs =config['model_kwargs'],
    pipeline_kwargs= config['pipeline_kwargs'])

# combine prompt and model into a unit of work (chain)
qa_chain = LLMChain(
  llm = llm,
  prompt = chat_prompt)

# COMMAND ----------

# MAGIC %md To actually trigger a response, we will loop through each of our docs from highest to lowest relevance and attempt to elicit a response.  Once we get a valid response, we'll stop.
# MAGIC
# MAGIC Please note, we aren't providing time-out handling or thoroughly validating the response from the model in this next cell.  We will want to make this logic more robust as we assemble our application class but for now we'll keep it simple to ensure the code is easy to read:

# COMMAND ----------

text = ""
for x in range(0,n_documents,3):
  for doc in docs[x:x+3]:
    text += "\nParagraph: \n" + doc.page_content
print(text)

# COMMAND ----------

# DBTITLE 1,Generate a Response
# for each provided document
text = ""
for doc in docs[0:3]:
  text += "\nParagraph: \n" + doc.page_content
# generate a response
output = qa_chain.generate([{'context': text, 'question': question}])

# get answer from results
generation = output.generations[0][0]
answer = generation.text

# display answer
if answer is not None:
  print(f"Question: {question}", '\n', f"Answer: {answer}")

# COMMAND ----------

# MAGIC %md ##Step 2: Assemble Model for Deployment
# MAGIC
# MAGIC Having explored the basic steps involved in generating a response, let's wrap our logic in a class to make deployment easier.  Our class will be initialized by passing the LLM model defintion, a vector store retriever and a prompt to the class.  The *get_answer* method will serve as the primary method for submitting a question and getting a response:

# COMMAND ----------

# DBTITLE 1,Define QABot Class
class QABot():


  def __init__(self, llm, retriever, prompt ,club_chunks = 3):
    self.llm = llm
    self.retriever = retriever
    self.prompt = prompt
    self.qa_chain = LLMChain(llm = self.llm, prompt=prompt)

  def _is_good_answer(self, answer):

    ''' check if answer is a valid '''

    result = True # default response

    badanswer_phrases = [ # phrases that indicate model produced non-answer
      "no information", "no context", "don't know", "no clear answer", "sorry","not mentioned","do not know","i don't see any information","i cannot provide information",
      "no answer", "no mention","not mentioned","not mention", "context does not provide", "no helpful answer", "not specified","not know the answer", 
      "no helpful", "no relevant", "no question", "not clear","not explicitly","provide me with the actual context document",
      "i'm ready to assist","I can answer the following questions"
      "don't have enough information", " does not have the relevant information", "does not seem to be directly related","cannot determine"
      ]
    if answer is None: # bad answer if answer is none
      results = False
    else: # bad answer if contains badanswer phrase
      for phrase in badanswer_phrases:
        if phrase in answer.lower():
          result = False
          break
    if answer[-1] == "?":
      result = False
    return result


  def _get_answer(self, context, question, timeout_sec=60):

    '''' get answer from llm with timeout handling '''

    # default result
    result = None

    # define end time
    end_time = time.time() + timeout_sec

    # try timeout
    while time.time() < end_time:

      # attempt to get a response
      try: 
        result =  qa_chain.generate([{'context': context, 'question': question}])
        break # if successful response, stop looping

      # if rate limit error...
      except openai.error.RateLimitError as rate_limit_error:
        if time.time() < end_time: # if time permits, sleep
          time.sleep(2)
          continue
        else: # otherwise, raiser the exception
          raise rate_limit_error

      # if other error, raise it
      except Exception as e:
        print(f'LLM QA Chain encountered unexpected error: {e}')
        raise e

    return result


  def get_answer(self, question):
    ''' get answer to provided question '''

    # default result
    result = {'answer':None, 'source':None, 'output_metadata':None}

    retriever_addon = "Represent this sentence for searching relevant passages: \n"

    # get relevant documents
    docs = self.retriever.get_relevant_documents(retriever_addon + question)

    # for each doc ...

    for x in range(0,len(docs),3):
      text = ""
      print(x,x+3)
      for doc in docs[x:x+3]:
        text += "\nParagraph: \n" + doc.page_content
    # print(text)

      # get key elements for doc
      # text = doc.page_content
      source = doc.metadata['source']

      # get an answer from llm
      output = self._get_answer(text, question)

      # get output from results
      generation = output.generations[0][0]
      answer = generation.text
      print("answer:",answer)
      output_metadata = output.llm_output

      # assemble results if not no_answer
      if self._is_good_answer(answer):
        result['answer'] = answer
        result['source'] = source
        result['output_metadata'] = output_metadata
        result['vector_doc'] = text
        return result
      else:
        result['answer'] = "Could not fine answer please rephrase the question or provide more context?"
        result['source'] = "NA"
        result['output_metadata'] = "NA"
        result['vector_doc'] = "NA"
        # print("text:",text)
      
    return result

# COMMAND ----------

# MAGIC %md Now we can test our class using the objects instantiated earlier:

# COMMAND ----------

# what is limit of the misfueling cost covered in the policy?
# what is the name of policy holder?
# what is the duration for the policy?
# what is the duration for the policy bought by the policy holder mentioned in the policy schedule / Validation schedule?
# "what is the vehicle age covered by the policy?"
# "what are the regions covered by the policy?"
# what happens if I lose my keys?

# COMMAND ----------

question =  "what is the duration for the policy mentioned in the policy schedule / Validation schedule"

# COMMAND ----------

# DBTITLE 0,Test the QABot Class
# instatiate bot object
qabot = QABot(llm, retriever, chat_prompt)

# get response to question
qabot.get_answer(question) 

# COMMAND ----------

# MAGIC %md ##Step 3: Persist Model to MLflow
# MAGIC
# MAGIC With our bot class defined and validated, we can now persist it to MLflow.  MLflow is an open source repository for model tracking and logging.  It's deployed by default with the Databricks platform, making it easy for us to record models with it.
# MAGIC
# MAGIC While MLflow now [supports](https://www.databricks.com/blog/2023/04/18/introducing-mlflow-23-enhanced-native-llm-support-and-new-features.html) both OpenAI and LangChain model flavors, the fact that we've written custom logic for our bot application means that we'll need to make use of the more generic [pyfunc](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#creating-custom-pyfunc-models) model flavor.  This model flavor allows us to write a custom wrapper for our model that gives us considerable control over how our model responds when deployed through standard, MLflow-provided deployment mechanisms. 
# MAGIC
# MAGIC To create a custom MLflow model, all we need to do is define a class wrapper of type *mlflow.pyfunc.PythonModel*. The *__init__* method will initialize an instance of our *QABot* class and persist it to an class variable.  And a *predict* method will serve as the standard interface for generating a reponse.  That method will receive our inputs as a pandas dataframe but we can write the logic with the knowledge that it will only be receiving one user-provided question at a time:

# COMMAND ----------

# DBTITLE 1,Define MLflow Wrapper for Model
class MLflowQABot(mlflow.pyfunc.PythonModel):

  def __init__(self, llm, retriever, chat_prompt):
    self.qabot = QABot(llm, retriever, chat_prompt)

  def predict(self, context, inputs):
    questions = list(inputs['question'])

    # return answer
    return [self.qabot.get_answer(q) for q in questions]

# COMMAND ----------

# MAGIC %md We can then instantiate our model and log the results to compare different model performance

# COMMAND ----------

try :
  spark.read.table("eval_questions")
except:
  print("Creating the evaluation dataset")
  questions = pd.DataFrame(
      {
          "question": [
              "what is limit of the misfueling cost covered in the policy?",
              "what happens if I lose my keys?",
              "What is the maximum Age of a Vehicle the insurance covers?",
              "what are the regions covered by the policy?",
              "what is the duration for the policy bought by the policy holder mentioned in the policy schedule / Validation schedule?"
          ]
      }
  )
  _ = (
    spark.createDataFrame(questions)
      .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable("eval_questions")
    )

# COMMAND ----------

# DBTITLE 1,Persist Evaluation to MLflow
# instantiate mlflow model
model = MLflowQABot(llm, retriever, chat_prompt)

with mlflow.start_run(run_name = config['model_id']):
  # Load the dataset and add it to the model run
    questions = mlflow.data.load_delta(table_name = 'eval_questions')
    mlflow.log_input(questions, context="eval_set")

  # Load model Parameters
    mlflow.log_param("model_id",config['model_id'])
    mlflow.log_param("embedding_model",config['embedding_model'])
    
    # mlflow.log_param("prompt template",config['template'])

  # log pipeline_params
    if "pipeline_kwargs" in config:
      mlflow.log_params(config['pipeline_kwargs'])
  
    if "model_kwargs" in config:
      mlflow.log_params(config['model_kwargs'])


    questions = spark.read.table("eval_questions").toPandas()
    outputs = model.predict(context = None,inputs =questions)



    # Evaluate the model on some example questions
    table_dict = {
    "questions": list(questions['question']),
    "outputs": [output['answer']for output in outputs],
    "source": [output['source']for output in outputs],
    "template": [config['template'] for output in outputs]
    }
    mlflow.log_table(table_dict,"eval.json")
    mlflow.end_run()

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | tiktoken | Fast BPE tokeniser for use with OpenAI's models | MIT  |   https://pypi.org/project/tiktoken/ |
# MAGIC | faiss-cpu | Library for efficient similarity search and clustering of dense vectors | MIT  |   https://pypi.org/project/faiss-cpu/ |
# MAGIC | openai | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/openai/ |

# COMMAND ----------


