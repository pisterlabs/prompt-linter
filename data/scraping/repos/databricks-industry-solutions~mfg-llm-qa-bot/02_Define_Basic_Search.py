# Databricks notebook source
# MAGIC %md You may find this notebook on https://github.com/databricks-industry-solutions/mfg-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Define Basic Search
# MAGIC
# MAGIC In this notebook, we will test out loading the vector database for similarity search. Additionally, we create a simple example of combining the open sourced LLM (defined in the /utils/configs) and the similarity search as a retriever. Think of this as a stand-alone implementation without any MLflow packaging
# MAGIC
# MAGIC
# MAGIC <p>
# MAGIC     <img src="https://github.com/databricks-industry-solutions/mfg-llm-qa-bot/raw/main/images/Basic-similarity-search.png" width="700" />
# MAGIC </p>
# MAGIC
# MAGIC This notebook was tested on the following infrastructure:
# MAGIC * DBR 13.2ML (GPU)
# MAGIC * g5.4xlarge or g5.8xlarge (AWS) - however comparable infra on Azure should work (A10s)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Install required libraries

# COMMAND ----------

# MAGIC %pip install -U langchain==0.0.203 transformers==4.30.1 accelerate==0.20.3 einops==0.6.1 xformers==0.0.20 sentence-transformers==2.2.2 typing-inspect==0.8.0 typing_extensions==4.5.0 faiss-cpu==1.7.4 tiktoken==0.4.0 

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load in common configs

# COMMAND ----------

# MAGIC %run "./utils/configs"

# COMMAND ----------

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from torch import cuda, bfloat16,float16
import transformers
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, pipeline
from langchain.chains import RetrievalQA

# COMMAND ----------

# MAGIC %md 
# MAGIC ##Test Similarity search
# MAGIC
# MAGIC In the code below we are loading in the vector store that we defined in the previous notebook with the embedding model that we used to create the vector database. In this case, we are using the [FAISS library from Meta](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) to store our embeddings. 

# COMMAND ----------

vector_persist_dir = configs['vector_persist_dir']
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

# Load from FAISS
vectorstore = FAISS.load_local(vector_persist_dir, embeddings)

#fetch_k : amount of documents to fetch to pass into search algorithm
# k: amount of documents to return
# filter = any keywords to pre-filter docs on
def similarity_search(question, filter={}, fetch_k=100, k=12):
  matched_docs = vectorstore.similarity_search(question, filter=filter, fetch_k=fetch_k, k=k)
  sources = []
  content = []
  for doc in matched_docs:
    sources.append(
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
    )
    content.append(doc.page_content)
    

  return matched_docs, sources, content


matched_docs, sources, content = similarity_search('Who provides recommendations on workspace safety on Acetone', {'Name':'ACETONE'})
print(content)
print(matched_docs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Initializing the Hugging Face Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC The first thing we need to do is initialize a `text-generation` pipeline with Hugging Face transformers. The Pipeline requires three things that we must initialize first, those are:
# MAGIC
# MAGIC * A LLM, in this case it will be defined in the /utils/configs notebook
# MAGIC
# MAGIC * The respective tokenizer for the model.
# MAGIC
# MAGIC We'll explain these as we get to them, let's begin with our model.
# MAGIC
# MAGIC We initialize the model using the externalized configs such as automodelconfigs and pipelineconfigs

# COMMAND ----------

#configs for the model are externalized in var automodelconfigs

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

print(f"{configs['model_name']} using configs {automodelconfigs}")
#account for small variations in code for loading models between models
if 'mpt' in configs['model_name']:
  modconfig = transformers.AutoConfig.from_pretrained(configs['model_name'] ,
    trust_remote_code=True
  )
  #modconfig.attn_config['attn_impl'] = 'triton'
  model = transformers.AutoModelForCausalLM.from_pretrained(
      configs['model_name'],
      config=modconfig,
      **automodelconfigs
  )
elif 'flan' in configs['model_name']:
  model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
      configs['model_name'],
      **automodelconfigs
  )
else:
  model = transformers.AutoModelForCausalLM.from_pretrained(
      configs['model_name'],
      **automodelconfigs
  )

#  model.to(device) -> `.to` is not supported for `4-bit` or `8-bit` models.
listmc = automodelconfigs.keys()
if 'load_in_4bit' not in listmc and 'load_in_8bit' not in listmc:
  model.eval()
  model.to(device)
if 'RedPajama' in configs['model_name']:
  model.tie_weights()

print(f"Model loaded on {device}")

# COMMAND ----------

# MAGIC %md
# MAGIC The pipeline requires a tokenizer which handles the translation of human readable plaintext to LLM readable token IDs. The Huggingface model card will give you info on the tokenizer

# COMMAND ----------

token_model= configs['tokenizer_name']
#load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(token_model)


# COMMAND ----------

# MAGIC %md
# MAGIC Finally we need to define the _stopping criteria_ of the model. The stopping criteria allows us to specify *when* the model should stop generating text. If we don't provide a stopping criteria the model just goes on a bit of a tangent after answering the initial question.

# COMMAND ----------

#If Stopping Criteria is needed
from transformers import StoppingCriteria, StoppingCriteriaList


# for example. mpt-7b is trained to add "<|endoftext|>" at the end of generations
stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])
print(stop_token_ids)
print(tokenizer.eos_token)
print(stop_token_ids)
# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    for stop_id in stop_token_ids:
      if input_ids[0][-1] == stop_id:
        return True
    return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

# COMMAND ----------

# MAGIC %md
# MAGIC Now we're ready to initialize the HF pipeline. There are a few additional parameters that we must define here. Comments explaining these have been included in the code.
# MAGIC The easiest way to tackle NLP tasks is to use the pipeline function. It connects a model with its necessary pre-processing and post-processing steps. This allows you to directly input any text and get an answer.

# COMMAND ----------

# device=device, -> `.to` is not supported for `4-bit` or `8-bit` models.
if 'load_in_4bit' not in listmc and 'load_in_8bit' not in listmc:
  generate_text = transformers.pipeline(
      model=model, tokenizer=tokenizer,
      device=device,
      pad_token_id=tokenizer.eos_token_id,
      #stopping_criteria=stopping_criteria,
      **pipelineconfigs
  )
else:
  generate_text = transformers.pipeline(
      model=model, tokenizer=tokenizer,
      pad_token_id=tokenizer.eos_token_id,
      #stopping_criteria=stopping_criteria,
      **pipelineconfigs
  )  

# COMMAND ----------

# MAGIC %md
# MAGIC The next block of code is the critical element to understand how the vectorstore is being passed to the QA chain as a retriever (the retrieval augmentation)
# MAGIC
# MAGIC Additional ref docs [here](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.RetrievalQA.html)

# COMMAND ----------

llm = HuggingFacePipeline(pipeline=generate_text)

promptTemplate = PromptTemplate(
        template=configs['prompt_template'], input_variables=["context", "question"])
chain_type_kwargs = {"prompt":promptTemplate}

# metadata filtering logic internal implementation, if interested, in 
# def similarity_search_with_score_by_vector in
# https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/vectorstores/faiss.py

# To test metadata based filtering.
#filterdict={'Name':'ACETALDEHYDE'}
filterdict={}
retriever = vectorstore.as_retriever(search_kwargs={"k": configs['num_similar_docs'], "filter":filterdict}, search_type = "similarity")

qa_chain = RetrievalQA.from_chain_type(llm=llm, 
                                       chain_type="stuff", 
                                       retriever=retriever, 
                                       return_source_documents=True,
                                       chain_type_kwargs=chain_type_kwargs,
                                       verbose=False)

# COMMAND ----------

# MAGIC %md Optionally dynamically pass a filter into the chain to pre-filter docs

# COMMAND ----------

#filterdict={'Name':'ACETONE'}

# fetch_k Amount of documents to pass to search algorithm
retriever.search_kwargs = {"k": 6, "filter":filterdict, "fetch_k":30}
res = qa_chain({"query":"What issues can acetone exposure cause"})
print(res)

print(res['result'])

# COMMAND ----------

# MAGIC %md
# MAGIC Confirm this is working:

# COMMAND ----------

filterdict={}
retriever.search_kwargs = {"k": 6, "filter":filterdict, "fetch_k":20}
res = qa_chain({"query":"Explain to me the difference between nuclear fission and fusion."})
res

#print(res['result'])

# COMMAND ----------

filterdict={}
retriever.search_kwargs = {"k": 6, "filter":filterdict, "fetch_k":40}
res = qa_chain({'query':'what should we do if OSHA is involved?'})
res

#print(res['result'])


# COMMAND ----------

# MAGIC %md
# MAGIC Optional Cleanup

# COMMAND ----------

# del qa_chain
# del tokenizer
# del model
# with torch.no_grad():
#     torch.cuda.empty_cache()
# import gc
# gc.collect()
