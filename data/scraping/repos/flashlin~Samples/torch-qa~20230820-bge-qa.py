#pip -q install langchain huggingface_hub tiktoken
#pip -q install chromadb
#pip -q install PyPDF2 pypdf sentence_transformers
#pip -q install --upgrade together
#pip -q install -U FlagEmbedding
import re

import torch
import transformers
from langchain import PromptTemplate, FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import HuggingFacePipeline
from lanchainlit import load_model, load_tokenizer, load_llm, load_chain
from pdf_utils import load_and_split_pdf_texts_from_directory
from vectordb_utils import load_chroma_from_documents

hf_token = ""
device = "cuda"
embedding_model_name = "BAAI/bge-base-en"
model_name = 'TheBloke/vicuna-13B-v1.5-16K-GPTQ'

# 記憶體不足
model_name = 'models/vicuna-13B-v1.5-16K-GPTQ'
checkpoint = 'models/vicuna-13B-v1.5-16K-GPTQ/gptq_model-4bit-128g.safetensors'

model_name = 'models/vicuna-7B-v1.5-16K-GPTQ'
checkpoint = 'models/vicuna-7B-v1.5-16K-GPTQ/gptq_model-4bit-128g.safetensors'

model_name = 'models/Chinese-Llama-2-7b-4bit'
model_name = 'models/trurl-2-13b-8bit'  # 記憶體不足

# SQLCoder is a 15B parameter model
# SQLCoder has been tested on an A100 40GB GPU with bfloat16 weights.
# You can also load an 8-bit quantized version of the model on consumer GPUs with 20GB
model_name = 'models/sqlcoder'



def display_gpu_info(device='cuda'):
    total_memory = torch.cuda.get_device_properties(device).total_memory
    max_allocated_memory = torch.cuda.max_memory_allocated(device)
    remaining_memory = total_memory - max_allocated_memory
    # 轉換成 GB 單位
    total_memory_gb = total_memory / 1024 ** 3
    max_allocated_memory_gb = max_allocated_memory / 1024 ** 3
    remaining_memory_gb = remaining_memory / 1024 ** 3
    print(f"Total GPU Memory: {total_memory_gb:.2f} GB")
    print(f"Max Allocated GPU Memory: {max_allocated_memory_gb:.2f} GB")
    print(f"Remaining GPU Memory: {remaining_memory_gb:.2f} GB")



encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
embedding = HuggingFaceBgeEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={'device': 'cuda'},
    encode_kwargs=encode_kwargs
)


persist_directory = 'output/faissdb'
texts = load_and_split_pdf_texts_from_directory('data')


def split_documents_to_vector_db(split_documents, embeddings, db_path):
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    #texts = text_splitter.split_documents(documents)
    db = FAISS.from_documents(split_documents, embeddings)
    db.save_local(db_path)
    return db


split_documents_to_vector_db(texts, embedding, persist_directory)

vectordb = load_chroma_from_documents(texts, embedding)
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
print("--------------------------------------------")
print("embedding ready")

# llm = CTransformers(model=model_name,
#                     model_type="llama",
#                     max_new_tokens=512,
#                     temperature=0.1)


def load_gptq_safetensors(model_name):
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop
    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
    return model

# model = load_gptq_safetensors(model_name).to(device)


def get_max_memory_dict(cpu_memory=None, gpu_memory=None):
    max_memory = {}
    if gpu_memory:
        memory_map = list(map(lambda x: x.strip(), gpu_memory))
        for i in range(len(memory_map)):
            max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]

        max_cpu_memory = cpu_memory.strip() if cpu_memory is not None else '99GiB'
        max_memory['cpu'] = f'{max_cpu_memory}GiB' if not re.match('.*ib$', max_cpu_memory.lower()) else max_cpu_memory

    # If --auto-devices is provided standalone, try to get a reasonable value
    # for the maximum memory of device :0
    elif gpu_memory == "auto":
        total_mem = (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
        suggestion = round((total_mem - 1000) / 1000) * 1000
        if total_mem - suggestion < 800:
            suggestion -= 1000

        suggestion = int(round(suggestion / 1000))
        #logger.warning(f"Auto-assiging --gpu-memory {suggestion} for your GPU to try to prevent out-of-memory errors. You can manually set other values.")
        max_memory = {0: f'{suggestion}GiB', 'cpu': f'{cpu_memory or 99}GiB'}

    return max_memory if len(max_memory) > 0 else None


# quantize_config = BaseQuantizeConfig(
#             bits=4,
#             group_size=-1,
#             desc_act=False
#         )
#
# params = {
#         'model_basename': model_name,
#         'device': device,
#         'use_triton': False,
#         'inject_fused_attention': False,
#         'inject_fused_mlp': False,
#         'use_safetensors': True,
#         'trust_remote_code': True,
#         'max_memory': get_max_memory_dict(),
#         'quantize_config': quantize_config,
#         'use_cuda_fp16': True,
#     }
# model = AutoGPTQForCausalLM.from_quantized(model_name, **params)

# from safetensors.torch import load_file as safe_load
# model.load_state_dict(safe_load(checkpoint), strict=False)
print("--------------------------------------")
print(f"loading model")


# normal model
model = load_model(hf_token, model_name)
tokenizer, stopping_criteria = load_tokenizer(hf_token, model_id=model_name, device=device)
llm = load_llm(model, tokenizer, stopping_criteria)
chain = load_chain(llm, retriever)




def ask_sqlcoder():
    sqlcoder_question = "What products has the biggest fall in sales in 2022 compared to 2021? Give me the product name, the sales amount in both years, and the difference."

    sqlcoder_prompt = """### Instructions:
    Your task is convert a question into a SQL query, given a Postgres database schema.
    Adhere to these rules:
    - **Deliberately go through the question and database schema word by word** to appropriately answer the question
    - **Use Table Aliases** to prevent ambiguity. For example, `SELECT table1.col1, table2.col1 FROM table1 JOIN table2 ON table1.id = table2.id`.
    - When creating a ratio, always cast the numerator as float

    ### Input:
    Generate a SQL query that answers the questioq `{question}`.
    This query will run on a database whose schema is represented in this string:
    CREATE TABLE products (
      product_id INTEGER PRIMARY KEY, -- Unique ID for each product
      name VARCHAR(50), -- Name of the product
      price DECIMAL(10,2), -- Price of each unit of the product
      quantity INTEGER  -- Current quantity in stock
    );

    CREATE TABLE customers (
       customer_id INTEGER PRIMARY KEY, -- Unique ID for each customer
       name VARCHAR(50), -- Name of the customer
       address VARCHAR(100) -- Mailing address of the customer
    );

    CREATE TABLE salespeople (
      salesperson_id INTEGER PRIMARY KEY, -- Unique ID for each salesperson
      name VARCHAR(50), -- Name of the salesperson
      region VARCHAR(50) -- Geographic sales region
    );

    CREATE TABLE sales (
      sale_id INTEGER PRIMARY KEY, -- Unique ID for each sale
      product_id INTEGER, -- ID of product sold
      customer_id INTEGER,  -- ID of customer who made purchase
      salesperson_id INTEGER, -- ID of salesperson who made the sale
      sale_date DATE, -- Date the sale occurred
      quantity INTEGER -- Quantity of product sold
    );

    CREATE TABLE product_suppliers (
      supplier_id INTEGER PRIMARY KEY, -- Unique ID for each supplier
      product_id INTEGER, -- Product ID supplied
      supply_price DECIMAL(10,2) -- Unit price charged by supplier
    );

    -- sales.product_id can be joined with products.product_id
    -- sales.customer_id can be joined with customers.customer_id
    -- sales.salesperson_id can be joined with salespeople.salesperson_id
    -- product_suppliers.product_id can be joined with products.product_id

    ### Response:
    Based on your instructions, here is the SQL query I have generated to answer the question `{question}`:
    ```sql
    """.format(question=sqlcoder_question)
    eos_token_id = tokenizer.convert_tokens_to_ids(["```"])[0]

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
        max_new_tokens=400,
        do_sample=False,
        num_beams=5
    )
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    torch.cuda.empty_cache()
    # empty cache so that you do generate more results w/o memory crashing
    # particularly important on Colab – memory management is much more straightforward
    # when running on an inference service
    print(outputs[0].split("```sql")[-1].split("```")[0].split(";")[0].strip() + ";")



template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context: {context}
Question: {question}
Only return the helpful answer below and nothing else.
Helpful answer:
"""
prompt = PromptTemplate(
    template=template,
    input_variables=['context', 'question'])
qa_llm = RetrievalQA.from_chain_type(llm=llm,
                                     chain_type='stuff',
                                     retriever=retriever,
                                     return_source_documents=True,
                                     chain_type_kwargs={'prompt': prompt})
print('llm ready')

while True:
    prompt = input("query: ")
    if prompt == 'q':
        break
    # prompt = "How to add new b2b2c domain?"
    output = qa_llm({'query': prompt})
    print(output["result"])