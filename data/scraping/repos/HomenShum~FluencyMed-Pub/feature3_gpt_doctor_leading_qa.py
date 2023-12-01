
import pandas as pd
import glob

# Initialize a list to hold all stringified rows
all_data = []

### All dialogue files in the datasets
# for file in glob.glob('datasets/*.feather'):
#     print(file)
#     df = pd.read_feather(file)
    
#     # Convert each row to a string and append it to all_data
#     for index, row in df.iterrows():
#         all_data.append(row.to_string())
        
### Just iCliniq dialogue files
df = pd.read_feather('datasets/iCliniq.feather')

# Convert each row to a string and append it to all_data
for index, row in df.iterrows():
    all_data.append(row.to_string())

# Convert all_data to a DataFrame and write it to a Feather file
pd.DataFrame(all_data, columns=['row']).to_feather('combined.feather')
# Import necessary modules

import os
import pandas as pd
import openai
from llama_index import Document
# from llama_index.node_parser import SimpleNodeParser


# Set the OpenAI API key
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

OPENAI_API_KEY = open_file('openaiapikey.txt')
openai.api_key = OPENAI_API_KEY

# Set the OpenAI API key as an environment variable
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load the data from the feather file
df = pd.read_feather('combined.feather')[:100]

# Create a dictionary to represent your data
docs = []
for i, row in df.iterrows():
    docs.append(Document(
        text = row['row'], 
        doc_id=i
        ))

print(len(docs))

# parser = SimpleNodeParser()

# nodes = parser.get_nodes_from_documents(docs)

# print(len(nodes))

import pinecone
from dotenv import load_dotenv

load_dotenv()

os.environ['PINECONE_API_KEY'] = 'd0dc3dbd-8599-4d68-9816-6a0d74a290c5'
os.environ['PINECONE_ENVIRONMENT'] = 'us-west4-gcp'

pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'), 
    environment=os.getenv('PINECONE_ENVIRONMENT')
    )

index_name = 'langchainpdfchat'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        index_name,
        dimension=1536,
        metric='cosine'
    )

pinecone_index = pinecone.Index(index_name)

from llama_index.vector_stores import PineconeVectorStore

vector_store = PineconeVectorStore(
    pinecone_index=pinecone_index
    )

from llama_index import VectorStoreIndex, StorageContext, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

embed_model = OpenAIEmbedding(
    model='text-embedding-ada-002',
    embed_batch_size = 100
)

service_context = ServiceContext.from_defaults(
    embed_model=embed_model
)

index = VectorStoreIndex.from_documents(
    docs, 
    storage_context=storage_context, 
    service_context=service_context
)



##### Leading QA #########################################################################################################

import openai
import json
import re
import tenacity
import pandas as pd
from step2_ccsr_categorization import ccsr_categories_list

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

openai.organization = "org-uORaE3sY8YDfC5uDwRPP9uu5"
openai.api_key = open_file('openaiapikey.txt')

@tenacity.retry(
    stop=tenacity.stop_after_delay(30),
    wait=tenacity.wait_exponential(multiplier=1, min=1, max=30),
    retry=tenacity.retry_if_exception_type(openai.error.APIError),
    reraise=True,
)
def gpt_completion(
    prompt,
    engine="gpt-4",
    temp=0,  # set at 0 to ensure consistent completion, increase accuracy along with UUID
    top_p=1.0,
    tokens=500,
    freq_pen=0.25,
    pres_pen=0.0,
    stop=["<<END>>"],
):
    prompt = prompt.encode(encoding="ASCII", errors="ignore").decode()
    response = openai.ChatCompletion.create(
        model=engine,
        messages=[
            {"role": "system", "content":   """You are an AI assistant specialized in biomedical topics. You are provided with a text description from a patient's screening notes. Analyze the patient's notes and ask follow up question. Here are your instructions:

                    - Highlight conditions, symptoms, medical history, and any other information that can be mapped to specific CCSR categories.

                    - Keep in mind that the CCSR is used for grouping a large number of diseases into manageable categories for statistical analysis and reporting. 

                    - Ensure the conversation includes information that can guide the mapping to CCSR categories, such as the type of disease, cause, location in the body, and patient's age and sex. 

                    - Highlight medical advice or diagnostic information quoted and summarized from the given information. 

                    - Ensure the output is in markdown bullet point format for clarity.

                    - Encourage the user to consult a healthcare professional for advice."""},

            {"role": "user", "content": prompt},
        ],
        max_tokens=tokens,
        temperature=temp,
        top_p=top_p,
        frequency_penalty=freq_pen,
        presence_penalty=pres_pen,
        stop=stop,
    )
    text = response["choices"][0]["message"]["content"].strip()
    text = re.sub("\s+", " ", text)
    return text

# prompt = """  Ask the patient a medical diagnosis question based on the patient note and sample dialogues dataset. 
#                 Provide supplemental knowledge about the below CCSR categories that can guide the patient to understand the CCSR categories assigned to their patient note. 
#                 The output should be appropriate for the patient's age. 
#                 Ask only one question at a time so the patient will not be distracted or overwhelmed by the medical diligence process.
#                 \n
#                 """

prompt = """- Ask the patient follow up diagnosis question, give detailed information on the "why" using given information. 
                
            - The output should be appropriate for the patient's age. 
            
            - Encourage the user to consult a healthcare professional for advice.
            \n
            """


input_var_1 = "Patient Notes:"

input_end_session_note = """ 
                        
                        "input": "Hello doctor,I had mumps five months ago and after that, I started to have an infection in my left testes. It was swollen and now it has shrunk to almost half the size of the other one. As I am sexually active, I feel a pain in each of the vas deferens after sex. If I do not have sex for days, they become sensitive. I was treated with Ceftum 500 mg, the first time I had an infection. Now my question is, is there any chance that the infection is still in my body? And, do I need to get examined for it? For the time being, please suggest some precautionary antibiotics for my relief.",

                        \n
                        """

input_var_2 = "Clinical Classifications Software Refined (CCSR) categories listed below: \n"

prompt = prompt + input_var_1 + input_end_session_note + input_var_2

for i, category in enumerate(ccsr_categories_list, start=1):
    content = category['content']
    prompt += f"{i}. {content}\n"

input_dialogues_data = "sample dialogues dataset: \n"

# build a query engine from the index document
query_engine = index.as_query_engine()
res = query_engine.query(str(ccsr_categories_list))

input_var_3 = str(res) 

prompt = prompt + input_dialogues_data + input_var_3


print(gpt_completion(prompt))
