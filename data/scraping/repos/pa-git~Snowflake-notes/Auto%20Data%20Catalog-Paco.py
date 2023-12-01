#!/usr/bin/env python
# coding: utf-8

# ## Design:
# 1. Receive object metadata
# 2. Retrieve related memory
# 3. Build Prompt using template
# 4. Prompt the LLM
# 5. Receive usable response
# 6. Save a log file of the response
# 7. Add the response to the memory

# In[304]:


import os
os.environ['OPENAI_API_KEY'] = "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"


# ## Define inputs
# -  Object metadata
# -  Memory
# -  Instructions
# -  Example
# -  System message

# ### Try different inputs

# In[477]:


# Whatever info we want to provide object the object that needs to be cataloged
object_metadata = { 
    "table_name" : "atlas_cc_class",
    "table_context" : "Financial organizational structure",
    "columns": {
        "cc_class_code" : {
            "data_type" : "VARCHAR",
            "size" : 5,        
            "sample_values" : "RTB, CTB"
        },
        "cc_class_description" : {
            "data_type" : "VARCHAR",
            "size" : 255,        
            "sample_values" : "RTB Headcount, CTB Tech Clearing"
        }
    }
}


# In[478]:


instructions = "Assign a real world business concept and a business description to each of the columns in the JSON input. "
instructions += "Do so by appending 2 new attributes called business_concept and business_description to the input JSON "
instructions += "as shown in the Desired format. "
instructions += "Your response should include only the augmented JSON and nothing else. "
instructions += "Use previously assigned business concepts and business descriptions as much as possible. "


# In[479]:


desired_output = { 
                    "table_name" : "phys_db",
                    "table_context" : "Database and Infrastructure management department",
                    "columns": {
                        "Data_Space_Allocated_MB" : {
                            "data_type" : "INTEGER",
                            "size" : 4,        
                            "sample_values" : "105, 2030, 500",
                            "business_concept" : "Available Space",
                            "business_description" : "Amount of disk space that can be used expressed in MB"
                        },
                        "Data_Space_Used_MB" : {
                            "data_type" : "NUMERIC",
                            "size" : 18,        
                            "sample_values" : "55, 1705, 350",
                            "business_concept" : "Used Space",
                            "business_description" : "Used disk space expressed in MB"
                        }
                    }
                 }


# In[480]:


example = { 
            "table_name" : "phys_db",
            "table_context" : "Database and Infrastructure management department",
            "columns": {
                "Data_Space_Allocated_MB" : {
                    "data_type" : "INTEGER",
                    "size" : 4,        
                    "sample_values" : "105, 2030, 500",
                    "business_concept" : "Available Space",
                    "business_description" : "Amount of disk space that can be used expressed in MB"
                },
                "Data_Space_Used_MB" : {
                    "data_type" : "NUMERIC",
                    "size" : 18,        
                    "sample_values" : "55, 1705, 350",
                    "business_concept" : "Used Space",
                    "business_description" : "Used disk space expressed in MB"
                    }
                }
            }


# In[481]:


system_msg = "automatically creates data catalogs"


# ### Try different templates

# In[482]:


system_msg_template = "You are a helpful assistant that {system_msg}."

human_msg_template = "{instructions} "
human_msg_template += "Here is an example of how your response should be: {example}. "
human_msg_template += "Desired format: {desired_output}. "
human_msg_template += "Here are some examples of previously assigned business concepts and business descriptions: {retrieved_memories}. "
human_msg_template += "Input: ###\n"
human_msg_template += "{object_metadata}\n"
human_msg_template += "###"


# ## Create initial memory

# ### Define what to put in the memory initially

# In[432]:


# What will be in the vector database initially
initial_memory = { 
    "table_name" : "phys_db",
    "table_context" : "Database and Infrastructure management department",
    "columns": {
        "Data_Space_Allocated_MB" : {
            "data_type" : "INTEGER",
            "size" : 4,        
            "sample_values" : "105, 2030, 500",
            "business_concept" : "Available Space",
            "business_description" : "Amount of disk space that can be used expressed in MB"
        },
        "Data_Space_Used_MB" : {
            "data_type" : "NUMERIC",
            "size" : 18,        
            "sample_values" : "55, 1705, 350",
            "business_concept" : "Used Space",
            "business_description" : "Used disk space expressed in MB"
        }
    }
}

# Serializing json
json_object = json.dumps(initial_memory, indent=4)
 
# Writing to file .json
with open('memories/initial_memory.json', "w") as outfile:
    outfile.write(json_object)


# ### Create the memory

# In[445]:


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# List all the memory files
def create_memory():
    memory_directory = "/home/ubuntu/files/memories"
    memory_files = [f for f in os.listdir(memory_directory) if os.path.isfile(os.path.join(memory_directory, f))]
    
    # Memory starts empty
    memory_texts = ""
    
    # Each memory is added
    for file in memory_files:
    
        # Get the file path
        memory_file_path = os.path.join(memory_directory, file)
        # print(memory_file)
        # Get the content
        
        with open(memory_file_path,'r') as memory_file:
            memory_content = memory_file.read()
    
        # Append it to the memory
        memory_texts += " "
        memory_texts += memory_content
    
    # The entire memory text is added
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    
    texts = text_splitter.split_text(memory_texts)
    embeddings = OpenAIEmbeddings()
    memory = Chroma.from_texts(texts, embeddings)

    return memory

# Memory is initialized
memory = create_memory()


# ## Prepare the Prompt

# In[483]:


# Retrieve from memory
retrieved = memory.similarity_search_with_score(json.dumps(object_metadata))

retrieved_memories = ""
for i, v in enumerate(retrieved):
    # the similarity score: 
    retrieved_memories += v[i].page_content    
    
print(retrieved_memories)


# In[493]:


## Prepare input
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

chat = ChatOpenAI(model_name='gpt-4', temperature=1.0)

system_message_prompt = SystemMessagePromptTemplate.from_template(system_msg_template)
human_message_prompt = HumanMessagePromptTemplate.from_template(human_msg_template)

chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

input = chat_prompt.format_prompt(
        system_msg=system_msg,
        instructions=instructions,
        example=example,
        desired_output=desired_output,
        retrieved_memories=retrieved_memories,
        object_metadata=object_metadata        
    ).to_messages()

print(input[1].content)


# In[485]:


# Contract output
input[1].content = " ".join(input[1].content.split())
print(input[1].content)


# ## Prompt the LLM

# In[488]:


response = chat(input)
output=response.content
print(output)


# ## Add output to the memory in Redis

# In[451]:


# Writing to file .json
import time
ts = time.time()
with open(f"memories/{object_metadata['table_name']}_{ts}.json", "w") as outfile:
    outfile.write(output)


# ## Refresh the memory

# In[454]:


memory = create_memory()


# ## Use the Output

# In[489]:


import json
output = output.replace("\'", "\"")
data = json.loads(output)
print(json.dumps(data, indent=4))

