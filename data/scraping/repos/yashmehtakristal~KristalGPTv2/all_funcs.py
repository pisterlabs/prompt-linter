#!/usr/bin/env python
# coding: utf-8

# All imports

import fitz
from pprint import pprint
import camelot
import PyPDF2
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.chains import create_extraction_chain
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import CSVLoader
from langchain.llms import OpenAI
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chains.openai_functions.utils import (
    _convert_schema,
    _resolve_schema_references,
    get_llm_kwargs,
)
from langchain.output_parsers.openai_functions import (
    JsonKeyOutputFunctionsParser,
    PydanticAttrOutputFunctionsParser,
)
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from llama_index import Document, SummaryIndex
from llama_index import VectorStoreIndex, ServiceContext, LLMPredictor
from llama_index.query_engine import PandasQueryEngine, RetrieverQueryEngine
from llama_index.retrievers import RecursiveRetriever
from llama_index.schema import IndexNode
from llama_index.llms import OpenAI
from llama_hub.file.pymu_pdf.base import PyMuPDFReader
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer
import pandas as pd
import os
import time
from typing import Any, List, Optional
from pathlib import Path
import pickle
import openai
from contextlib import redirect_stdout
import io
import warnings
warnings.filterwarnings("ignore")

from tenacity import retry, stop_after_attempt, wait_random_exponential

# OpenAI API key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai.api_key = OPENAI_API_KEY
openai_api_key = OPENAI_API_KEY


# Function to read documents in particular directory 
def read_documents(directory):
    '''
    read_documents:
    Explanation: Reads the pdf documents in a particular directory

    Input - 
    directory: the directory/folder of where the pdf documents are located. Default value is "Documents"

    Output -
    docs: list of documents (contains data of each of the documents)
    '''

    reader = PyMuPDFReader()

    # Initialize docs as an empty list (contains data of each of the documents)
    docs = []

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            
            # Load the file and append the data to docs
            docs.append(reader.load(file_path))

    # Return list of documents (contains data of each of the documents)
    return docs

# Call read_documents function
docs = read_documents(directory = "Documents")


# Function to get tables from particular directory 
def get_tables(path: str):

    '''
    get_tables:  
    Explanation: Obtain tables from particular directory

    Input - 
    path: the actual path to the file (including filename [not extension])

    Output - 
    table_dfs: list containing dataframe of various tables
    '''

    # Dataframe of the tables from the document we will extract
    table_dfs = []

    with open(path, 'rb') as file:
        doc = fitz.open(path)
        num_pages = doc.page_count

    # Loop through all pages
    for page in range(0, num_pages):
        
        start_time = time.time()
                
        current_page = doc[page]
        
        # Read pdf to extract tables from that specific page
        table_object = current_page.find_tables()
        
        # This uses the tables attribute of object named TableFinder - Gives list of tables
        table_list = table_object.tables
        
        non_empty_table_counter = 0
        empty_table_counter = 0
        
        # If table_list is empty
        if len(table_list) == 0:
            pass

        else:            
            
            for table in table_list:
                if table.to_pandas().empty:

                    empty_table_counter += 1

                    elapsed_time_for_empty_table = time.time() - start_time

                else:

                    non_empty_table_counter += 1

                    table_df = table.to_pandas()
                    
                    table_df = (
                        table_df.rename(columns=table_df.iloc[0])
                        .drop(table_df.index[0])
                        .reset_index(drop=True)
                    )

                    # Append to list
                    table_dfs.append(table_df)

                    elapsed_time_for_table = time.time() - start_time
    
    # return table_dfs (list containing dataframe of various tables)
    return table_dfs


# iterate over files in that directory & remove all "\n" characters from the dataframe
# Note this is where the above function (get_tables is called)
def iterate_files_directory(directory):
    '''
    iterate_files_directory:  
    Explanation: iterate over files in that directory & remove all "\n" characters from the dataframe
    Explanation (contd): This is where the above function (get_tables) is called

    Input - 
    directory: the directory/folder to where our files are stored

    Output - 
    table_dfs: list containing dataframe of various tables
    '''

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            
            # Call get_tables function
            table_dfs = get_tables(file_path)


    # Remove all "\n" characters from dataframe
    for i in range(len(table_dfs)):
        table_dfs[i] = table_dfs[i].replace('\n','', regex=True)

    # return table_dfs (list containing dataframe of various tables)
    return table_dfs

# Call iterate_files_directory function
table_dfs = iterate_files_directory(directory = "Documents")


# Save to pickle file
def save_to_pickle(directory_pickles, table_dfs):
    '''
    save_to_pickle:  
    Explanation: Save the table_dfs into a pickle file (for easy saving and loading)

    Input - 
    directory_pickles: the directory/folder to where our files are stored

    Output - 
    table_dfs: list containing dataframe of various tables
    '''

    # Open the file in binary write mode ('wb') and save the list
    with open(directory_pickles, 'wb') as file:
        
        # Dump the list into a pickle object
        pickle.dump(table_dfs, file)

    return directory_pickles

# Call function to save the table_dfs to pickle file
directory_pickles = save_to_pickle(directory_pickles = "Pickle/table_dfs.pkl", table_dfs = table_dfs)


# Load from pickle file
def load_from_pickle(directory_pickles):
    '''
    load_from_pickle:  
    Explanation: This will load the pickle file 

    Input - 
    directory_pickles: the directory/folder to where our files are stored

    Output - 
    table_dfs: list containing dataframe of various tables
    '''

    # Load the DataFrame from the pickle file
    table_dfs = pd.read_pickle(directory_pickles)

    return table_dfs

# Call function to load from pickle file - Uncomment this code if you want to load from pickle file
# table_dfs = load_from_pickle(directory_pickles = "Pickle/table_dfs.pkl")


# Defining query engine over tables
def query_engine(table_dfs):
    '''
    query_engine: This function defines the llm, service context object and df_query_engines

    Input - 
    table_dfs: list containing dataframe of various tables

    Output - 
    llm, service context, df_query_engines: The respective defined objects
    '''
    # GPT 4 Model used: "gpt-4-0613"
    # GPT 3.5 Model used: 
    llm = ChatOpenAI(model="gpt-3.5-turbo", request_timeout = 120, max_retries = 6)

    # Create a service context object
    service_context = ServiceContext.from_defaults(llm=llm)

    # Create a query engine for each table in the list of table dataframes
    df_query_engines = [
        PandasQueryEngine(table_df, service_context = service_context)
        for table_df in table_dfs
    ]

    # Returns the llm, service context, and query engine
    return llm, service_context, df_query_engines

# Calling query_engine function
llm, service_context, df_query_engines = query_engine(table_dfs = table_dfs)


### Build Vector Index
def build_vector_index(service_context, df_query_engines, docs, nodes_to_retrieve):
    '''
    build_vector_index: This function ultimately builds the vector index for each of the documents

    Input - 
    service_context: service_context object defined above
    df_query_engines: Query engine for each table in list of tables dataframe
    docs: A list of documents
    nodes_to_retrieve: Number of nodes to retrieve from vector_retriever

    Output - 
    vector_index: vector_index object created 
    vector_retriever: Top 3 nodes of vector index
    df_id_query_engine_mapping: Mapping of the query engine with each dataframe
    '''

    doc_nodes = []

    for doc in docs:        
        doc_nodes.extend(service_context.node_parser.get_nodes_from_documents(doc))

    summaries = [
        "This node provides information stored in the tables in the PDFs. Information could be anything about the financial product.",
    ]

    df_nodes = [
        IndexNode(text=summary, index_id=f"pandas{idx}")
        for idx, summary in enumerate(summaries)
    ]

    df_id_query_engine_mapping = {
        f"pandas{idx}": df_query_engine
        for idx, df_query_engine in enumerate(df_query_engines)
    }

    # Creating vector index and retrieving top 3 nodes (this code will consume the most time)
    vector_index = VectorStoreIndex(doc_nodes + df_nodes)
    vector_retriever = vector_index.as_retriever(similarity_top_k = nodes_to_retrieve)

    return vector_index, vector_retriever, df_id_query_engine_mapping, nodes_to_retrieve

vector_index, vector_retriever, df_id_query_engine_mapping, nodes_to_retrieve = build_vector_index(service_context = service_context, df_query_engines = df_query_engines, docs = docs, nodes_to_retrieve = 3)

def iterate_files(excel_directory, input_excel_file, file_extension):
    '''
    iterate_files: This function ultimately reads prompts from the excel file

    Input - 
    excel_directory: Directory storing excel files
    input_excel_file: Excel file name
    file_extension: Extension of excel file

    Output - 
    excel_file: The dataframe of the first sheet in the results excel file
    info_excel_file: The dataframe of sheet name, "Info", in the results excel file
    '''

    for filename in os.listdir(excel_directory):

        if filename.endswith(f".{file_extension}"):

            name_without_extension = os.path.splitext(filename)[0]

            if name_without_extension == "results":
                
                file_path = os.path.join(excel_directory, filename)
                
                excel_file = pd.read_excel(f"{excel_directory}/{input_excel_file}.{file_extension}")
                
                info_excel_file = pd.read_excel(f"{excel_directory}/{input_excel_file}.{file_extension}", sheet_name='Info')

                excel_file.dropna(axis=1, how='all', inplace=True)

                info_excel_file.dropna(axis=0, how='all').dropna(axis=1, how='all')


    # return the first sheet of the excel file and info sheet of excel file as pandas dataframe 
    return excel_file, info_excel_file


# Call function for directory results, input file results and xlsx file extension
orignal_excel_file, info_excel_file = iterate_files(excel_directory = "Results", input_excel_file = "results", file_extension = "xlsx")

 
def conditions_excel(orignal_excel_file):
    '''
    conditions_excel: Checking for certain conditions and creating a filtered dataframe

    Input - 
    orignal_excel_file: Dataframe of results excel file

    Output - 
    LLM_inputs: Displays rows of orignal_excel_file, where source type column is equal to LLM
    info_excel_file: Displays rows of orignal_excel_file, where source type column is equal to Discretionary
    '''

    LLM_inputs = orignal_excel_file[orignal_excel_file["Source Type"] == "LLM"]

    Discretionary_inputs = orignal_excel_file[orignal_excel_file["Source Type"] == "Discretionary"]

    return LLM_inputs, Discretionary_inputs


# Call conditions_excel function
LLM_inputs, Discretionary_inputs = conditions_excel(orignal_excel_file)


# Function to extract fund variable
def extract_fund_variable(info_excel_file):
    '''
    extract_fund_variable: This function extracts the fund variable

    Input - 
    info_excel_file: Dataframe of the info sheet of results excel file

    Output - 
    fund_variable: Fund variable that was extracted from info sheet of results excel file
    '''
    
    for index, row in info_excel_file.iterrows():

        if "Fund variable" in row.values:

            date_index = list(row).index("Fund variable")

            fund_variable = row[date_index + 1]

    # Return fund_variable
    return fund_variable

# Call function to obtain fund variable
fund_variable = extract_fund_variable(info_excel_file = info_excel_file)


# Define function to obtain the prompts where we substitute variable name
# This code should ultimately create a new column, "Automatic Processed Input Prompt"
def prompts_to_substitute_variable(orignal_excel_file, fund_variable, LLM_inputs):
    '''
    prompts_to_substitute_variable: This function creates a new column, "Automatic Processed Input Prompt" and writes the prompt result there.

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    fund_variable: Fund variable that was extracted from info sheet of results excel file
    LLM_inputs: Displays rows of orignal_excel_file, where source type column is equal to LLM

    Output -
    orignal_excel_file: Dataframe of the results excel file
    llm_full_index: List of index of rows where "Source Type" column is equal to LLM
    '''
    
    variable_replace = orignal_excel_file['Variable replace'] == 'Yes'
    prompt_values = orignal_excel_file.loc[variable_replace, 'Input prompt'].tolist()
    prompt_indices = orignal_excel_file.loc[variable_replace].index.tolist()

    new_prompt_values = []

    for prompt in prompt_values:
        
        modified_prompt = prompt.replace("fund", fund_variable + " fund")
        new_prompt_values.append(modified_prompt)

    orignal_excel_file.loc[prompt_indices, 'Automatic Processed Input prompt'] = new_prompt_values

    llm_full_index = LLM_inputs.index.tolist()

    rest_of_index = [x for x in llm_full_index if x not in prompt_indices]

    orignal_excel_file.loc[rest_of_index, 'Automatic Processed Input prompt'] = orignal_excel_file.loc[rest_of_index, 'Input prompt']

    excel_columns = orignal_excel_file.columns.tolist() 
    excel_columns.remove('Automatic Processed Input prompt') 
    excel_columns.insert(excel_columns.index('Variable replace'), 'Automatic Processed Input prompt') 
    orignal_excel_file = orignal_excel_file[excel_columns] 


    return orignal_excel_file, llm_full_index


# Call function to run prompts_to_substitute_variable function
orignal_excel_file, llm_full_index = prompts_to_substitute_variable(orignal_excel_file = orignal_excel_file, fund_variable = fund_variable, LLM_inputs = LLM_inputs)


def storing_input_prompt_in_list(orignal_excel_file, llm_full_index):
    '''
    storing_input_prompt_in_list: This function creates a list of prompts that we pass into our LLM

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    llm_full_index: List of index of rows where "Source Type" column is equal to LLM

    Output -
    orignal_excel_file: Dataframe of the results excel file
    llm_prompts_to_use: The list prompts that we pass into our LLM (filtered for NA values in rows where Source Type = LLM even)
    llm_prompts_index: Index of the prompts that we have passed to our LLM
    '''

    llm_index_len = len(llm_full_index)

    processed_input_prompts = orignal_excel_file["Automatic Processed Input prompt"]

    non_nan_indices = processed_input_prompts.notna()
    non_nan_values = processed_input_prompts[non_nan_indices]
    llm_prompts_index = non_nan_indices[non_nan_indices].index.tolist()

    # These are the processed input prompts in a list format to use as input to our query engine
    llm_prompts_to_use = non_nan_values.tolist()

    # Return the llm_prompts_to_use
    return orignal_excel_file, llm_prompts_to_use, llm_prompts_index

orignal_excel_file, llm_prompts_to_use, llm_prompts_index = storing_input_prompt_in_list(orignal_excel_file = orignal_excel_file, llm_full_index = llm_full_index)


def recursive_retriever(orignal_excel_file, vector_retriever, df_id_query_engine_mapping, service_context):
    '''
    recursive_retriever: This function uses recursive retriever in our RetrieverQueryEngine

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    vector_retriever: Top 3 nodes of vector index
    df_id_query_engine_mapping: Mapping of the query engine with each dataframe
    service_context: service_context object defined above

    Output -
    recursive_retriever: Instance of RecursiveRetriever class
    response_synthesizer: Output of get_response_synthesizer
    query_engine: Instance of Retriever Query Engine class
    '''

    recursive_retriever = RecursiveRetriever(
    "vector", 
    retriever_dict={"vector": vector_retriever}, 
    query_engine_dict = df_id_query_engine_mapping, 
    verbose = False,
    )

    response_synthesizer = get_response_synthesizer(
        service_context=service_context, 
        response_mode="no_text"
    )    

    query_engine = RetrieverQueryEngine.from_args(
        recursive_retriever, response_synthesizer = response_synthesizer
    )

    return recursive_retriever, response_synthesizer, query_engine


recursive_retriever, response_synthesizer, query_engine = recursive_retriever(orignal_excel_file = orignal_excel_file, vector_retriever = vector_retriever, df_id_query_engine_mapping = df_id_query_engine_mapping, service_context = service_context)

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def individual_prompt(query_engine, prompt):
    '''
    individual_prompt: This function runs for a single prompt and displays the output result

    Input - 
    query_engine: An instance of the Retriever Query Engine class 
    prompt: The prompt inputted by the user 

    Output -
    final_output: The output of the prompt by the LLM
    '''

    start_time = time.time()

    response = query_engine.query(prompt)

    # Calculate elapsed time for this iteration with an empty table
    calculate_elapsed_time = time.time() - start_time

    # The final output from LLM
    final_output = str(response)

    return final_output


@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def prompt_loop(query_engine, llm_prompts_to_use):
    '''
    prompt_loop: This function runs a loop by inputting multiple prompts (from list llm_prompts_to_use) and stores the output 

    Input - 
    query_engine: An instance of the Retriever Query Engine class 
    llm_prompts_to_use: List of input prompts to LLM 

    Output -
    output_response: List containing response of prompts passed to LLM
    output_context: List containing context of the response of prompts passed to LLM
    '''

    output_response = []
    output_context = []
    count = 1

    for prompt in llm_prompts_to_use:
        
        response = query_engine.query(f"{prompt}")
        output_context.append(response)
        
        output_response.append(str(response))
        
        count += 1
        
        # Wait 8 seconds before executing next prompt
        time.sleep(8)

    return output_response, output_context

output_response, output_context = prompt_loop(query_engine = query_engine, llm_prompts_to_use = llm_prompts_to_use)


def create_output_result_column(orignal_excel_file, llm_prompts_index, output_response):
    '''
    create_output_result_column: This function ultimately creates the "Output Result" column

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    llm_prompts_index: List of the index of the rows of prompts (in orignal_excel_file) that were fed to LLM
    output_response: List containing response of prompts passed to LLM

    Output -
    orignal_excel_file: Dataframe of the results excel file
    '''
    
    # Create new column "Output result" which adds the output prompts from list
    orignal_excel_file.loc[llm_prompts_index, 'Output result'] = output_response

    # Rearrange the columns so that 'Automatic Processed Input prompt' is in front of 'Manual Processed Input prompt'
    excel_columns = orignal_excel_file.columns.tolist()
    excel_columns.remove('Output result')  
    excel_columns.insert(excel_columns.index('Source Type'), 'Output result')
    orignal_excel_file = orignal_excel_file[excel_columns] 

    return orignal_excel_file

# Call function to create output result column
orignal_excel_file = create_output_result_column(orignal_excel_file = orignal_excel_file, llm_prompts_index = llm_prompts_index, output_response = output_response)


def create_output_context_column(orignal_excel_file, llm_prompts_index, nodes_to_retrieve):
    '''
    create_output_context_column: This function ultimately creates the "Output Context" column

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    llm_prompts_index: List of the index of the rows of prompts (in orignal_excel_file) that were fed to LLM 
    nodes_to_retrieve: Number of nodes to retrieve from vector_retriever

    Output -
    orignal_excel_file: Dataframe of the results excel file
    '''

    new_output = orignal_excel_file.loc[llm_prompts_index, 'Field name']

    field_name_output_list = new_output.tolist()

    output_dict = dict(zip(field_name_output_list, output_context))

    output_prompt_context_value_list = []

    for key, value in output_dict.items():
        
        output_prompt_object = value

        combined_context = ""

        for i in range(nodes_to_retrieve):

            output_prompt_context_value = output_prompt_object.source_nodes[i].node.get_content()

            if combined_context:
                combined_context += '\n'
            
            combined_context += output_prompt_context_value

        # Append the combined context (of a single prompt) to the list
        output_prompt_context_value_list.append(combined_context)


    # Take each context and add appropriate line spacing if there is "\n" present
    for index, value in enumerate(output_prompt_context_value_list):

        lines = value.splitlines()

        result_string = '\n'.join(lines)

        output_prompt_context_value_list[index] = result_string

    orignal_excel_file.loc[llm_prompts_index, 'Output context'] = output_prompt_context_value_list

    return orignal_excel_file


# Call create_output_context_column function
orignal_excel_file = create_output_context_column(orignal_excel_file, llm_prompts_index, nodes_to_retrieve = nodes_to_retrieve)


def intermediate_output_to_excel(orignal_excel_file, excel_directory, output_excel_filename, file_extension):
    '''
    intermediate_output_to_excel: This is an intermediary function outputting the excel file [Should stop here if you only want to fill the first document (FAF form)]

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    excel_directory: Directory storing excel files
    output_excel_filename: Excel file name that we will output
    file_extension: Extension of the file that we wish output file will be 

    Output - None
    '''
    output_excel_file = f"{output_excel_filename}.{file_extension}"
    excel_file_path = os.path.join(excel_directory, output_excel_file)
    orignal_excel_file.to_excel(excel_file_path, index=True)


# Call function output_files for directory results, input file results and xlsx file extension
intermediate_output_to_excel(orignal_excel_file = orignal_excel_file, excel_directory = "Results", output_excel_filename = "results_output", file_extension = "xlsx")


def create_schema_from_excel(orignal_excel_file, llm_prompts_index):
    '''
    create_schema_from_excel: This function will automatically create a schema based on the "Field Name" and "Data Type" columns in results_output.xlsx

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    llm_prompts_index: List of the index of the rows of prompts (in orignal_excel_file) that were fed to LLM 

    Output - None
    '''

    filtered_df = orignal_excel_file.iloc[llm_prompts_index]

    schema = {"properties": {}}

    for index, row in filtered_df.iterrows():

        field_name = row["Field name"]
        data_type = row["Data Type"]

        property_dict = {"type": data_type}
        
        schema["properties"][field_name] = property_dict
        
    # Return the schema generated
    return schema 

# Call function create_schema_from_excel
schema = create_schema_from_excel(orignal_excel_file, llm_prompts_index)


def _get_extraction_function(entity_schema: dict) -> dict:
    '''
    _get_extraction_function: This is the information_extraction function returning a dictionary

    Input - 
    entity_schema: Takes the entity schema dictionary as input

    Output - 
    Below dictionary
    '''
    
    return {
        "name": "information_extraction",
        "description": "Extracts the relevant information from the passage.",
        "parameters": {
            "type": "object",
            "properties": {
                "info": {"type": "array", "items": _convert_schema(entity_schema)} # calling _convert_schema function from langchain 
            },
            "required": ["info"],
        },
    }


def create_extraction_chain(
    schema: dict,
    llm: BaseLanguageModel,
    prompt: Optional[BasePromptTemplate] = None,
    verbose: bool = False,
) -> Chain:
    
    """
    Create_extraction_chain: Creates a chain that extracts information from a passage.

    Input - 
        schema: The schema of the entities to extract.
        llm: The language model to use.
        prompt: The prompt to use for extraction.
        verbose: Whether to run in verbose mode. In verbose mode, some intermediate
            logs will be printed to the console. Defaults to the global `verbose` value,
            accessible via `langchain.globals.get_verbose()`.

    Output - 
        Chain that can be used to extract information from a passage.
    """
    
    # Call _get_extraction_function which returns a dictionary 
    function = _get_extraction_function(schema)

    # Extraction template that user enters
    # Note: recommended you keep here 'Passage: {input}' and the extraction template as follows as well

    _EXTRACTION_TEMPLATE = """Extract and save the relevant entities mentioned\
    in the following passage together with their properties.

    Only extract the properties mentioned in the 'information_extraction' function.

    If a property is not present and is not required in the function parameters, do not include it in the output.

    If output is a Date then change it to dd/mm/yyyy format.

    Passage:
    {input}
    """ 
        
    extraction_prompt = prompt or ChatPromptTemplate.from_template(_EXTRACTION_TEMPLATE)
    
    output_parser = JsonKeyOutputFunctionsParser(key_name="info")
    
    llm_kwargs = get_llm_kwargs(function)
    
    # Construct the LLMChain
    chain = LLMChain(
        llm=llm,
        prompt = extraction_prompt,
        llm_kwargs=llm_kwargs,
        output_parser=output_parser,
        verbose=verbose,
    )
    
    # Return the chain
    return chain


def parse_value(output_response, llm_prompts_index):
    '''
    parse_value: This function will take the values in column, "Output Result", feed it to parser and generate a list of key-value pairs in the dictionary

    Input - 
    output_response: List containing response of prompts passed to LLM
    llm_prompts_index: List of the index of the rows of prompts (in orignal_excel_file) that were fed to LLM 

    Output - 
    orignal_excel_file: Dataframe of the results excel file
    '''

    final_output_value = []

    for output_value in output_response:
        
        # Create chain
        chain = create_extraction_chain(schema, llm)
        chain_result = chain.run(output_value)
        final_output_value.append(chain_result)

    # Iterate through llm_prompts_index and assign values from final_output_value to a new column, "Final Output result"
    for index, info_dict in zip(llm_prompts_index, final_output_value):
        orignal_excel_file.at[index, 'Final Output result'] = info_dict
        
    return orignal_excel_file

# Call function parse_value
orignal_excel_file = parse_value(output_response = output_response, llm_prompts_index = llm_prompts_index)


def create_filtered_excel_file(orignal_excel_file, llm_prompts_index):
    '''
    create_filtered_excel_file: This function will just create a new dataframe called filtered_excel_file (for safety) containing those prompts passed to LLM (Source Type = LLM, and not NA)

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    llm_prompts_index: List of the index of the rows of prompts (in orignal_excel_file) that were fed to LLM 

    Output - 
    filtered_excel_file: New dataframe containing those prompts passed to LLM (Source Type = LLM, and not NA)
    '''

    # Creating new filtered_excel_file for safety of the data, so far - this is based on llm_prompts_index (where Source Type == LLM AND Excluding NA in Input prompt)
    filtered_excel_file = orignal_excel_file.iloc[llm_prompts_index] 

    return filtered_excel_file

# Calling create_filtered_excel_file function
filtered_excel_file = create_filtered_excel_file(orignal_excel_file = orignal_excel_file, llm_prompts_index = llm_prompts_index)


# Define a function to create 'Final Output result_cleaned' column
def create_final_output_result_cleaned_column(row):
    '''
    create_final_output_result_cleaned_column: This function will extract the value of the corresponding key, if it is equal to field name

    Input:
    row - Each row of the filtered_excel_file dataframe (containing llm_prompts_index only)

    Output:
    For that particular row, it will provide the value in the final_output_result_cleaned column
    '''
    
    field_name = row['Field name']
    final_output = row["Final Output result"]
    extracted_values = []
    
    if final_output:

        if isinstance(final_output, list):

            for dictionary in final_output:
                if field_name in dictionary:
                    extracted_values.append(dictionary[field_name])

        elif isinstance(final_output, dict):
            if field_name in final_output:
                extracted_values.append(final_output[field_name])
            
    if extracted_values:
        return extracted_values[0]
    
    return None

def final_result_orignal_excel_file(filtered_excel_file):
    '''
    final_result_orignal_excel_file: This function will simply call previous function, copy the column (Final Output result_cleaned) to orignal_excel_file 

    Input:
    filtered_excel_file: New dataframe containing those prompts passed to LLM (Source Type = LLM, and not NA)

    Output:
    Orignal_excel_file: Dataframe of the results excel file (with the newly added 'Final Output result_cleaned' column)
    '''

    # Apply the function (create_final_output_result_cleaned_column) to each row (axis = 1) and store in a new column, 'Final Output result_cleaned'
    filtered_excel_file['Final Output result_cleaned'] = filtered_excel_file.apply(create_final_output_result_cleaned_column, axis=1)

    # Update orignal_excel_file for specific indexes according to filtered_excel_file
    orignal_excel_file.loc[llm_prompts_index, 'Final Output result_cleaned'] = filtered_excel_file['Final Output result_cleaned']

    return orignal_excel_file

# Call final_result_orignal_excel_file function
orignal_excel_file = final_result_orignal_excel_file(filtered_excel_file = filtered_excel_file)


def reordering_columns(orignal_excel_file):
    '''
    reordering_columns: This function will reorder columns of orignal_excel_file according to the columns  

    Input:
    filtered_excel_file: New dataframe containing those prompts passed to LLM (Source Type = LLM, and not NA)

    Output:
    Orignal_excel_file: Dataframe of the results excel file (with the newly added 'Final Output result_cleaned' column)
    '''

    # Defining new column order of dataframe
    new_column_order = ['Field name', 'Data Type', 'Input prompt', 'Source Type', 'Variable replace', 'Automatic Processed Input prompt', 'Output context', 'Output result', 'Final Output result', 'Final Output result_cleaned']

    # Renaming columns
    orignal_excel_file = orignal_excel_file[new_column_order]

    return orignal_excel_file
    
# Call reordering_columns function 
orignal_excel_file = reordering_columns(orignal_excel_file)

def find_result_fund_name(orignal_excel_file):
    '''
    find_result_fund_name: This function will find the value from 'Final Output result_cleaned' column where field name is fund name.

    Input - 
    orignal_excel_file: Dataframe of the results excel file

    Output - 
    results_fund_name_value: Variable containing fund name
    '''

    result_fund_name = orignal_excel_file[orignal_excel_file['Field name'] == 'Fund Name']

    if not result_fund_name.empty:

        series_output = result_fund_name['Final Output result_cleaned']
        results_fund_name_value = series_output.str.cat(sep=' ')

    return results_fund_name_value


def find_result_fund_house(orignal_excel_file):
    '''
    find_result_fund_house: This function will find the value from 'Final Output result_cleaned' column where field name is fund house.

    Input - 
    orignal_excel_file: Dataframe of the results excel file

    Output - 
    results_fund_house_value: Variable containing fund house
    '''

    result_fund_house = orignal_excel_file[orignal_excel_file['Field name'] == 'Fund House']

    if not result_fund_house.empty:
        
        series_output = result_fund_house['Final Output result_cleaned']

        result_fund_house_value = series_output.str.cat(sep=' ')

    return result_fund_house_value

def find_result_fund_class(orignal_excel_file):
    '''
    find_result_fund_class: This function will find the value from 'Final Output result_cleaned' column where field name is fund class.

    Input - 
    orignal_excel_file: Dataframe of the results excel file

    Output - 
    result_fund_class_value: Variable containing fund class
    '''

    result_class = orignal_excel_file[orignal_excel_file['Field name'] == 'Class']

    if not result_class.empty:
        
        series_output = result_class['Final Output result_cleaned']

        result_fund_class_value = series_output.str.cat(sep=' ')

    return result_fund_class_value

def find_result_currency(orignal_excel_file):
    '''
    find_result_currency: This function will find the value from 'Final Output result_cleaned' column where field name is currency.

    Input - 
    orignal_excel_file: Dataframe of the results excel file

    Output - 
    result_currency_value: Variable containing currency
    '''

    result_currency = orignal_excel_file[orignal_excel_file['Field name'] == 'Currency']

    if not result_currency.empty:
        
        series_output = result_currency['Final Output result_cleaned']
        result_currency_value = series_output.str.cat(sep=' ')

    return result_currency_value


def find_result_acc_or_inc(orignal_excel_file):

    '''
    find_result_currency: This function will find the value from 'Final Output result_cleaned' column where field name is "acc or inc".

    Input - 
    orignal_excel_file: Dataframe of the results excel file

    Output - 
    result_acc_or_inc: Variable containing acc or inc
    '''

    result_acc_or_inc = orignal_excel_file[orignal_excel_file['Field name'] == 'Acc or Inc']

    if not result_acc_or_inc.empty:
        
        series_output = result_acc_or_inc['Final Output result_cleaned']
        result_acc_or_inc_value = series_output.str.cat(sep=' ')

    return result_acc_or_inc_value

# Call all the previous functions
results_fund_name_value = find_result_fund_name(orignal_excel_file)
result_fund_house_value = find_result_fund_house(orignal_excel_file)
result_fund_class_value = find_result_fund_class(orignal_excel_file)
result_currency_value = find_result_currency(orignal_excel_file)
result_acc_or_inc_value = find_result_acc_or_inc(orignal_excel_file)

def create_new_kristal_alias(results_fund_name_value, result_fund_house_value, result_fund_class_value, result_currency_value, result_acc_or_inc_value):
    '''
    create_new_kristal_alias: Create kristal alias by concatenating all the variables names.

    Input - 
    results_fund_name_value: Variable containing string of fund name
    result_fund_house_value: Variable containing string of fund house
    result_fund_class_value: Variable containing string of fund class 
    result_currency_value: Variable containing string of currency
    result_acc_or_inc_value: Variable containing string of "acc or inc" value

    Output - 
    kristal_alias: Variable containing concatenated kristal alias
    '''

    kristal_alias = f"{result_fund_house_value} {results_fund_name_value} - {result_fund_class_value} ({result_currency_value}) {result_acc_or_inc_value}"

    return kristal_alias

# Storing kristal_alias variable
kristal_alias = create_new_kristal_alias(results_fund_name_value, result_fund_house_value, result_fund_class_value, result_currency_value, result_acc_or_inc_value)

def update_kristal_alias(orignal_excel_file, kristal_alias):
    '''
    update_kristal_alias: Function for updating the kristal alias value in the orignal_excel_file dataframe

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    kristal_alias: Variable containing updated kristal alias

    Output - 
    orignal_excel_file: Dataframe of the results excel file (with updated kristal alias value)
    '''
    index_kristal_alias = orignal_excel_file.index[orignal_excel_file['Field name'] == 'Kristal Alias'].tolist()
    orignal_excel_file.loc[index_kristal_alias, "Final Output result_cleaned"] = kristal_alias

    return orignal_excel_file

# Calling update_kristal_alias function to update orignal_excel_file
orignal_excel_file = update_kristal_alias(orignal_excel_file = orignal_excel_file, kristal_alias = kristal_alias)


def update_sponsored_by(orignal_excel_file, sponsored_by):
    '''
    update_sponsored_by: Function for updating the kristal alias value in the orignal_excel_file dataframe

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    sponsored_by: Variable containing updated sponsored by value

    Output - 
    orignal_excel_file: Dataframe of the results excel file (with updated sponsored by value)
    '''

    index_sponsored_by = orignal_excel_file.index[orignal_excel_file['Field name'] == 'Sponsored By'].tolist()
    orignal_excel_file.loc[index_sponsored_by, "Final Output result_cleaned"] = sponsored_by

    return orignal_excel_file

# Calling update_sponsored_by function to update orignal_excel_file
orignal_excel_file = update_sponsored_by(orignal_excel_file = orignal_excel_file, sponsored_by = "backend-staging+hedgefunds@kristal.ai")


def update_required_broker(orignal_excel_file, required_broker):
    '''
    update_required_broker: Function for updating the kristal alias value in the orignal_excel_file dataframe

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    required_broker: Variable containing updated required broker value

    Output - 
    orignal_excel_file: Dataframe of the results excel file (with updated required broker value)
    '''

    index_required_broker = orignal_excel_file.index[orignal_excel_file['Field name'] == 'Required Broker'].tolist()
    orignal_excel_file.loc[index_required_broker, "Final Output result_cleaned"] = required_broker

    return orignal_excel_file

# Calling update_sponsored_by function to update orignal_excel_file
orignal_excel_file = update_required_broker(orignal_excel_file = orignal_excel_file, required_broker = "Kristal Pooled")


def update_transactional_fund(orignal_excel_file, transactional_fund):
    '''
    update_transactional_fund: Function for updating the kristal alias value in the orignal_excel_file dataframe

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    required_broker: Variable containing updated required broker value

    Output - 
    orignal_excel_file: Dataframe of the results excel file (with updated required broker value)
    '''

    index_transactional_fund = orignal_excel_file.index[orignal_excel_file['Field name'] == 'Transactional Fund'].tolist()
    orignal_excel_file.loc[index_transactional_fund, "Final Output result_cleaned"] = transactional_fund

    return orignal_excel_file


# Calling update_transactional_fund function to update orignal_excel_file
orignal_excel_file = update_transactional_fund(orignal_excel_file = orignal_excel_file, transactional_fund = "Yes")


def update_disclaimer(orignal_excel_file, disclaimer):
    '''
    update_disclaimer: Function for updating the disclaimer in the orignal_excel_file dataframe

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    transactional_fund: Variable containing updated required broker value

    Output - 
    orignal_excel_file: Dataframe of the results excel file (with updated required broker value)
    '''

    index_result_disclaimer = orignal_excel_file.index[orignal_excel_file['Field name'] == 'Disclaimer'].tolist()
    orignal_excel_file.loc[index_result_disclaimer, "Final Output result_cleaned"] = disclaimer

    return orignal_excel_file


# Calling update_disclaimer function to update orignal_excel_file
orignal_excel_file = update_disclaimer(
    orignal_excel_file = orignal_excel_file,
    disclaimer = '''
    The recommendations contained herein are for the exclusive use of investor and prohibits any form of disclosure or reproduction. The content cannot be relied upon by any other person for any other purpose. The recommendations are preliminary information to the investors, are subject to risks and may change based on investment objectives, financials, liabilities or the risk profile of an investor. Any recommendations including financial advice provided by Kristal.AI or its affiliates shall be subject to contractual understanding, necessary documentation, applicable laws, approvals and regulations. The recommendations contained herein may not be eligible for sale/purchase in some jurisdictions, in specific, are not intended for residents of the USA or within the USA.Though the recommendations are based on information obtained from reliable sources and are provided in good faith, they may be valid only on the date and time the recommendations are provided and shall be subject to change without notice. Kristal.AI
    '''
    )


def update_risk_disclaimer(orignal_excel_file, risk_disclaimer):
    '''
    update_risk_disclaimer: Function for updating the risk disclaimer in the orignal_excel_file dataframe

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    transactional_fund: Variable containing updated required broker value

    Output - 
    orignal_excel_file: Dataframe of the results excel file (with updated required broker value)
    '''

    index_kristal_risk_disclaimer = orignal_excel_file.index[orignal_excel_file['Field name'] == 'Risk Disclaimer'].tolist()
    orignal_excel_file.loc[index_kristal_risk_disclaimer, "Final Output result_cleaned"] = risk_disclaimer

    return orignal_excel_file

# Calling update_risk_disclaimer function to update orignal_excel_file
orignal_excel_file = update_risk_disclaimer(
    orignal_excel_file = orignal_excel_file,
    result_kristal_risk_disclaimer = '''
    The recommendations contained herein are for the exclusive use of investor and prohibits any form of disclosure or reproduction. The content cannot be relied upon by any other person for any other purpose. The recommendations are preliminary information to the investors, are subject to risks and may change based on investment objectives, financials, liabilities or the risk profile of an investor. Any recommendations including financial advice provided by Kristal.AI or its affiliates shall be subject to contractual understanding, necessary documentation, applicable laws, approvals and regulations. The recommendations contained herein may not be eligible for sale/purchase in some jurisdictions, in specific, are not intended for residents of the USA or within the USA.Though the recommendations are based on information obtained from reliable sources and are provided in good faith, they may be valid only on the date and time the recommendations are provided and shall be subject to change without notice. Kristal.AI
    '''
    )

def find_nav_value(orignal_excel_file):
    '''
    find_nav_value: Function to find NAV value in "Final Output result_cleaned" where "Field name" is "NAV"

    Input - 
    orignal_excel_file: Dataframe of the results excel file

    Output - 
    result_nav_value: Value where 'Field Name' contains 'NAV'
    '''

    # Find rows where 'Field Name' contains 'NAV'
    result_nav_row = orignal_excel_file[orignal_excel_file['Field name'] == 'NAV']

    # Set result_nav_value variable
    result_nav_value = result_nav_row["Final Output result_cleaned"].iloc[0]

    return result_nav_value

# Call function find_nav_value
result_nav_value = find_nav_value(orignal_excel_file)

def update_nav_value(orignal_excel_file, result_nav_value):
    '''
    update_nav_value: Function to update NAV value in "Final Output result_cleaned" where "Field name" is "Default NAV"

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    result_nav_value: Value where 'Field Name' contains 'NAV'

    Output - 
    orignal_excel_file: Dataframe of the results excel file (updated with the NAV value)
    '''

    # Find rows where 'Field Name' contains 'Default NAV'
    index_default_nav = orignal_excel_file.index[orignal_excel_file['Field name'] == 'Default NAV'].tolist()

    # Output default_nav_value to "Final Output result_cleaned"
    orignal_excel_file.loc[index_default_nav, "Final Output result_cleaned"] = result_nav_value

    return orignal_excel_file

orignal_excel_file = update_nav_value(orignal_excel_file = orignal_excel_file, result_nav_value = result_nav_value)


def output_to_excel(orignal_excel_file, excel_directory, output_excel_filename, file_extension):
    '''
    output_to_excel: This function outputs the orignal_excel_file dataframe to an excel file

    Input - 
    orignal_excel_file: Dataframe of the results excel file
    excel_directory: Directory storing excel files
    output_excel_filename: Excel file name that we will output
    file_extension: Extension of the file that we wish output file will be 

    Output - None
    '''
    output_excel_file = f"{output_excel_filename}.{file_extension}"
    excel_file_path = os.path.join(excel_directory, output_excel_file)
    orignal_excel_file.to_excel(excel_file_path, index=True)

# Call function output_files for directory results, input file results and xlsx file extension
output_to_excel(orignal_excel_file = orignal_excel_file, excel_directory = "Results", output_excel_filename = "results_output", file_extension = "xlsx")