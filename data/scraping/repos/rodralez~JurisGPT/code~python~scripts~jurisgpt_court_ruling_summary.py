import langchain
import os
import yaml
from langchain.llms import TextGen
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

langchain.debug = True

#%% Define embedding function
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import NLTKTextSplitter

embedding_fnc = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = NLTKTextSplitter(chunk_size=3000, chunk_overlap=200)

#%% Define the path to the project folder
PROJECT_PATH = os.path.join(os.getenv('HOME'), 'JurisGPT')
# Change to the project folder using $HOME environment variable
os.chdir(PROJECT_PATH)

#%% IMPORT FUNCTIONS
import sys
sys.path.append('code/python/libraries')
import jurisgpt_functions as jur
import text_generator_api as tg

#%% LOAD CONFIG FILE
# open the YAML file and load the contents
with open("config/config.yaml", "r") as f:
    config_data = yaml.load(f, Loader=yaml.FullLoader)

langchain_api_key = config_data['langchain']['api_key']

#%% SET ENVIRONMENT VARIABLES
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
os.environ["LANGCHAIN_SESSION"] = "jurisgpt"

#%% DEFINE THE LLM 
model_url = "http://localhost:5000"
llm = TextGen(model_url=model_url)

template = """
### Humano: Haz un resumen en español del siguiente texto en español.
Texto:
{text}
### Resumen:"""

prompt = PromptTemplate(template=template, input_variables=["text"])

summary_chain = load_summarize_chain(llm=llm,
                                    chain_type='map_reduce',
                                    map_prompt=prompt,
                                    combine_prompt=prompt,
                                    verbose=True,
                                    # return_intermediate_steps=True
                                    )

#%% LOAD THE DOCUMENTS
loader = DirectoryLoader("./rawdata/laboral", glob="**/*.pdf")
documents = loader.load()

for doc in documents:
    #%% EXTRACT TEXT
    text_raw = doc.page_content
    text = text_raw.replace("(cid:0)", "")
    text = text.replace("þÿ", "")
    text = text.replace("\"&", "\"")
    text = text.replace("\" &", "\"")
    text = text.replace("\"&", "\"")
    text = text.replace("&\"", "\"")
    text = text.replace("& \"", "\"")

    # jur.save_text(text_raw, "text_raw.txt")

    #%% EXTRACT SECTIONS
    titles_all = ['ANTECEDENTES', 'A N T E C E D E N T E S', 'P R I M E R A' , 'S E G U N D A', 'T E R C E R A',
              'SOBRE LA', 'R E S U E L V E', 'CUESTIÓN OBJETO']
    titles = ['ANTECEDENTES',  'A N T E C E D E N T E S', 'SOBRE LA', 'R E S U E L V E', 'CUESTIÓN OBJETO']   
    sections_all = jur.extract_sections(text, titles_all)
    # remove section in sections that not cointains any element in titles ignoring case
    sections = [section for section in sections_all if any(title.lower() in section.lower() for title in titles)]

    sections_short = []
    for section in sections:
        sections_short.append(section[:150])

    #%% FILE NAME
    file_path = doc.metadata['source'] # get the path to the file
    file_name = os.path.basename(file_path) # get the file name
    file_name = file_name.replace(".pdf", ".txt") # replace the extension
    file_name = os.path.join("./data/laboral", file_name) # add the path to the data folder

    # Open the file in write mode
    # with open(file_name, "w") as file:
    #     j = 0
    #     # Iterate over the elements in the lists and write them to the file
    #     for section in sections:
    #         j += 1
    #         section_string = "Sección " + str(j) + ": " + section + "\n"
    #         file.write(section_string)
    #         file.write("-------------------------------------------\n")

    #%% NO LANGCHAIN APPROACH
    # summaries = []

    # for i in range(len(sections)):
    #     instruction = "Haz un resumen en español del siguiente texto en español."
    #     prompt = jur.prompt_summary(instruction, sections[i])

    #     print ("Asking a summary to the LLM...")

    #     start_time = jur.tic()
    #     response = tg.query_llm(prompt)
    #     summaries.append(response)

    #     end_time = jur.toc()

    #     elapsed_time = end_time - start_time

    #     print("----------------------------------------------")
    #     print("Sección", i+1, sections_short[i])
    #     print("\n")
    #     print("Resumen", i+1, ":", response)
    #     print(f"LLM response time: {elapsed_time:.3f} seconds.")

    # file_path = doc.metadata['source'] # get the path to the file
    # file_name = os.path.basename(file_path) # get the file name
    # file_name = file_name.replace(".pdf", ".txt") # replace the extension
    # file_name = os.path.join("./data/laboral", file_name) # add the path to the data folder

    # # Open the file in write mode
    # with open(file_name, "w") as file:
    #     j = 0
    #     # Iterate over the elements in the lists and write them to the file
    #     for section, summary in zip(sections, summaries):
    #         j += 1
    #         section_string = "Sección " + str(j) + ": " + section + "\n"
    #         file.write(section_string)
    #         file.write("\n")
    #         summary_string = "Resumen " + str(j) + ": " + summary + "\n"
    #         file.write(summary_string)
    #         file.write("-------------------------------------------\n")

#%% LANGCHAIN APPROACH
    summaries = []

    for i in range(len(sections)):

        from langchain.docstore.document import Document

        jur.save_text(sections[i], "tmp.txt")

        loader_tmp = TextLoader("tmp.txt")
        doc_tmp = loader_tmp.load()
        doc_split = text_splitter.split_documents(doc_tmp)
    
        print ("Asking a summary to the LLM...")
        
        start_time = jur.tic()
        response = summary_chain.run(doc_split)
        summaries.append(response)
        end_time = jur.toc()

        elapsed_time = end_time - start_time

        print("----------------------------------------------")
        print("Sección", i+1, sections_short[i])
        print("\n")
        print("Resumen", i+1, ":", response)
        print(f"LLM response time: {elapsed_time:.3f} seconds.")

    # Open the file in write mode
    with open(file_name, "w") as file:
        j = 0
        # Iterate over the elements in the lists and write them to the file
        for section, summary in zip(sections, summaries):
            j += 1
            section_string = "Sección " + str(j) + ": " + section + "\n"
            file.write(section_string)
            file.write("\n")
            summary_string = "Resumen " + str(j) + ": " + summary + "\n"
            file.write(summary_string)
            file.write("-------------------------------------------\n")

dump_break = 0