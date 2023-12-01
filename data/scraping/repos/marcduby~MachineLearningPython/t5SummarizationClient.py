

# imports
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
import transformers
from transformers import pipeline
import torch
from langchain import PromptTemplate,  LLMChain
import os 
import pymysql as mdb
from time import gmtime, strftime
import time



# constants
KEY_GPT = os.environ.get('HUG_KEY')
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
PROMPT_LLM = """
Write a concise summary of the following text delimited by triple backquotes.
Return your response in sentences which covers the key points of the text.
```{input_text}```
BULLET POINT SUMMARY:
"""

PROMPT_COMMAND = "Summarize the information related to {} from the following text delimited by triple backquotes.\n"
PROMPT_COMMAND = "Summarize in 200 words the information related to {} from the following text: \n"
PROMPT_LLM_GENE = """
{input_text}
"""


# DB constants
DB_PASSWD = os.environ.get('DB_PASSWD')
NUM_ABSTRACT_LIMIT = 5
SCHEMA_GPT = "gene_gpt"
DB_PAPER_TABLE = "pgpt_paper"
DB_PAPER_ABSTRACT = "pgpt_paper_abtract"

SQL_SELECT_ABSTRACT_BY_TITLE = "select id from {}.pgpt_paper_abstract where title = %s".format(SCHEMA_GPT)
SQL_SELECT_ABSTRACT_LIST_LEVEL_0 = """
select abst.id, abst.abstract 
from {}.pgpt_paper_abstract abst, {}.pgpt_search_paper seapaper 
where abst.document_level = 0 and seapaper.paper_id = abst.pubmed_id and seapaper.search_id = %s limit %s
""".format(SCHEMA_GPT, SCHEMA_GPT, SCHEMA_GPT)
# and abst.id not in (select child_id from {}.pgpt_gpt_paper where search_id = %s) limit %s

SQL_SELECT_ABSTRACT_LIST_LEVEL_HIGHER = """
select distinct abst.id, abst.abstract, abst.document_level
from {}.pgpt_paper_abstract abst, {}.pgpt_gpt_paper gpt
where abst.document_level = %s and gpt.parent_id = abst.id and gpt.search_id = %s
and abst.id not in (select child_id from {}.pgpt_gpt_paper where search_id = %s) limit %s
""".format(SCHEMA_GPT, SCHEMA_GPT, SCHEMA_GPT)

# SQL_INSERT_PAPER = "insert into {}.pgpt_paper (pubmed_id) values(%s)".format(SCHEMA_GPT)
SQL_INSERT_ABSTRACT = "insert into {}.pgpt_paper_abstract (abstract, title, journal_name, document_level) values(%s, %s, %s, %s)".format(SCHEMA_GPT)
SQL_INSERT_GPT_LINK = "insert into {}.pgpt_gpt_paper (search_id, parent_id, child_id, document_level) values(%s, %s, %s, %s)".format(SCHEMA_GPT)
SQL_UPDATE_ABSTRACT_FOR_TOP_LEVEL = "update {}.pgpt_paper_abstract set search_top_level_of = %s where id = %s".format(SCHEMA_GPT)

SQL_SELECT_SEARCHES = "select id, terms, gene from {}.pgpt_search where ready='Y' limit %s".format(SCHEMA_GPT)
SQL_UPDATE_SEARCH_AFTER_SUMMARY = "update {}.pgpt_search set ready = 'N', date_last_summary = sysdate() where id = %s ".format(SCHEMA_GPT)

# methods
def get_model_tokenizer(name, log=False):
    '''
    returns the associated model and tokenizer
    '''
    # initialize
    tokenizer = AutoTokenizer.from_pretrained(name)
    pipeline = transformers.pipeline(
        "text-generation", #task
        model=name,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=1000,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id, use_auth_token=True
    )

    model = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0}, use_auth_token=True)

    # return
    return model, tokenizer


def summarize(model, gene, command, prompt, text_to_summarize, log=False):
    '''
    summarixes the test input using the model and prompt provided
    '''
    # initialize
    summarizer = pipeline("summarization", model="t5-base")
    # params = {"max_length": 1000, 
    #         "min_length": 1000,
    #         "do_sample": False}
    params = {"max_length": 300, 
            "min_length": 200,
            "do_sample": False}
    command_gene = PROMPT_COMMAND.format(gene)
    prompt_and_command = command_gene + text_to_summarize
    result_summary = None

    # log
    if log:
        print("using prompt: \n{}".format(prompt_and_command))

    # summarize
    summary = summarizer(prompt_and_command, **params)
    result_summary = summary[0]['summary_text']
    # summary = llm_chain.run(text_to_summarize)

    # return
    return result_summary

def get_list_abstracts(conn, id_search, num_level=0, num_abstracts=NUM_ABSTRACT_LIMIT, log=False):
    '''
    get a list of abstract map objects
    '''
    # initialize
    list_abstracts = []
    cursor = conn.cursor()

    # pick the sql based on level
    if log:
        print("searching for abstracts got input search: {}, doc_level: {}, limit: {}".format(id_search, num_level, num_abstracts))
    if num_level == 0:
        # cursor.execute(SQL_SELECT_ABSTRACT_LIST_LEVEL_0, (id_search, id_search, num_abstracts))
        cursor.execute(SQL_SELECT_ABSTRACT_LIST_LEVEL_0, (id_search, num_abstracts))
    else:
        cursor.execute(SQL_SELECT_ABSTRACT_LIST_LEVEL_HIGHER, (num_level, id_search, id_search, num_abstracts))

    # query 
    db_result = cursor.fetchall()
    for row in db_result:
        paper_id = row[0]
        abstract = row[1]
        list_abstracts.append({"id": paper_id, 'abstract': abstract})

    # return
    return list_abstracts

def get_connection():
    ''' 
    get the db connection 
    '''
    conn = mdb.connect(host='localhost', user='root', password=DB_PASSWD, charset='utf8', db=SCHEMA_GPT)

    # return
    return conn


# main
if __name__ == "__main__":
    # initialize
    num_level = 0
    id_search = 2

    # get the db connection
    conn = get_connection()

    # get the abstracts
    list_abstracts = get_list_abstracts(conn=conn, id_search=id_search, num_level=num_level, num_abstracts=2, log=True)

    # get the llm summary
    str_input = ""
    if len(list_abstracts) > 1:
        # top level is not this level if more than 2 abstracts found at this level
        found_top_level = False
        for item in list_abstracts:
            abstract = item.get('abstract')
            print("using abstract: \n{}".format(abstract))
            str_input = str_input + " " + abstract

        # log
        print("using {} for gpt query for level: {} and search: {}".format(len(list_abstracts), num_level, id_search))

    # get the model
    # llm_model = get_model_tokenizer(MODEL_NAME)
    # print("got model: {}".format(llm_model))

    # get the summary
    summary = summarize(model=None, gene='UBE2NL', command=PROMPT_COMMAND, prompt=PROMPT_LLM_GENE, text_to_summarize=str_input, log=True)
    print("got summary: \n{}".format(summary))


