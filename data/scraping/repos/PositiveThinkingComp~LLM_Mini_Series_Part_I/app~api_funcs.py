
import json
import os
import ast
from pathlib import Path
from typing import List
import langchain.llms as LLM
from app.custom_langchain import ModAPIChain, APIResponse, FlexAPIChain, FlexAPIChainPayload
from app.utils import find_value, custom_api_prompt
from app.custom_api_prompts import API_REQUEST_PROMPT, \
    API_REQUEST_PROMPT2, API_RESPONSE_PROMPT, API_RESPONSE_PROMPT2, PERMISSION_PROMPT

PATH = Path(os.path.abspath(os.path.dirname(__file__)))

# Open the YAML file
api_doc_path = PATH / 'dbr_api_docs'

def prepare_api_docs():
    """
    Add environment variables to the individual
    API docs.
    """
    updated_api_docs = {}
    for filename in os.listdir(api_doc_path):
        with open(os.path.join(api_doc_path, filename)) as f:
            updated_api_docs[filename] = f.read()
            updated_api_docs[filename] = updated_api_docs[filename].replace(
                "{DATABRICKS_HOST}", os.getenv('DATABRICKS_HOST'))
    return updated_api_docs


def get_run(llm: LLM, runID: str, query: str, api_text: str, headers: dict) -> str:
    """
    Get infos from an MLflow run.

    Args:
        llm (LLM): Large language model
        runID (str): Run ID
        query (str): Query
        api_text (str): API documentation for a particular endpoint
        headers (dict): Authorization for the Databricks API
    
    Return:
        (str): the response of the language model
    """
    mlflow_run_get_chain = APIResponse.from_llm_and_api_docs(llm, api_text, headers=headers, verbose=True)
    run_infos = json.loads(mlflow_run_get_chain.run(
        f"""Please give me the infos about the run '{runID}'"""))
    # Please return the f1 score
    return llm(f"""{query} of the metrics : {find_value(run_infos, 'metrics')}""")


def trans_model(llm: LLM, query: str, api_text: str, headers: dict):
    """
    Transition a MLflow model to another stage.

    Args:
        llm (LLM): Large language model
        query (str): Query for the model
        api_text (str): API documentation for a particular endpoint
        headers (dict): Authorization for the Databricks API
    """
    mlflow_model_transition_chain = FlexAPIChainPayload.from_llm_and_api_docs(
        llm, api_text, headers=headers, api_url_prompt=API_REQUEST_PROMPT2,
        api_response_prompt=API_RESPONSE_PROMPT2, verbose=True
    )
    mlflow_model_transition_chain.run(query)


def get_model(llm: LLM, model_query: str, api_text: str, headers: dict) -> str:
    """
    Get all infos about a model and its particular stages

    Args:
        llm (LLM): Large language model
        model_query (str): Query
        api_text (str): API documentation for a particular endpoint
        headers (dict): Authorization for the Databricks API
    
    Return:
        (str): response of the language model
    """
    mlflow_model_get_chain = APIResponse.from_llm_and_api_docs(llm, api_text, headers=headers, verbose=True)
    api_result = mlflow_model_get_chain.run(model_query)  

    return llm(
        f"""What is the runID and their version of the latest version with stage 'None' \
            and stage 'Production' respectively, given the following info: {api_result}"""
    )


def batch_mod_permission(
    logger, llm: LLM, updated_api_docs: dict, api_text: str, api_name: str,
    headers: dict, permission_mod: str, jobs: List[str]
):
    """
    Get or modify permissions for a batch of jobs.

    Args:
        logger (Logger): Logger object
        llm (LLM): Large language model
        updated_api_docs (dict): Updated API docs
        api_text (str): API documentation for a particular endpoint
        api_name (str): API name
        headers (dict): Authorization for the Databricks API
        permission_mod (str): Permission modification
        jobs (list): List of jobs
    """
    get_permission_txt = updated_api_docs['get_permissions.txt']
    for job in jobs:
        logger.info(f"Permission for job: {job}")
        mod_permissions(
            logger, llm, permission_mod, job, headers,
            get_permission_txt, api_text, api_name
        )


def get_permissions(logger, llm: LLM, api_text: str, headers: dict, jobID: str) -> dict:
    """
    Get the permissions of a particular job.
    
    Args:
        logger (Logger): Logger object
        llm (LLM): Large language model
        api_text (str): API documentation for a particular endpoint
        headers (dict): Authorization for the Databricks API
        jobID (str): ID of the Databricks job
    
    Return:
        (dict): Access control list
    """
    init_query = f"""Get the jobs permissions for jobID {jobID}"""
    permission_get_chain = APIResponse.from_llm_and_api_docs(llm, api_text, headers=headers, verbose=True)

    acc_control_list = {'access_control_list': json.loads(permission_get_chain.run(init_query))['access_control_list']}
    logger.info(acc_control_list)
    return acc_control_list    

def mod_permissions(
    logger,
    llm: LLM,
    permission_mod: str,
    jobID: str, headers: dict,
    get_permission_txt: str,
    api_text: str, api_name: str
):
    """
    Modify permissions for a job.

    Args:
        logger (Logger): Logger object
        llm (LLM): Large language model
        permission_mod (str): Permission modification
        jobID (str): ID of the Databricks job
        headers (dict): Authorization for the Databricks API
        get_permission_txt (str): API text for getting permissions
        api_text (str): API documentation for a particular endpoint
        api_name (str): file name of the API documentation
    """
    acc_control_list = get_permissions(logger, llm, get_permission_txt, headers, jobID)

    # Update the permission, if the query demands it
    if api_name == 'update_permissions.txt':
        permission_prompt= PERMISSION_PROMPT.format(
            acc_control_list=acc_control_list, permission_mod=permission_mod)

        output_acc_list = llm(permission_prompt)
        patch_payload = ast.literal_eval(output_acc_list.rstrip().lstrip()) #
        # Reformat the general permission response into the form for the patch payload
        for idx, _ in enumerate(patch_payload['access_control_list']):
            patch_payload['access_control_list'][idx]['permission_level'] = patch_payload['access_control_list'][idx]['all_permissions'][0]['permission_level']
            patch_payload['access_control_list'][idx].pop('all_permissions')

        permission_update_chain = FlexAPIChain.from_llm_and_api_docs(
            llm, api_text, headers=headers, verbose=True,
            api_url_prompt=API_REQUEST_PROMPT, api_response_prompt=API_RESPONSE_PROMPT
        ) # use updated API Chain which can do post, put and patch also....

        permission_update_chain.body = patch_payload
        logger.info(f"Payload for permission update: {patch_payload}")
        permission_update_chain(f"""Update the jobs permissions of jobID {jobID}""")    


def get_job_infos(
    llm: LLM, query: str, response_query: str, api_text: str, headers: dict
) -> str:
    """
    Get particular infos about the list of existing jobs.

    Args:
        llm (LLM): Large language model
        query (str): Query to get the list of jobs
        response_query (str): Query to get the response
        api_text (str): API documentation for a particular endpoint
        headers (dict): Authorization for the Databricks API
    
    Return:
        (str): response from the large language model
    """
    job_chain = ModAPIChain.from_llm_and_api_docs(
        llm, api_text, headers=headers, verbose=True
    )

    custom_resp = f"This response contains a list of databricks jobs. {response_query}"
    jobs_result = custom_api_prompt(llm, job_chain, query, custom_resp)
    return jobs_result 