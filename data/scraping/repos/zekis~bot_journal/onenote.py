import sys
import traceback
import bot_config
import requests

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Type

sys.path.append("/root/projects")
import common.bot_logging
from common.bot_comms import publish_event_card, publish_list, publish_error
from common.bot_utils import tool_description, tool_error


from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.tools import BaseTool


#common.bot_logging.bot_logger = common.bot_logging.logging.getLogger('ToolLogger')
#common.bot_logging.bot_logger.addHandler(common.bot_logging.file_handler)

def get_access_token(tenant_id, client_id, client_secret):
    url = f'https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token'
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    body = {
        'grant_type': 'client_credentials',
        'client_id': client_id,
        'client_secret': client_secret,
        'scope': 'https://graph.microsoft.com/.default'
    }
    response = requests.post(url, headers=headers, data=body)
    response_json = response.json()
    access_token = response_json.get('access_token')
    return access_token


def create_notebook(access_token, user_principal, notebook_name):
    url = f'https://graph.microsoft.com/v1.0/users/{user_principal}/onenote/notebooks'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    body = {
        'displayName': notebook_name
    }
    response = requests.post(url, headers=headers, json=body)
    common.bot_logging.bot_logger.debug(response.text)
    return response.json()

def create_page(access_token, user_principal, section_id, page_name, new_content):
    url = f'https://graph.microsoft.com/v1.0/users/{user_principal}/onenote/sections/{section_id}/pages'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'text/html'
    }
    # Prepending the page title to the HTML content
    body = f'<html><head><title>{page_name}</title></head><body>{new_content}</body></html>'
    response = requests.post(url, headers=headers, data=body)
    return response.status_code


def get_notebook_id(access_token, user_principal, notebook_name):
    url = f'https://graph.microsoft.com/v1.0/users/{user_principal}/onenote/notebooks'
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(url, headers=headers)
    common.bot_logging.bot_logger.info(response.text)
    for notebook in response.json()['value']:
        if notebook['displayName'] == notebook_name:
            return notebook['id']
    publish_error(f'Could not find notebook {notebook_name}',notebook_name)
    return None

def get_section_id(access_token, user_principal, notebook_id, section_name):
    url = f'https://graph.microsoft.com/v1.0/users/{user_principal}/onenote/notebooks/{notebook_id}/sections'
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(url, headers=headers)
    common.bot_logging.bot_logger.info(response.text)
    for section in response.json()['value']:
        if section['displayName'] == section_name:
            return section['id']
    publish_error(f'Could not find notebook section {section_name}',section_name)
    return None

def get_page_id(access_token, user_principal, section_id, page_name):
    url = f'https://graph.microsoft.com/v1.0/users/{user_principal}/onenote/sections/{section_id}/pages'
    headers = {'Authorization': f'Bearer {access_token}'}
    response = requests.get(url, headers=headers)
    common.bot_logging.bot_logger.info(response.text)
    for page in response.json()['value']:
        if page['title'] == page_name:
            return page['id']
    publish_error(f'Could not find notebook page {page_name}',page_name)
    return None

def update_page(access_token, user_principal, page_id, new_content):
    url = f'https://graph.microsoft.com/v1.0/users/{user_principal}/onenote/pages/{page_id}/content'
    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }
    # Get the current date and time
    current_date_time = datetime.now()

    # Format the current date and time to display only hours and minutes
    current_time = current_date_time.strftime('%H:%M')


    body = [
        {
            "target": "body",
            "action": "append",
            "content": f"</p></p></p></p><p><h2>{current_time}</h2><p>{new_content}</p></p></p></p></p></p></p></p>"
        }
    ]
    response = requests.patch(url, headers=headers, json=body)
    common.bot_logging.bot_logger.info(response.text)
    return response.status_code



class NoteAppend(BaseTool):
    parameters = []
    optional_parameters = []
    name = "JOURNAL_PAGE_APPEND"
    summary = "Useful for when you need to add an entry to existing page in the journal"
    parameters.append({"name": "content", "description": "HTMl formatted text or content to be added to the specified page within the journal."})
    description = tool_description(name, summary, parameters, optional_parameters)
    return_direct = True

    def _run(self, content: str = None, publish: str = "True", run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            ai_summary = ""
            human_summary = []
            #bot_config.APP_ID
            #bot_config.APP_SECRET
            #bot_config.TENANT_ID
            #bot_config.USER_NAME
            #bot_config.OFFICE_USER
            #bot_config.NOTEBOOK
            #bot_config.SECTION
            tenant_id = bot_config.TENANT_ID
            client_id = bot_config.APP_ID
            client_secret = bot_config.APP_SECRET
            user_principal = bot_config.OFFICE_USER
            notebook_name = bot_config.NOTEBOOK
            section_name = bot_config.SECTION

            
            # Get the current date
            now = datetime.now()

            # Calculate the start of the week (Monday)
            start_of_week = now - timedelta(days=now.weekday())

            # Format the start_of_week as a string in the format 'YYYY-MM-DD'
            start_of_week_str = start_of_week.strftime('%Y-%m-%d')

            # Create the page name
            page_name = f'Week Begins {start_of_week_str}'

            common.bot_logging.bot_logger.info(f"tenant_id: {tenant_id}, client_id: {client_id}, client_secret: {client_secret}, user_principal: {user_principal}, notebook_name: {notebook_name}, section_name: {section_name}, page_name: {page_name}")

            #page_name = 'Week Begins 2023-09-25'
            new_content = content

            access_token = get_access_token(tenant_id, client_id, client_secret)
            
            notebook_id = get_notebook_id(access_token, user_principal, notebook_name)
            if notebook_id:
                section_id = get_section_id(access_token, user_principal, notebook_id, section_name)
                if section_id:
                    page_id = get_page_id(access_token, user_principal, section_id, page_name)  # Add page_name as an argument
                    if not page_id:
                        status_code = create_page(access_token, user_principal, section_id, page_name, new_content)
                        if status_code == 204:
                            'do nothing' 
                            return(f"The page in '{notebook_name}' notebook and '{section_name}' section was successfully created.")
                        else:
                            return(f"Failed to create the page. Status Code: {status_code}")
                    else:
                        status_code = update_page(access_token, user_principal, page_id, new_content)
                        if status_code == 204:
                            'do nothing'
                            return(f"""Added the following to your journal:
                            
{content}""")
                        else:
                            return(f"Failed to update the page. Status Code: {status_code}")
                    
                else:
                    return("Section not found.")
            else:
                return("Notebook not found.")
            
        except Exception as e:
            #traceback.print_exc()
            tb = traceback.format_exc()
            publish_error(e, tb)
            return tool_error(e, tb, self.description)
    
    async def _arun(self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("JOURNAL_GET does not support async")

