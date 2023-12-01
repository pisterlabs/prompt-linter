import os
from dotenv import load_dotenv
from typing import Optional
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from backend.agent.tool import Tool
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from backend.gcloud.main import GoogleCloudSetting, get_instances, auth, get_databases, get_db

load_dotenv()
DEFAULT_PROJECT = os.getenv("DEFAULT_PROJECT")


# 1
class GoogleCloudProject(Tool):
    description = (
        "This must be the very first tool to get Google Cloud API to get credentials, connection, project id."
        "Get all parameters to pass to other tools"
    )
    arg_description = "Retrieve Google Cloud Project connections, credentials, communicate with other agent this NEVER be an empty!"
    public_description = "Google Cloud Project"

    async def call(
        self, goal: str, task: str, input_str: str
    ) -> str:
        """Return google cloud project id from environment setting"""
        cred = {var_name: var_value for var_name, var_value in vars(
            GoogleCloudSetting).items() if not var_name.startswith('__')}

        return {
            "goal": goal,
            "input_str": input_str,
            "value": cred
        }

# 2


class GoogleCloudAuth(Tool):
    description = (
        "This tool is use when you need to authenticate Google Cloud API"
    )
    public_description = "Authenticate Google Cloud API"
    arg_description = "Authenticate Google Cloud API, NEVER be an empty!"

    parent = GoogleCloudProject

    async def call(
        self, goal: str, task: str, input_str: str
    ) -> str:
        """Authenticate Google Cloud API"""
        di = {
            "query": goal,
            "task": task,
            "input_str": input_str
        }

        print(di)
        return auth()

# 3


class GoogleCloudInstances(Tool):
    description = (
        # "This tool must use after GoogleCloudProject, otherwise ERROR!."
        "Retrieve all instances of the project on Google Cloud SQL."
        "args format in dict python: project: projectId"
        "args could not be empty work with json.loads(args)."
    )

    arg_description = "project: projectId, could not be empty work with json.loads(args)."
    public_description = "Retrieve all instances of the project on Google Cloud SQL."

    parent = GoogleCloudProject

    async def call(
        self, goal: str, task: str, input_str: str
    ) -> str:
        """Get all Google Cloud SQL instances"""

        di = {
            "query": goal,
            "task": task,
            "input_str": input_str
        }

        print(di)

        # return "Tool under maintain"
        all_instances = await get_instances(input_str=input_str)

        return {
            "goal": goal,
            "input_str": input_str,
            "value": all_instances
        }


class GoogleCloudDatabases(Tool):
    description = (
        # "Must use these functions [GoogleCloudProject, GoogleCloudInstances], otherwise ERROR!.."
        "NOT for retrieving anything related SQL such as columns, row."
        "This tool is get status databases informations on an Google Cloud SQL instance."
        "args format in dict python: project: [project id], instance: [instance name]"
        "args could not be empty: json.loads(args)."
    )

    arg_description = "format: project: [project id], instance: [instance name]"
    public_description = "This tool is get all databases"

    parent = GoogleCloudInstances

    async def call(
        self, goal: str, task: str, input_str: str
    ) -> str:
        """
        Get all google cloud databases on an Google Cloud SQL instance
        We need two things: project and instance name
        """

        di = {
            "query": goal,
            "task": task,
            "input_str": input_str
        }
        try:
            all_db = await get_databases(input_str=input_str)
            return all_db
        except Exception as e:
            print(e)
            return {
                "status": "failed",
                "error": e
            }


class GoogleCloudGetDB(Tool):
    description = (
        # "Must use these tools [GoogleCloudProject, GoogleCloudInstances], otherwise ERROR!"
        "NOT for retrieving anything related SQL such as columns, row.!"
        "This tool is get stat on a database of an instance."
        "args format in dict python: project: [project id], useful when you need to get instance: [instance name], database: [database name]"
    )

    parent = GoogleCloudInstances

    arg_description = "args format in dict python: project: [project id], instance: [instance name], database: [database name]"
    public_description = "useful when you need to get a database meta information on instance"

    async def call(
        self, goal: str, task: str, input_str: str
    ) -> str:
        """Get meta data of a specfic Google Cloud SQL Database"""

        di = {
            "query": goal,
            "task": task,
            "input_str": input_str
        }

        print(di)

        database_detail = await get_db(input_str=input_str)

        return {
            "goal": goal,
            "input_str": input_str,
            "value": database_detail
        }


# Below are not in use
class GoogleCloudInstanceUsersList(Tool):
    description = (
        "Note: Instance on Google Cloud, is not user of any databases."
        "useful when you need to get users list on specific an instance on Google Cloud"
        "use format: project: [project_id], instance: [instance name]"
    )
    arg_description = "useful when you need to get users list on specific an instance on Google Cloud"
    public_description = "useful when you need to get users list on specific an instance on Google Cloud"

    async def call(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        print(query)

        # all_users = get_instance_users_list(project=project, instance=instance)

        return "tool not available!"


class GoogleCloudDeleteDB(Tool):
    description = (
        "useful when you need to delete sepecfic database on instance"
        "format: project: [project id], instance: [instance name], db: [database name]"
    )

    arg_description = "format: project: [project id], instance: [instance name], db: [database name]"
    public_description = "useful when you need to delete sepecfic database on instance"

    async def call(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        values = extract_values(query)
        try:
            return ""
        except Exception as e:
            print(e)
            return {
                "status": "failed",
                "error": e
            }


class GoogleCloudDeleteInstanceUser(Tool):
    description = (
        "Note: Instance on Google Cloud, is not user of any databases."
        "useful when you need to delete a user on specific an instance on Google Cloud"
        "use format: project: [project_id], instance: [instance name], user: [user_name]"
    )

    arg_description = "use format: project: [project_id], instance: [instance name], user: [user_name]"
    public_description = "useful when you need to delete a user on specific an instance on Google Cloud"

    async def call(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        return ""


class GoogleCloudGetInstanceUser(Tool):
    description = (
        " Note: Instance on Google Cloud, is not user of any databases."
        "useful when you need to get a user on specific an instance on Google Cloud"
        "use format: project: [project_id], instance: [instance name], user: [user_name]"
    )

    arg_description = "use format: project: [project_id], instance: [instance name], user: [user_name]"
    public_description = "useful when you need to get a user on specific an instance on Google Cloud"

    async def call(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""

        # get_specific_instance_user(project=project, instance=instance, user=user)

        return ""


class GoogleCloudGetBackupInstanceUser(Tool):
    description = (
        "Note: Instance on Google Cloud, is not user of any databases."
        "useful when you need to GET backup list on specific an instance on Google Cloud"
        "use format: project: [project_id], instance: [instance name]"
    )

    arg_description = "use format: project: [project_id], instance: [instance name]"
    public_description = "useful when you need to GET backup list on specific an instance on Google Cloud"

    async def call(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:

        return ""


class GoogleCloudGetSpecificBackupInstaceId(Tool):

    description = (
        "Note: Instance on Google Cloud, is not user of any databases."
        "useful when you need to GET specifc backup id on specific an instance on Google Cloud"
        "use format: project: [project_id], instance: [instance name], backup_id: [backup_id]"
    )
    arg_description = "use format: project: [project_id], instance: [instance name], backup_id: [backup_id]"
    public_description = "useful when you need to GET specifc backup id on specific an instance on Google Cloud"

    async def call(
        self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        return ""
