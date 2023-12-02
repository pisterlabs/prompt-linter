from backend.agent.tool import Tool
from typing import Optional
from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
)
from backend.agent.database.sql_tool import CustomSQL
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import create_sql_agent
from backend.agent.database.tools import CustomSQLDatabaseToolkit
from backend.gcloud.main import GoogleCloudSetting
import re
import json


def get_key_value_pairs(input_string, keys):
    pairs = re.split(r', (?=\w+:)', input_string)

    result = {}

    for pair in pairs:
        key, value = map(str.strip, pair.split(':', 1))
        if key in keys:
            if value.startswith('[') and value.endswith(']'):
                value = [item.strip()
                         for item in value[1:-1].split(',') if item.strip()]
            result[key] = value

    return result


class GoogleCloudDatabaseSQLQuery(Tool):
    description = (
        "This tool is useful when you need to Create, Read, Delete information with SQL query on a database in Google Cloud SQL."
        "format must be python dict and work with json.loads(param): connection: [instance connection name], user: [user name], password: [user password], database: [db name], task: [task]"
    )

    arg_description = f"format must be python dict and work with json.loads(param): connection: [instance connection name], user: {GoogleCloudSetting.AI_DB_USER}, password: {GoogleCloudSetting.AI_DB_PASS}, database: [db name], task: [task]"

    async def call(
        self, goal: str, task: str, input_str: str
    ) -> str:

        """This tool useful when you need to run SQL query"""

        di = {
            "query": goal,
            "task": task,
            "input_str": input_str
        }

        selected_keys = ['connection', 'user', "password", "database"]
        cred = json.loads(input_str)

        missing_keys = [key for key in selected_keys if key not in cred]

        if missing_keys:
            return {
                "query": goal,
                "task": task,
                "input_str": input_str,
                "status": f"Failed, missing keys and value {missing_keys}"
            }

        db = CustomSQL.from_gcloud(
            connection=cred["connection"],
            user=cred["user"],
            password=cred["password"],
            db=cred["database"]
        )

        toolkit = CustomSQLDatabaseToolkit(db=db, llm=OpenAI(temperature=0))

        agent_executor = create_sql_agent(
            llm=OpenAI(temperature=0),
            toolkit=toolkit,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

        test = await agent_executor.arun(task)

        print(test)

        return {
            "goal": goal,
            "input_str": input_str,
            "value": test
        }

        # return "Tools currently not available"
