from openai import OpenAI, OpenAIError
from app.core.config import settings
from app.models.database import assistants, project_assistant_association
from app.dependencies import get_database
from app.config.project_config import project_config
from sqlalchemy import select
import datetime

client = OpenAI(api_key=settings.openai_api_key)
database = get_database()

async def create_assistant_util(name, instructions, model):
    try:
        return client.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=[{"type": "code_interpreter"}],
            model=model
        )
    except OpenAIError as e:
        raise e

async def delete_project_assistants_util(project_id: int):
    try:
        # Retrieve assistant IDs associated with the project
        assistant_ids = await get_project_openai_assistant_ids_util(project_id)

        # Delete each assistant from OpenAI
        for assistant_id in assistant_ids:
            client.beta.assistants.delete(assistant_id=assistant_id)

        # Delete assistant associations from the database
        delete_association_query = project_assistant_association.delete().where(
            project_assistant_association.c.project_id == project_id
        )
        await database.execute(delete_association_query)

        # Delete assistants from the database
        delete_assistants_query = assistants.delete().where(
            assistants.c.assistant_id.in_(assistant_ids)
        )
        await database.execute(delete_assistants_query)

        return {"message": f"All assistants associated with project {project_id} have been deleted"}
    except Exception as e:
        raise e

async def insert_assistant_data_util(assistant_id, assistant_name, assistant_type, assistant_instructions, assistant_model):
    assistant_query = assistants.insert().values(
        assistant_id=assistant_id,
        assistant_name=assistant_name,
        assistant_type=assistant_type,
        assistant_instructions=assistant_instructions,
        assistant_model=assistant_model,
        created_at=datetime.datetime.now()
    )
    assistant_id = await database.execute(assistant_query)
    return assistant_id

async def associate_assistant_with_project_util(project_id, assistant_id):
    association_query = project_assistant_association.insert().values(
        project_id=project_id,
        assistant_id=assistant_id
    )
    await database.execute(association_query)

async def assistant_exists_util(name: str, assistant_type: str):
    query = assistants.select().where(
        (assistants.c.assistant_name == name) & (assistants.c.assistant_type == assistant_type)
    )
    result = await database.fetch_one(query)
    return result is not None

async def get_project_openai_assistant_ids_util(project_id: int):
    # Define a SQL query to join project_assistant_association with assistants
    # to fetch the OpenAI assistant IDs (assistant_id from assistants table)
    join_query = select([
        assistants.c.assistant_id
    ]).select_from(
        project_assistant_association.join(assistants, project_assistant_association.c.assistant_id == assistants.c.id)
    ).where(
        project_assistant_association.c.project_id == project_id
    )

    # Execute the query and fetch results
    result = await database.fetch_all(join_query)
    return [row['assistant_id'] for row in result]

async def get_openai_assistant_id_by_name_util(assistant_name: str):
    query = select([assistants.c.assistant_id]).where(assistants.c.assistant_name == assistant_name)
    result = await database.fetch_one(query)
    return result['assistant_id'] if result else None

def generate_assistant_instructions(assistant_type):
    instructions = project_config["assistants"][assistant_type]["instructions"]
    global_props = project_config["global_properties"]
    return instructions.format(
        max_exchanges=global_props["max_exchanges"]
    )