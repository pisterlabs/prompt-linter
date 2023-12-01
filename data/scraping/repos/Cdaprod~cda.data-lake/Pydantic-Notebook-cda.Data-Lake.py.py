#!/usr/bin/env python
# coding: utf-8

# In[1]:


# LCEL + Kor Extraction Method
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text
from langchain.chat_models import ChatOpenAI

schema = Object(
    id="credential",
    description="Credentials information",
    attributes=[
        Text(id="service", description="The service the credentials are for."),
        Text(id="key", description="The key."),
        Text(id="secret", description="The secret.")
    ],
    many=True,
)

from langchain.llms import OpenAI

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    max_tokens=2000,
    frequency_penalty=0,
    presence_penalty=0,
    top_p=1.0,
)

chain = create_extraction_chain(llm, schema)
extracted_data = chain.run(markdown_data)["data"]

from kor import from_pydantic
from pydantic import BaseModel, Field

class Credential(BaseModel):
    service: str = Field(description="The service the credentials are for.")
    key: str = Field(description="The key.")
    secret: str = Field(description="The secret.")

schema, validator = from_pydantic(
    Credential,
    description="Credentials Information",
    many=True,
)

chain = create_extraction_chain(llm, schema, validator=validator)


# In[ ]:


# Supabase Client with Env
# !pip install supabase-py

import os
from dotenv import load_dotenv
from supabase_py import create_client

# Load environment variables from .env file
load_dotenv()

# Access environment variables
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_API_KEY = os.getenv('SUPABASE_API_KEY')

# Initialize a Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_API_KEY)


# In[ ]:


# MariaDB Client with Env
# !pip install mysql-connector-python

import os
from dotenv import load_dotenv
import mysql.connector

# Load environment variables from .env file
load_dotenv()

# Access environment variables
MARIADB_HOST = os.getenv('MARIADB_HOST')
MARIADB_USER = os.getenv('MARIADB_USER')
MARIADB_PASSWORD = os.getenv('MARIADB_PASSWORD')
MARIADB_DATABASE = os.getenv('MARIADB_DATABASE')

# Initialize a MariaDB connection
mariadb_connection = mysql.connector.connect(
    host=MARIADB_HOST,
    user=MARIADB_USER,
    password=MARIADB_PASSWORD,
    database=MARIADB_DATABASE
)


# In[ ]:


# minio_client using .env
import os
from dotenv import load_dotenv
from minio import Minio

# Load environment variables from .env file
load_dotenv()

# Access environment variables
endpoint = os.getenv('MINIO_ENDPOINT')
access_key = os.getenv('MINIO_ACCESS_KEY')
secret_key = os.getenv('MINIO_SECRET_KEY')
secure = os.getenv('MINIO_SECURE') == 'True'  # Convert string to boolean

# Initialize a Minio client
minio_client = Minio(
    endpoint=endpoint,
    access_key=access_key,
    secret_key=secret_key,
    secure=secure
)
minio_client


# In[ ]:


# Given the list of repositories provided by the user, we will fetch data from GitHub API for each.

import requests
from pydantic import BaseModel, ValidationError
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from datetime import datetime

# Define the Owner model
class Owner(BaseModel):
    name: str
    id: int
    type: str  # User or Organization

# Define the Repository model
class Repository(BaseModel):
    id: int
    node_id: str
    name: str
    full_name: str
    owner: Owner
    private: bool
    html_url: HttpUrl
    description: Optional[str]
    fork: bool
    url: HttpUrl
    created_at: datetime
    updated_at: datetime
    pushed_at: datetime
    git_url: str
    ssh_url: str
    clone_url: HttpUrl
    size: int
    stargazers_count: int
    watchers_count: int
    language: Optional[str]
    has_issues: bool
    has_projects: bool
    has_downloads: bool
    has_wiki: bool
    has_pages: bool
    forks_count: int
    mirror_url: Optional[HttpUrl]
    archived: bool
    disabled: bool
    open_issues_count: int
    license: Optional[str]
    allow_forking: bool
    is_template: bool
    topics: List[str]
    visibility: str  # 'public' or 'private'

repo_names = [
    'cda.langchain', 'cda.agents',
    'cda.Juno', 'cda.actions',
    'cda.data-lake', 'cda.ml-pipeline', 'cda.notebooks',
    'cda.CMS_Automation_Pipeline', 'cda.ml-pipeline',
    'cda.data', 'cda.databases', 'cda.s3',
    'cda.docker', 'cda.kubernetes', 'cda.jenkins',
    'cda.weaviate', 'cda.WeaviateApiFrontend',
    'cda.Index-Videos',
    'cda.dotfiles', 'cda.faas', 'cda.pull', 'cda.resumes', 'cda.snippets', 'cda.superagent', 'cda.ZoomVirtualOverlay', 'cda.knowledge-platform',
    'cda.nginx'
]

import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

def fetch_repo_data(repo_name: str):
    # Get GitHub API token from environment variables
    gh_token = os.getenv('GH_TOKEN')

    headers = {'Authorization': f'token {gh_token}'}
    response = requests.get(f'https://api.github.com/repos/Cdaprod/{repo_name}', headers=headers)
    if response.status_code == 200:
        return response.json()  # Returns the repo data as a dictionary
    else:
        raise Exception(f'Failed to fetch data for {repo_name}, status code {response.status_code}')


# Function to construct the repository JSON object
def construct_repo_json(repo_data: dict):
    try:
        # Validate and create a Repository object
        repo = Repository(**repo_data)
        # Return the JSON representation
        return repo.json(indent=2)
    except ValidationError as e:
        raise Exception(f"Error in data for repository {repo_data['name']}: {e}")

# Main function to construct the data lake/monorepo JSON
def construct_data_lake_json(repo_names: List[str]):
    all_repos_json = []
    for repo_name in repo_names:
        repo_data = fetch_repo_data(repo_name)
        repo_json = construct_repo_json(repo_data)
        all_repos_json.append(repo_json)
    return all_repos_json

# Fetch and construct the JSON object for the data lake
data_lake_json = construct_data_lake_json(repo_names)

import json

# Fetch and construct the JSON object for the data lake
data_lake_json = construct_data_lake_json(repo_names)

# Write the JSON object to a file
with open('data_lake.json', 'w') as f:
    json.dump(data_lake_json, f, indent=2)

from pydantic import BaseModel, Field, root_validator
from typing import Any, List, Optional, Dict, Union
from datetime import datetime

# Define connection settings that could apply to any data source/storage
class ConnectionDetails(BaseModel):
    type: str  # E.g., 'S3', 'database', 'api', etc.
    identifier: str  # Name of the bucket, database, endpoint, etc.
    credentials: Dict[str, str]  # Could include tokens, keys, etc.
    additional_config: Optional[Dict[str, Any]]  # Any other necessary configuration

# Define a generic source model for extraction
class Source(BaseModel):
    source_id: str
    connection_details: ConnectionDetails
    data_format: Optional[str] = None  # E.g., 'csv', 'json', 'parquet', etc.
    extraction_method: Optional[str] = None  # E.g., 'full', 'incremental', etc.
    extraction_query: Optional[str] = None  # SQL query, API endpoint, etc.

# Define a generic transformation model
class Transformation(BaseModel):
    transformation_id: str
    description: Optional[str] = None
    logic: Optional[str] = None  # Reference to a transformation script or function
    dependencies: Optional[List[str]] = []  # IDs of transformations this one depends on

# Define a generic destination model for loading
class Destination(BaseModel):
    destination_id: str
    connection_details: ConnectionDetails
    data_format: Optional[str] = None

# Define a data lifecycle model
class DataLifecycle(BaseModel):
    stage: str  # E.g., 'raw', 'transformed', 'aggregated', etc.
    retention_policy: Optional[str] = None
    archival_details: Optional[str] = None
    access_permissions: Optional[List[str]] = None  # E.g., 'read', 'write', 'admin', etc.

# Define a job control model for ETL orchestration
class JobControl(BaseModel):
    job_id: str
    schedule: Optional[str] = None  # Cron expression for job scheduling
    dependencies: Optional[List[str]] = []  # IDs of jobs this one depends on
    alerting_rules: Optional[Dict[str, Any]] = None  # Alerting configuration

# Define a quality validation model
class QualityValidation(BaseModel):
    checks: Optional[Dict[str, Any]] = None  # E.g., {'null_check': 'No null values'}
    thresholds: Optional[Dict[str, float]] = None  # E.g., {'accuracy': 99.5}
    validation_rules: Optional[Dict[str, Any]] = None  # Custom validation rules

# Define an audit model for tracking ETL jobs
class Audit(BaseModel):
    timestamps: Dict[str, datetime] = Field(default_factory=lambda: {'created_at': datetime.now(), 'modified_at': datetime.now()})
    user_info: Optional[Dict[str, Any]] = None
    operation_type: Optional[str] = None  # E.g., 'ETL Process', 'Data Import', etc.

# Define a performance model for monitoring ETL jobs
class Performance(BaseModel):
    metrics: Optional[Dict[str, Any]] = None  # E.g., {'runtime_seconds': 120}
    logs: Optional[Dict[str, List[str]]] = None  # E.g., {'errors': ['error1', 'error2']}
    bottlenecks: Optional[List[str]] = None

# Define the main ETL process model
class ETLProcess(BaseModel):
    source: List[Source]
    transformations: List[Transformation]
    destination: List[Destination]
    lifecycle: DataLifecycle
    job_control: JobControl
    quality_validation: QualityValidation
    audit: Audit
    performance: Performance

@root_validator(pre=True)
def validate_structure(cls, values):
    """
    Custom validation to ensure the ETL process structure is coherent.
    This could include checks like ensuring all dependencies exist within
    the process definition, or that the data formats between source and
    destination are compatible.
    """
    # Example validation: Check if transformation dependencies are valid
    transformations = values.get('transformations', [])
    transformation_ids = {t.transformation_id for t in transformations}
    for transformation in transformations:
        if any(dep not in transformation_ids for dep in transformation.dependencies):
            raise ValueError('Invalid transformation dependency.')
    return values

if __name__ == "__main__":
    # Define an ETL process
    etl_process = ETLProcess(
        source=[
            Source(
                source_id='source1',
                connection_details=ConnectionDetails(
                    type='S3',
                    identifier='my-s3-bucket',
                    credentials={'access_key': 'ACCESSKEY', 'secret_key': 'SECRETKEY'},
                    additional_config={'region': 'us-east-1'}
                ),
                data_format='csv'
            )
        ],
        transformations=[
            Transformation(
                transformation_id='trans1',
                description='Normalize column names',
                logic='path/to/transformation/script.py',
                dependencies=[]
            )
        ],
        destination=[
            Destination(
                destination_id='dest1',
                connection_details=ConnectionDetails(
                    type='database',
                    identifier='my_database',
                    credentials={'username': 'user', 'password': 'pass'}
                ),
                data_format='table'
            )
        ],
        lifecycle=DataLifecycle(
            stage='raw',
            retention_policy='30 days',
            archival_details='Archive to Glacier after 1 year',
            access_permissions=['read', 'write']
        ),
        job_control=JobControl(
            job_id='job1',
            schedule='0 0 * * *',  # Run daily at midnight
            dependencies=[],
            alerting_rules={'email': 'alert@example.com'}
        ),
        quality_validation=QualityValidation(
            checks={'null_check': 'No null values allowed'},
            thresholds={'accuracy': 99.5},
            validation_rules={'regex': '^[a-zA-Z0-9]+$'}
        ),
        audit=Audit(
            user_info={'initiated_by': 'ETL System'},
            operation_type='Data Import'
        ),
        performance=Performance(
            metrics={'runtime_seconds': 120},
            logs={'errors': []},
            bottlenecks=['transformation_time']
        )
    )

    # Print the ETL process details
    print(etl_process.json(indent=2))




# Assuming repo_model.py and etl_model.py exist and are in the same directory or in PYTHONPATH
from repo_model import fetch_repo_data, construct_repo_json, construct_data_lake_json
from etl_model import ETLProcess, Source, Transformation, Destination, ConnectionDetails, DataLifecycle, JobControl, QualityValidation, Audit, Performance

# Function to run the ETL process
def run_etl_process(repo_names):
    # Extract data from GitHub repositories
    extracted_data = construct_data_lake_json(repo_names)
    
    # Here we would define the transformations required on the extracted data
    # For simplicity, we assume it's just passed through
    transformed_data = extracted_data
    
    # Load the transformed data into S3
    # Define the S3 bucket details
    s3_connection = ConnectionDetails(
        type='S3',
        identifier='my-s3-bucket',
        credentials={'access_key': 'ACCESSKEY', 'secret_key': 'SECRETKEY'},
        additional_config={'region': 'us-east-1'}
    )
    
    # Define the destination for the ETL process
    destination = Destination(
        destination_id='dest1',
        connection_details=s3_connection,
        data_format='json'
    )
    
    # Here we would actually write the code to load data into S3
    # For now, this is just a placeholder function call
    load_to_s3(transformed_data, destination)
    
    # Return a confirmation message
    return "ETL process completed successfully"

# Placeholder function for loading data to S3
def load_to_s3(data, destination):
    # This function would contain the actual logic to connect to S3 and upload the data
    # Currently, it's just a placeholder to illustrate the workflow
    print("Data would be loaded to S3 here")

# List of repository names to run the ETL process on
repo_names = [
    'cda.langchain-templates', 'cda.agents',
    # ... other repositories
]

# Call to run the ETL process
etl_confirmation = run_etl_process(repo_names)

# Output the result
print(etl_confirmation)

