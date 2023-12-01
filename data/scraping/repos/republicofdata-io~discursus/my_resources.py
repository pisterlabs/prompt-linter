from dagster import config_from_pkg_resources, file_relative_path
from dagster_snowflake import snowflake_resource
from dagster_dbt import dbt_cli_resource
from dagster_hex.resources import hex_resource

from discursus_data_platform.utils.resources import bigquery_resource, web_scraper_resource

from saf_aws import aws_resource
from saf_gdelt import gdelt_resource
from saf_openai import openai_resource


# dbt paths
DBT_PROFILES_DIR = file_relative_path(__file__, "./../../dp_data_warehouse/config/")
DBT_PROJECT_DIR = file_relative_path(__file__, "./../../dp_data_warehouse/")


# Configuration files
snowflake_configs = config_from_pkg_resources(
    pkg_resource_defs=[
        ('discursus_data_platform.utils.configs', 'snowflake_configs.yaml')
    ],
)
hex_configs = config_from_pkg_resources(
    pkg_resource_defs=[
        ('discursus_data_platform.utils.configs', 'hex_configs.yaml')
    ],
)
openai_configs = config_from_pkg_resources(
    pkg_resource_defs=[
        ('discursus_data_platform.utils.configs', 'openai_configs.yaml')
    ],
)


# Initiate resources
my_gdelt_resource = gdelt_resource.initiate_gdelt_resource.configured(None)
my_bigquery_resource = bigquery_resource.bigquery_client.configured(None)
my_snowflake_resource = snowflake_resource.configured(snowflake_configs)
my_dbt_resource = dbt_cli_resource.configured({
    "profiles_dir": DBT_PROFILES_DIR, 
    "project_dir": DBT_PROJECT_DIR})
my_openai_resource = openai_resource.initiate_openai_resource.configured(openai_configs)
my_aws_resource = aws_resource.initiate_aws_resource.configured(None)
my_web_scraper_resource = web_scraper_resource.initiate_web_scraper_resource.configured(None)
my_hex_resource = hex_resource.configured(hex_configs)