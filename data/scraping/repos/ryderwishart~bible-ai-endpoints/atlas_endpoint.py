from pathlib import Path
from modal import Image, Secret, Stub, web_endpoint

image = Image.debian_slim().pip_install(
    "langchain~=0.0.209",
    "tiktoken==0.3.0",
    "sentence_transformers~=2.2.2",
    "openai~=0.27.8",
    "httpx~=0.23.3",
    "gql~=3.4.1",
    "requests-toolbelt~=1.0.0",
)
stub = Stub(
    name="atlas-agent",
    image=image,
    secrets=[Secret.from_name("openai-qa")],
)

import requests
import logging

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

import requests

# import yaml # NOTE: for testing whether YAML schema is more efficient token-wise

endpoint = "https://macula-atlas-api-qa-25c5xl4maa-uk.a.run.app/graphql/"


def get_macula_atlas_schema():
    """Query the macula atlas api for its schema"""
    logger.info("Querying MACULA ATLAS API for its schema...")

    global endpoint
    query = """
    query IntrospectionQuery {
        __schema {
            types {
                name
                kind
                fields {
                    name
                    type {
                        name
                        kind
                        ofType {
                            name
                            kind
                        }
                    }
                }
            }
        }
    }"""
    request = requests.post(endpoint, json={"query": query})
    json_output = request.json()

    # Simplify the schema
    simplified_schema = {}
    for type_info in json_output["data"]["__schema"]["types"]:
        if not type_info["name"].startswith("__"):
            fields = type_info.get("fields")
            if fields is not None and fields is not []:
                simplified_schema[type_info["name"]] = {
                    "kind": type_info["kind"],
                    "fields": ", ".join(
                        [
                            field["name"]
                            for field in fields
                            if not field["name"].startswith("__")
                        ]
                    ),
                }
            else:
                simplified_schema[type_info["name"]] = {
                    "kind": type_info["kind"],
                }

    return simplified_schema

    # Convert the simplified schema to YAML
    # yaml_output = yaml.dump(simplified_schema, default_flow_style=False)

    # return yaml_output


from langchain import OpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.utilities import GraphQLAPIWrapper


@stub.function()
@web_endpoint(method="GET")
def answer_question_using_atlas(query: str, show_sources: bool = False):
    global endpoint

    llm = OpenAI(temperature=0)

    logger.info("Loading tools...")
    tools = load_tools(
        ["graphql"],
        graphql_endpoint=endpoint,
        llm=llm,
    )

    logger.info("Initializing agent...")
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    logger.info("Querying ATLAS db...")
    graphql_fields = get_macula_atlas_schema()
    examples = """
    ## All features and instances for 2 Corinthians 8:2
    query AnnotationFeatures {
        annotationFeatures(filters: {reference: "2CO 8:2"}) {
        label
            uri
            instances(filters: {reference: "2CO 8:2"}) {
                uri
                tokens {
                    ref
                }
            }
        }
    }
    
    ## First 10 annotations with featureLabel "Main clauses"
    query Annotations {
        annotations(
            filters: { featureLabel: "Main clauses" }
            pagination: { limit: 10, offset: 0 }
        ) {
            uri
            depth
            tokens {
                ref
            }
        }
    }

    ## All features and instances for John 1:12
    query {
        annotationFeatures(filters: {reference: "JHN 1:12"}) {
            label
            uri
            instances(filters: {reference: "JHN 1:12"}) {
                uri
                tokens {
                    ref
                }
            }
        }
    }
    
    Note that the bible reference is repeated for features and for instances. If searching for features without a passage reference filter, be sure to use pagination to limit the number of results returned!
"""

    prompt = f"""Here are some example queries for the graphql endpoint described below:
    {examples}

    Answer the following question: {query} in the graphql database that has this schema {graphql_fields}"""

    result = agent.run(prompt)

    logger.info(f"Result: {result}")
    return result
