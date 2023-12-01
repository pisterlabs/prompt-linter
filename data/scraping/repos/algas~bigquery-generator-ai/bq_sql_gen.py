import sys
import os
from langchain import PromptTemplate, OpenAI, LLMChain
from google.cloud import bigquery
import json
import argparse

TEMPLATE = '''
Write a BigQuery SQL that achieves the following.
```
{{ content }}
```

The format of the target tables is as follows.
```json
{{ schema }}
```

Output the SQL in raw text.
    '''

def get_schema(table_name: str) -> str:
    client = bigquery.Client()
    table = client.get_table(table_name)
    project_id = table.project
    dataset_id = table.dataset_id
    table_id = table.table_id
    schema = list(map(lambda x: x.to_api_repr(), table.schema))
    return {'project':project_id,'dataset':dataset_id,'table':table_id,'schema':schema}

def get_schemas(table_names: list[str]):
    return json.dumps([get_schema(n) for n in table_names])

def predict(content: str, table_names: list[str], verbose: bool = False):
    prompt = PromptTemplate(
        input_variables=["content","schema"],
        template=TEMPLATE,
        template_format='jinja2',
    )
    llm_chain = LLMChain(
        llm=OpenAI(temperature=0), 
        prompt=prompt, 
        verbose=verbose,
    )
    return llm_chain.predict(content=content, schema=get_schemas(table_names))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BigQuery SQL generator with ChatGPT.')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('content')
    parser.add_argument('table_name', nargs='+')
    args = parser.parse_args()
    print(predict(args.content, args.table_name, args.verbose))
