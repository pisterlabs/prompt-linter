from .NLP.Anthropic import Anthropic
from .NLP.OpenAI import OpenAI
from typing import Optional, Dict, List, Tuple
import inquirer
import os
import sys
import re
import json
from enum import Enum
import yaml
from ruamel.yaml import YAML
from dataclasses import dataclass
import itertools
import threading
import asyncio
import argparse
import subprocess

stdout_lock = threading.Lock()

class OAIRequest:
    def __init__(self, model: str, prompt: str, temperature: float, max_tokens: int):
        self.model = model
        self.prompt = prompt
        self.temperature = temperature
        self.max_tokens = max_tokens

class OAIRequestWithUserInfo:
    def __init__(self, prompt: str, email: str):
        self.prompt = prompt
        self.email = email

class OAIChoice:
    def __init__(self, text: str):
        self.text = text

class OAIResponse:
    def __init__(self, choices: List[OAIChoice]):
        self.choices = choices

class ColumnMetadata:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

class ArguParseException(Exception):
    def __init__(self, message):
        super().__init__(message)

class Depends:
    def __init__(self, nodes: Optional[List[str]] = None, macros: Optional[List[str]] = None):
        self.nodes = nodes
        self.macros = macros

class NodeMetadata:
    def __init__(
        self,
        original_file_path: str,
        patch_path: Optional[str] = None,
        compiled_code: Optional[str] = None,
        raw_code: Optional[str] = None,
        description: Optional[str] = None,
        database: Optional[str] = None,
        schema: Optional[str] = None,
        resource_type: Optional[str] = None,
        package_name: Optional[str] = None,
        path: Optional[str] = None,
        alias: Optional[str] = None,
        checksum: Optional[str] = None,
        config: Optional[str] = None,
        tags: Optional[str] = None,
        meta: Optional[str] = None,
        group: Optional[str] = None,
        docs: Optional[str] = None,
        build_path: Optional[str] = None,
        deferred: Optional[str] = None,
        unrendered_config: Optional[str] = None,
        created_at: Optional[str] = None,
        name: str = None,
        unique_id: str = None,
        fqn: List[str] = None,
        columns: Dict[str, ColumnMetadata] = None,
        depends_on: Optional[Depends] = None,
    ):
        self.original_file_path = original_file_path
        self.patch_path = patch_path
        self.compiled_code = compiled_code
        self.raw_code = raw_code
        self.description = description
        self.database = database
        self.schema = schema
        self.resource_type = resource_type
        self.package_name = package_name
        self.path = path
        self.alias = alias
        self.checksum = checksum
        self.config = config
        self.tags = tags
        self.meta = meta
        self.group = group
        self.docs = docs
        self.build_path = build_path
        self.deferred = deferred
        self.unrendered_config = unrendered_config
        self.created_at = created_at
        self.name = name
        self.unique_id = unique_id
        self.fqn = fqn
        self.columns = columns
        self.depends_on = depends_on

class Manifest:
    def __init__(self, nodes: Dict[str, NodeMetadata]):
        self.nodes = nodes

class KeyOrUserInfo:
    def __init__(self, key: Optional[str] = None, user_info: Optional[str] = None):
        self.key = key
        self.user_info = user_info

class Env:
    def __init__(
        self,
        api_key: KeyOrUserInfo,
        base_path: str,
        project_name: str,
        models: Optional[set[str]] = None,
        dry_run: bool = False,
        llm: str = "OpenAI",
    ):
        self.api_key = api_key
        self.base_path = base_path
        self.project_name = project_name
        self.models = models
        self.dry_run = dry_run
        self.llm = llm

@dataclass
class Arguments:
    pass

@dataclass
class Working_Directory(Arguments):
    path: str

class Gen_Undocumented(Arguments):
    pass

@dataclass
class Gen_Specific(Arguments):
    models_list: str

class DbtDocGen(Arguments):
    pass

class llm(Arguments):
    path: str

class Dry_Run(Arguments):
    pass

class GenMode(Enum):
    undocumented = 1
    specific = 2

documented_nodes = {}

class ArgsConfig:
    def __init__(self, working_directory: str, gen_mode: GenMode, dbtDocGen: bool, llm: str, dry_run: bool):
        self.working_directory = working_directory
        self.gen_mode = gen_mode
        self.dbtDocGen = dbtDocGen
        self.llm = llm
        self.dry_run = dry_run

class SummarizedResult:
    def __init__(
        self,
        patch_path: Optional[str],
        summary: str,
        original_file_path: str,
        column_summaries: Dict[str, str],
        name: str,
    ):
        self.patch_path = patch_path
        self.summary = summary
        self.original_file_path = original_file_path
        self.column_summaries = column_summaries
        self.name = name

class TooManyTokensError(Exception):
    pass 

async def gen_column_summaries(env: Env, node: NodeMetadata) -> Dict[str, str]:
    prefix = "[ai-gen] "
    
    async def mapper(k: str, column: ColumnMetadata) -> Tuple[str, str]:
        if env.llm.lower()[0] == "a":
            result = await Anthropic.run_request(env, Anthropic.mk_column_prompt(node, column, documented_nodes))
        else:
            result = await OpenAI.run_request(env, OpenAI.mk_column_prompt(node, column, documented_nodes))
            
        return (k, prefix + result)
    
    filtered_columns = {k: v for k, v in node.columns.items() if v.description == ""}
    
    result_seq = await asyncio.gather(*(mapper(k, col) for k, col in filtered_columns.items()))

    return dict(result_seq)    

async def open_ai_summarize(env: Env, reverse_deps: Dict[str, List[str]], node: NodeMetadata) -> Optional[SummarizedResult]:
    with stdout_lock:
        print(f"Generating docs for: {node.name}")

    summary_prefix = "This description is generated by an AI model. Take it with a grain of salt!\n"

    try:
        if env.llm.lower()[0]=="a":
            tbl_result, col_result = await asyncio.gather(
                Anthropic.run_request(env, Anthropic.mk_prompt(reverse_deps, node)),
                gen_column_summaries(env, node)
            )
        else:
            tbl_result, col_result = await asyncio.gather(
                OpenAI.run_request(env, OpenAI.mk_prompt(reverse_deps, node)),                
                gen_column_summaries(env, node)
            )

        return SummarizedResult(
            patch_path=node.patch_path,
            name=node.name,
            original_file_path=node.original_file_path,
            summary=summary_prefix + tbl_result,
            column_summaries=col_result
        )
    except TooManyTokensError:
        with stdout_lock:
            print(f"Prompt for {node.name} returned too many tokens to fit into GPT-3. Perhaps the SQL code or dependency map is too large?")
        return None
    except Exception as e:
        with stdout_lock:
            print(f"OAI request to {node.name} failed: {e}")
        raise e  # Reraise the exception to trigger a retry

def insert_column_description(env, node_result: SummarizedResult, col_map: Dict[str, str], model_node) -> None:
    model_node_ = model_node

    name_node = model_node_["name"]

    name = name_node

    if name not in col_map:
        return

    col_result = col_map[name]
    doc_name = f"tql_generated_doc__{node_result.name}__{name}"

    md_path = os.path.join(
        env.base_path,
        os.path.dirname(node_result.original_file_path),
        f"{doc_name}.md"
    )

    header = f"{{% docs {doc_name} %}}"
    footer = "{% enddocs %}"

    doc_content = "\n".join([header, col_result, footer])

    with stdout_lock:
        print(f"Writing new docs to: {md_path}")

    if env.dry_run:
        print(doc_content)
    else:
        with open(md_path, "w") as f:
            f.write(doc_content)

    ## Removing the `columns` from dictionary only to be added after `description`
    model_node_.pop("description", None)
    columns = model_node_.pop("columns", None)
    model_node_["description"] = f"{{{{ doc(\"{doc_name}\") }}}}"
    model_node_["columns"] = columns

def insert_description(env, node_map: Dict[str, SummarizedResult], model_node) -> None:    
    model_node_ = model_node

    name_node = model_node_["name"]

    name = name_node

    if name not in node_map:        
        return
        
    node = node_map[name]
    doc_name = f"tql_generated_doc__{node.name}"

    md_path = os.path.join(
        env.base_path,
        os.path.dirname(node.original_file_path),
        f"{doc_name}.md"
        )

    header = f"{{% docs {doc_name} %}}"
    footer = "{% enddocs %}"

    doc_content = "\n".join([header, node.summary, footer])

    with stdout_lock:
        print(f"Writing new docs to: {md_path}")

    if "columns" in model_node_:
        cols_node = model_node_["columns"]
    
        for col in cols_node:
            insert_column_description(env, node, node.column_summaries, col)

    if env.dry_run:
        print(doc_content)
    else:
        with open(md_path, "w") as f:
            f.write(doc_content)

    model_node_.pop("description", None)
    model_node_["description"] = f"{{{{ doc(\"{doc_name}\") }}}}"
        
def insert_docs(env: Env, patch_path_may: Optional[str], nodes: List[SummarizedResult]) -> None:
    if patch_path_may is None:        
        return
    
    path = os.path.join(env.base_path, patch_path_may.replace(f"{env.project_name}://", ""))

    with open(path, "r") as f:
        contents = f.read()

    deserializer = yaml.SafeLoader(contents)
    config = deserializer.get_single_data()

    data = yaml.safe_load(contents)
    models = data["models"]

    result_map = {n.name: n for n in nodes}

    models_node = config["models"]

    for model_obj in models_node:
        model_name = model_obj['name']
        model = model_obj.get('columns', [])  # If 'columns' doesn't exist, default to an empty list.        
        if model_name in result_map:
            insert_description(env, result_map, model_obj)

    with stdout_lock:
        print(f"Adding description to {len(nodes)} models in {path}")

    if env.dry_run:
        ## Prints to console
        YAML().dump(config, sys.stdout)
    else:
        ## Writes to yaml file
        with open(path, "w") as f:
            YAML().dump(config, f)

def read_project_config(base_path: str) -> str:
    path = os.path.join(base_path, "dbt_project.yml")
    
    with open(path, "r") as f:
        contents = f.read()

    deserializer = yaml.SafeLoader(contents)
    config = deserializer.get_single_data()

    data = yaml.safe_load(contents)
    name_node = data["name"]
    
    return name_node

def is_model(name: str) -> bool:
    node_type = name.split('.')[0]
    return node_type == "model"

def should_write_doc(env: Env, pair: Tuple[str, NodeMetadata]) -> bool:
    def pred(nm):
        if env.models is None:
            return pair[1].description == ""
        return nm in env.models

    has_patch_path = pair[1].patch_path is not None

    cond = is_model(pair[0]) and pred(pair[1].name)

    if not has_patch_path and cond:
        print(f"Model {pair[0]} doesn't appear to be declared in a .yml file. Generating docs isn't yet supported for models without a corresponding yaml declaration.")
        user_input = input("Do you want to generate the missing .yml file? Y/n: ")
        if user_input.lower() == 'y':
            generateYaml(env,pair[1])
            return True

    return has_patch_path and cond

def generateYaml(env: Env,node_metadata: NodeMetadata):
    # Get the path to the .yml file
    yaml_file_path = os.path.join(env.base_path, os.path.dirname(node_metadata.original_file_path), node_metadata.name + '.yml')

    try:
        catalog_path = os.path.join(env.base_path, "target", "catalog.json")
        catalog = getCatalog(catalog_path)
    except Exception as e:
        print("catalog.json deserialization failed")
        raise e

    # Extract column names
    column_list = []
    for model_name, (columns, types) in catalog.items():
        if model_name == node_metadata.name:
            column_list = [{'name': col_name} for col_name in columns]

    # Create the data structure for the YAML file
    data = {
        "version": 2,
        "models": [
            {
                "name": node_metadata.name,
                "columns": column_list
            }
        ]
    }

    # Write the data structure to the YAML file
    with open(yaml_file_path, 'w') as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

    print(f"YAML file successfully generated at {yaml_file_path}!")

def mk_reverse_dependency_map(nodes: Dict[str, NodeMetadata]) -> Dict[str, List[str]]:
    ans: Dict[str, List[str]] = {}

    def folder(nm: str, metadata: NodeMetadata) -> None:
        nodes = metadata.depends_on.nodes if metadata.depends_on else []

        if is_model(nm):
            for model_dep in nodes:
                if model_dep in ans:
                    ans[model_dep].append(nm)
                else:
                    ans[model_dep] = [nm]

    for key, value in nodes.items():
        folder(key, value)

    return ans

class ApiKeyNotFound(Exception):
    pass

def parse_args(argv) -> ArgsConfig:
    parser = argparse.ArgumentParser(prog="DbtHelper")
    
    parser.add_argument("--working-directory", type=str, default="./", help="Specify the working directory.")
    
    gen_mode_group = parser.add_mutually_exclusive_group()
    gen_mode_group.add_argument("--undocumented", dest="gen_mode", action="store_const", const=GenMode.undocumented, default=GenMode.undocumented, help="Use undocumented gen mode.")
    gen_mode_group.add_argument("--specific", dest="gen_mode", type=lambda s: GenMode.specific(list(s.split(','))), help="Use specific gen mode with a list of models.")
    
    parser.add_argument("--dbtDocGen", type=bool, default=True, help="Run command `dbt docs generate` automatically.")

    parser.add_argument("--llm", type=str, default="OpenAI", help="Specify the preferred LLM, OpenAI or Anthropic. Default is OpenAI. (you can type just 'o' or 'a')")

    parser.add_argument("--dry-run", action="store_true", help="Enable dry run mode.")
    
    args = parser.parse_args(argv)

    return ArgsConfig(working_directory=args.working_directory, gen_mode=args.gen_mode, dbtDocGen=args.dbtDocGen, llm=args.llm, dry_run=args.dry_run)

def parse_columns(json_data: Dict[str, dict]) -> Dict[str, ColumnMetadata]:
    columns = {}
    for column_name, column_data in json_data.items():
        if 'depends_on' in column_data:
            column_data['depends_on'] = Depends(**column_data['depends_on'])

        # Extract only the fields needed for ColumnMetadata
        column_data_subset = {
            k: column_data.get(k) for k in
            ['name', 'description']
        }

        columns[column_name] = ColumnMetadata(**column_data_subset)
    return columns

def parse_node_metadata(json_data: Dict[str, dict]) -> Dict[str, NodeMetadata]:
    node_metadata = {}
    for node_id, node_data in json_data.items():
        columns = parse_columns(node_data.get('columns', {}))
        node_data['columns'] = columns
        if 'depends_on' in node_data:
            node_data['depends_on'] = Depends(**node_data['depends_on'])

        # Extract only the fields needed for NodeMetadata
        node_data_subset = {
            k: node_data.get(k) for k in
            ['original_file_path', 'patch_path', 'compiled_code', 'raw_code', 'description',
             'database', 'schema', 'resource_type', 'package_name', 'path', 'alias', 'checksum',
             'config', 'tags', 'meta', 'group', 'docs', 'build_path', 'deferred', 'unrendered_config',
             'created_at', 'name', 'unique_id', 'fqn', 'columns', 'depends_on']
        }

        node_metadata[node_id] = NodeMetadata(**node_data_subset)
    return node_metadata

def parse_manifest(json_data: Dict[str, dict]) -> Manifest:
    node_metadata = parse_node_metadata(json_data['nodes'])
    return Manifest(nodes=node_metadata)

def read_json_file(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def load_manifest_from_json(file_path: str) -> Manifest:
    json_data = read_json_file(file_path)
    return parse_manifest(json_data)

def getCatalog(json_file: str) -> Dict[str, List[str]]:
    with open(json_file) as f:
        data = json.load(f)
    
    models_columns = dict()
    
    # Loop through nodes, there might be multiple models
    for node in data['nodes'].values():
        # Get model name
        model_name = node['metadata']['name']
        
        # Get column names
        columns = [column_info['name'] for column_info in node['columns'].values()]
        
        # Get column types
        types = [column_info['type'] for column_info in node['columns'].values()]
        
        # Add model and its columns to the result
        models_columns[model_name] = (columns, types)
    
    return models_columns

def get_nodes_with_description(manifest_path: str) -> Dict[str, NodeMetadata]:
    manifest = load_manifest_from_json(manifest_path)
    nodes_with_description = {}
    for node_name, node_metadata in manifest.nodes.items():
        if node_metadata.description != "":
            nodes_with_description[node_name] = node_metadata
    return nodes_with_description

class UserInfo:
    def __init__(self, email: str):
        self.email = email

class Key:
    def __init__(self, key: str):
        self.key = key

def run_dbt_docs_generate(path_to_dbt_project, arg_dbtDocGen):
    if arg_dbtDocGen:
        try:
            print("Running DBT docs generate...")
            subprocess.check_output(['dbt', 'docs', 'generate'], cwd=path_to_dbt_project)
            print("DBT docs successfully generated!")
            return ""
        except subprocess.CalledProcessError as e:
            print("Could not generate DBT docs.")
            print("Error:")
            print(e.output)            
            error = str(e.output)
            return error
    else:
        print("DBT docs generation is turned off... Make sure to run `dbt docs generate` before and after running this tool.")

### Metrics ###
async def generateMetrics(env: Env, selected_nodes: Dict[str, NodeMetadata]):
    for node_name, node_metadata in selected_nodes:
        print("Generating metrics for model: " + node_name)        

        try:
            catalog_path = os.path.join(env.base_path, "target", "catalog.json")
            catalog = getCatalog(catalog_path)
        except Exception as e:
            print("catalog.json deserialization failed")
            raise e

        # Extract column names and types for the model
        column_list = []
        for model_name, (column_names, column_types) in catalog.items():
            if model_name == node_metadata.name:
                column_list = [{col_name : col_type} for col_name, col_type in zip(column_names, column_types)]

        #Asking the model for a base requirement for the metric
        if env.llm.lower()[0]=="a":
            promptDes = Anthropic.promptDesignMetrics(node_metadata, column_list)
            res1 = await Anthropic.run_request(env, promptDes)
        else:
            promptDes = OpenAI.promptDesignMetrics(node_metadata, column_list)        
            res1 = await OpenAI.run_request(env, promptDes)
        
        queries = [block.strip() for block in res1.split(";")]        
        
        # Create the data structure for the YAML file
        for rawquery in queries:
            if len(rawquery.split(":")) == 2:
                metric_name, query = rawquery.split(":")
                if env.llm.lower()[0]=="a":
                    prompt = Anthropic.promptGenMetrics(node_metadata, column_list, query, metric_name)
                else:
                    prompt = OpenAI.promptGenMetrics(node_metadata, column_list, query, metric_name)
                
                print("Generating metric: " + metric_name + "\n" + "With the prompt: " + query)

                #Call OpenAI API
                if env.llm.lower()[0]=="a":
                    result = await Anthropic.run_request(env, prompt)
                else:
                    result = await OpenAI.run_request(env, prompt)
                
                result = result.replace("\\n", "").replace("\\\\", "")

                # Write the data structure to the YAML file
                yaml_file_path = os.path.join(env.base_path, os.path.dirname(node_metadata.original_file_path), "tql_genmetric_" + metric_name + '.yml')            
                with open(yaml_file_path, 'w') as yaml_file:
                    yaml_file.write(result)                
                    print(f"Metrics YAML file successfully generated at {yaml_file_path}!")

                if env.llm.lower()[0]=="a":
                    sqlPrompt = Anthropic.promptGenMetricSQL(metric_name, result)
                    sqlText = await Anthropic.run_request(env, sqlPrompt)
                else:
                    sqlPrompt = OpenAI.promptGenMetricSQL(metric_name, result)
                    sqlText = await OpenAI.run_request(env, sqlPrompt)                

                # Write the query in to a SQL file
                sql_file_path = os.path.join(env.base_path, os.path.dirname(node_metadata.original_file_path), "metric_" + metric_name + '.sql')            
                with open(sql_file_path, 'w') as sql_file:
                    sql_file.write(sqlText)
                    print(f"Metrics SQL file successfully generated at {sql_file_path}!")

            else:
                continue        

async def reviewMetrics(errorMessage: str, env: Env):
    filename = re.search(r'Error reading .*?: (.*?) -', errorMessage)
    if filename:
        file = filename.group(1).rstrip().replace("\\\\", "\\")
        print("Reviewing metrics for file: " + file)
        file_path = os.path.join(env.base_path, 'models\\', file)

        with open(file_path, "r") as f:
            contents = f.read()

        if env.llm.lower()[0]=="a":
            fixPrompt = Anthropic.promptFixMetric(errorMessage, contents)
            fixedText = await Anthropic.run_request(env, fixPrompt)
        else:
            fixPrompt = OpenAI.promptFixMetric(errorMessage, contents)
            fixedText = await OpenAI.run_request(env, fixPrompt)

        result = fixedText

        with open(file_path, 'w') as yaml_file:
            yaml_file.write(result)                
            print(f"Metrics YAML file was successfully rewritten at {file_path}!")

### Metrics ###

async def main(argv) -> int:
    DDG_error = "error message"    
    try:
        args_env = parse_args(sys.argv[1:])

        DDG_error = run_dbt_docs_generate(args_env.working_directory, args_env.dbtDocGen)
        while DDG_error != "":
            if (input("Would you like to try again? (y/n) ").lower() == "y"):
                DDG_error = run_dbt_docs_generate(args_env.working_directory, args_env.dbtDocGen)
            else:
                DDG_error = ""

        manifest_path = os.path.join(args_env.working_directory, "target", "manifest.json")        

        try:
            manifest = load_manifest_from_json(manifest_path)
        except Exception as e:
            print("manifest.json deserialization failed")
            raise e        

        try:            
            project_name = read_project_config(args_env.working_directory)
        except Exception as e:
            print("Reading dbt_project.yml failed. Please re-run from a dbt project root.")
            raise e

        if args_env.llm.lower()[0]=="a":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            api_key = os.environ.get("OPENAI_API_KEY")

        if api_key is None:
            if args_env.llm.lower()[0]=="a":
                print("You are using the Anthropic model, please specify your API Key in the environment variable: ANTHROPIC_API_KEY.")                
            else:
                print("You haven't specified an API Key. No worries, this one's on TextQL!")
                print("In return, please type your email address. We don't collect any other data, nor sell your email to third parties.")
                print("If you're okay with this, press enter. Otherwise, type 'no' and set the OPENAI_API_KEY environment variable.")
                email = input("Email (type no to abort): ")

                if email == "no":
                    raise ApiKeyNotFound()
                else:
                    user_info = UserInfo(email)
        else:
            user_info = Key(api_key)

        models = None if args_env.gen_mode == GenMode.undocumented else set(args_env.gen_mode.value)

        documented_nodes = get_nodes_with_description(manifest_path)

        init = (manifest, Env(api_key=user_info, base_path=args_env.working_directory, project_name=project_name, models=models, llm=args_env.llm, dry_run=args_env.dry_run))

    except ArguParseException as e:
        print(e.message)
        return 1
    except Exception as e:
        print("Initialization failed. Aborting")
        print(e)
        return 1

    manifest, env = init

    if env.dry_run:
        print("Dry Run. Results will not be written.")
    
    r_deps = mk_reverse_dependency_map(manifest.nodes)

    ### Model Selection ###
    options = [(k, (k, v)) for k, v in manifest.nodes.items()]

    questions = [
        inquirer.Checkbox('models',
                        message='Select the models you want to document: (Press Spacebar to select, Enter to confirm) ',
                        choices=options,
                        ),
    ]

    answers = inquirer.prompt(questions)

    selected_nodes = dict(answers['models']).items()
    ### Model Selection ###

    nodes_to_process = [pair for pair in selected_nodes if should_write_doc(env, pair)]

    #Assigning generated .yml files to nodes
    for node in nodes_to_process:
        if node[1].patch_path is None:
            node[1].patch_path = os.path.join(env.base_path, os.path.dirname(node[1].original_file_path), node[1].name + ".yml")
    
    summarized_nodes = await asyncio.gather(*[open_ai_summarize(env, r_deps, pair[1]) for pair in nodes_to_process])
    summarized_nodes = [node for node in summarized_nodes if node is not None]
    
    for patch_path, group in itertools.groupby(sorted(summarized_nodes, key=lambda x: x.patch_path), key=lambda x: x.patch_path):
        insert_docs(env, patch_path, list(group))

    # Metrics generation    
    makeMetrics = input("Do you want to generate the metrics for the selected models? Y/n: ")
    if makeMetrics.lower() == 'y':
        await generateMetrics(env, nodes_to_process)
        #await generateMetrics(env, selected_nodes)
    # Metrics generation
    
    DDG_error = run_dbt_docs_generate(args_env.working_directory, args_env.dbtDocGen)    
    counter = 0
    while DDG_error != "" and counter < 3:
        counter += 1
        if ("Syntax error near line" in DDG_error):
            await reviewMetrics(DDG_error, env)
            DDG_error = run_dbt_docs_generate(args_env.working_directory, args_env.dbtDocGen)
        elif ("Compilation Error" in DDG_error):
            await reviewMetrics(DDG_error, env)
            DDG_error = run_dbt_docs_generate(args_env.working_directory, args_env.dbtDocGen)
        elif (input("Would you like to try again? (y/n) ").lower() == "y"):
            DDG_error = run_dbt_docs_generate(args_env.working_directory, args_env.dbtDocGen)
        else:
            DDG_error = ""

        if counter == 3:
            print("Too many errors. Aborting.")
    return 0

async def async_main(argv):
    await main(argv)

def run_async_main():
    if sys.platform == "win32" and sys.version_info >= (3, 8):        
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(async_main(sys.argv[1:]))

if __name__ == "__main__":
    run_async_main(sys.argv[1:])        
    run_async_main()