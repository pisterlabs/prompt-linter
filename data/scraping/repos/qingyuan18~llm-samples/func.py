import boto3
import json
import sagemaker
import requests
import time
from collections import defaultdict
from requests_aws4auth import AWS4Auth
import os
from typing import Dict
from typing import Any, Dict, List, Optional
from botocore.config import Config
from opensearchpy import OpenSearch, RequestsHttpConnection
from langchain.vectorstores import OpenSearchVectorSearch
from langchain import PromptTemplate, SagemakerEndpoint
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain import LLMChain
from langchain.tools.base import BaseTool, Tool, tool
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType
from typing import Optional, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
    CallbackManagerForChainRun
)
from langchain.llms.bedrock import Bedrock



aos_endpoint="vpc-llm-rag-aos-seg3mzhpp76ncpxezdqtcsoiga.us-west-2.es.amazonaws.com"
embedding_endpoint_name="bge-zh-15-2023-09-25-07-02-01-080-endpoint"
sqlcoder_endpoint_name="sqlcoder-2023-10-07-01-50-46-950-endpoint"
dburi = "database-us-west-2-demo.cluster-c1qvx9wzmmcz.us-west-2.rds.amazonaws.com"
region='us-west-2'
username="admin"
passwd="(OL>0p;/"


credentials = boto3.Session().get_credentials()
region = boto3.Session().region_name
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, 'es', session_token=credentials.token)
pwdauth = (username, passwd)
size=10

sess = sagemaker.Session()
sagemaker_session_bucket=None
if sagemaker_session_bucket is None and sess is not None:
    sagemaker_session_bucket = sess.default_bucket()
try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']
sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)
sm_client = boto3.client("sagemaker-runtime")

### for sqlcoder
class TextGenContentHandler2(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        input_str = json.dumps({
                "inputs": prompt,
                "parameters": model_kwargs
            })
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        #print(response_json)
        #sql_result=response_json["outputs"].split("```sql")[-1].split("```")[0].split(";")[0].strip().replace("\\n"," ") + ";"
        sql_result=response_json["outputs"]
        return sql_result

content_hander2=TextGenContentHandler2()
parameters = {
  "max_new_tokens": 350,
  #"do_sample":False,
  #"temperatual" : 0
  #"no_repeat_ngram_size": 2,
}
sm_sql_llm=SagemakerEndpoint(
        endpoint_name=sqlcoder_endpoint_name,
        region_name="us-west-2", 
        model_kwargs=parameters,
        content_handler=content_hander2
)



class CustomerizedSQLDatabaseChain(SQLDatabaseChain):

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        input_text = f"{inputs[self.input_key]}\nSQLQuery:"
        _run_manager.on_text(input_text, verbose=self.verbose)
        # If not present, then defaults to None which is all tables.
        table_names_to_use = inputs.get("table_names_to_use")
        if table_names_to_use is None:
            table_names_to_use=self.database._include_tables
            print("table_names_to_use==")
            print(table_names_to_use)
        table_info = self.database.get_table_info(table_names=table_names_to_use)

        llm_inputs = {
            "input": input_text,
            "top_k": str(self.top_k),
            "dialect": self.database.dialect,
            "table_info": table_info,
            #"stop": ["\nSQLResult:"],
        }
        intermediate_steps: List = []
        try:
            intermediate_steps.append(llm_inputs)  # input: sql generation
            sql_cmd = self.llm_chain.predict(
                callbacks=_run_manager.get_child(),
                **llm_inputs,
            ).strip()
            print("orginal sql_cmd=="+sql_cmd)
            if self.return_sql:
                return {self.output_key: sql_cmd}
            if not self.use_query_checker:
                _run_manager.on_text(sql_cmd, color="green", verbose=self.verbose)
                intermediate_steps.append(
                    sql_cmd
                )  # output: sql generation (no checker)
                #########定制sqlcoder 模型输出##############
                pattern = r"SQLQuery: (.*?)\n"
                matches = re.findall(pattern, sql_cmd)
                match = matches[1]
                sql_cmd = match
                #print("query sql=="+sql_cmd)

                intermediate_steps.append({"sql_cmd": sql_cmd})  # input: sql exec
                result = self.database.run(sql_cmd)
                intermediate_steps.append(str(result))  # output: sql exec
            else:
                query_checker_prompt = self.query_checker_prompt or PromptTemplate(
                    template=QUERY_CHECKER, input_variables=["query", "dialect"]
                )
                query_checker_chain = LLMChain(
                    llm=self.llm_chain.llm, prompt=query_checker_prompt
                )
                query_checker_inputs = {
                    "query": sql_cmd,
                    "dialect": self.database.dialect,
                }
                checked_sql_command: str = query_checker_chain.predict(
                    callbacks=_run_manager.get_child(), **query_checker_inputs
                ).strip()
                intermediate_steps.append(
                    checked_sql_command
                )  # output: sql generation (checker)
                _run_manager.on_text(
                    checked_sql_command, color="green", verbose=self.verbose
                )
                intermediate_steps.append(
                    {"sql_cmd": checked_sql_command}
                )  # input: sql exec
                result = self.database.run(checked_sql_command)
                intermediate_steps.append(str(result))  # output: sql exec
                sql_cmd = checked_sql_command

            _run_manager.on_text("\nSQLResult: ", verbose=self.verbose)
            _run_manager.on_text(result, color="yellow", verbose=self.verbose)
            # If return direct, we just set the final result equal to
            # the result of the sql query result, otherwise try to get a human readable
            # final answer
            if self.return_direct:
                final_result = result
            else:
                _run_manager.on_text("\nAnswer:", verbose=self.verbose)
                input_text += f"{sql_cmd}\nSQLResult: {result}\nAnswer:"
                llm_inputs["input"] = input_text
                intermediate_steps.append(llm_inputs)  # input: final answer
                final_result = self.llm_chain.predict(
                    callbacks=_run_manager.get_child(),
                    **llm_inputs,
                ).strip()
                intermediate_steps.append(final_result)  # output: final answer
                _run_manager.on_text(final_result, color="green", verbose=self.verbose)
            chain_result: Dict[str, Any] = {self.output_key: final_result}
            if self.return_intermediate_steps:
                chain_result[INTERMEDIATE_STEPS_KEY] = intermediate_steps
            return chain_result
        except Exception as exc:
            # Append intermediate steps to exception, to aid in logging and later
            # improvement of few shot prompt seeds
            exc.intermediate_steps = intermediate_steps  # type: ignore
            raise exc

def run_query(query):
    infos=""
    table_name=""
    question=""
    ####post process agent action output######
    if "\n" in query:
        infos = query.split("\n")
    elif "," in query:
        infos = query.split(",")

    if ":" in infos[0]:
        table_name=infos[0].split(":")[1]
    else:
        table_name=infos[0]

    if ":" in infos[1]:
        question=infos[1].split(":")[1]
    else:
        question=infos[1]

    table_name=table_name.strip()
    question=question.strip()

    db_chain = CustomerizedSQLDatabaseChain.from_llm(sm_sql_llm, db, verbose=True, return_sql=True,return_intermediate_steps=False)

    if table_name is not None:
        db_chain.database._include_tables=[table_name]
    response=db_chain.run(question)
    return response



def aos_knn_search_v2(client, field,q_embedding, index, size=1):
    if not isinstance(client, OpenSearch):
        client = OpenSearch(
            hosts=[{'host': aos_endpoint, 'port': 443}],
            http_auth = pwdauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
    query = {
        "size": size,
        "query": {
            "knn": {
                field: {
                    "vector": q_embedding,
                    "k": size
                }
            }
        }
    }
    opensearch_knn_respose = []
    query_response = client.search(
        body=query,
        index=index
    )
    opensearch_knn_respose = [{'idx':item['_source'].get('idx',1),'database_name':item['_source']['database_name'],'table_name':item['_source']['table_name'],'exactly_query_text':item['_source']['exactly_query_text'],"score":item["_score"]}  for item in query_response["hits"]["hits"]]
    return opensearch_knn_respose


def aos_knn_search(client, field,q_embedding, index, size=1):
    if not isinstance(client, OpenSearch):
        client = OpenSearch(
            hosts=[{'host': aos_endpoint, 'port': 443}],
            http_auth = pwdauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
    query = {
        "size": size,
        "query": {
            "knn": {
                field: {
                    "vector": q_embedding,
                    "k": size
                }
            }
        }
    }
    opensearch_knn_respose = []
    query_response = client.search(
        body=query,
        index=index
    )
    opensearch_knn_respose = [{'idx':item['_source'].get('idx',1),'database_name':item['_source']['database_name'],'table_name':item['_source']['table_name'],'query_desc_text':item['_source']['query_desc_text'],"score":item["_score"]}  for item in query_response["hits"]["hits"]]
    return opensearch_knn_respose

def aos_reverse_search(client, index_name, field, query_term, exactly_match=False, size=1):
    """
    search opensearch with query.
    :param host: AOS endpoint
    :param index_name: Target Index Name
    :param field: search field
    :param query_term: query term
    :return: aos response json
    """
    if not isinstance(client, OpenSearch):
        client = OpenSearch(
            hosts=[{'host': aos_endpoint, 'port': 443}],
            http_auth = pwdauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
    query = None
    if exactly_match:
        query =  {
            "query" : {
                "match_phrase":{
                    "doc": {
                        "query": query_term,
                        "analyzer": "ik_smart"
                      }
                }
            }
        }
    else:
        query = {
            "size": size,
            "query": {
                "query_string": {
                "default_field": "query_desc_text",
                "query": query_term
              }
            },
           "sort": [{
               "_score": {
                   "order": "desc"
               }
           }]
    }
    query_response = client.search(
        body=query,
        index=index_name
    )
    result_arr = [{'idx':item['_source'].get('idx',1),'database_name':item['_source']['database_name'],'table_name':item['_source']['table_name'],'query_desc_text':item['_source']['query_desc_text'],"score":item["_score"]}  for item in query_response["hits"]["hits"]]
    return result_arr




def get_vector_by_sm_endpoint(questions, sm_client, endpoint_name):
    parameters = {
    }

    response_model = sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(
            {
                "inputs": questions,
                "parameters": parameters,
                "is_query" : True,
                "instruction" :  "为这个句子生成表示以用于检索相关文章："
            }
        ),
        ContentType="application/json",
    )
    json_str = response_model['Body'].read().decode('utf8')
    json_obj = json.loads(json_str)
    embeddings = json_obj['sentence_embeddings']
    return embeddings




def get_topk_items(opensearch_query_response, topk=5):
    opensearch_knn_nodup = []
    unique_ids = set()
    for item in opensearch_query_response:
        if item['id'] not in unique_ids:
            opensearch_knn_nodup.append(item['score'], item['idx'],item['database_name'],item['table_name'],item['query_desc_text'])
            unique_ids.add(item['id'])
    return opensearch_knn_nodup



def k_nn_ingestion_by_aos(docs,index,hostname,username,passwd):
    auth = (username, passwd)
    search = OpenSearch(
        hosts = [{'host': aos_endpoint, 'port': 443}],
        http_auth = awsauth ,
        #http_auth = auth ,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    for doc in docs:
        query_desc_embedding = doc['query_desc_embedding']
        database_name = doc['database_name']
        table_name = doc['table_name']
        query_desc_text = doc["query_desc_text"]
        document = { "query_desc_embedding": query_desc_embedding, 'database_name':database_name, "table_name": table_name,"query_desc_text":query_desc_text}
        search.index(index=index, body=document)

def k_nn_ingestion_by_aos_v2(docs,index,hostname,username,passwd):
    auth = (username, passwd)
    search = OpenSearch(
        hosts = [{'host': aos_endpoint, 'port': 443}],
        http_auth = awsauth ,
        #http_auth = auth ,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    for doc in docs:
        exactly_query_embedding = doc['exactly_query_embedding']
        database_name = doc['database_name']
        table_name = doc['table_name']
        exactly_query_text = doc["exactly_query_text"]
        document = { "exactly_query_embedding": exactly_query_embedding, 'database_name':database_name, "table_name": table_name,"exactly_query_text":exactly_query_text}
        search.index(index=index, body=document)


def get_bedrock_client(
        assumed_role: Optional[str] = None,
        region: Optional[str] = None,
        runtime: Optional[bool] = True,
    ):

    if region is None:
        target_region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION"))
    else:
        target_region = region

    print(f"Create new client\n  Using region: {target_region}")
    session_kwargs = {"region_name": target_region}
    client_kwargs = {**session_kwargs}

    profile_name = os.environ.get("AWS_PROFILE")
    if profile_name:
        print(f"  Using profile: {profile_name}")
        session_kwargs["profile_name"] = profile_name

    retry_config = Config(
        region_name=target_region,
        retries={
            "max_attempts": 10,
            "mode": "standard",
        },
    )
    session = boto3.Session(**session_kwargs)

    if assumed_role:
        print(f"  Using role: {assumed_role}", end='')
        sts = session.client("sts")
        response = sts.assume_role(
            RoleArn=str(assumed_role),
            RoleSessionName="langchain-llm-1"
        )
        print(" ... successful!")
        client_kwargs["aws_access_key_id"] = response["Credentials"]["AccessKeyId"]
        client_kwargs["aws_secret_access_key"] = response["Credentials"]["SecretAccessKey"]
        client_kwargs["aws_session_token"] = response["Credentials"]["SessionToken"]


    if runtime:
        service_name='bedrock-runtime'
    else:
        service_name='bedrock'

    client_kwargs["aws_access_key_id"] = os.environ.get("AWS_ACCESS_KEY_ID","")
    client_kwargs["aws_secret_access_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY","")

    bedrock_client = session.client(
        service_name=service_name,
        config=retry_config,
        **client_kwargs
    )

    print("boto3 Bedrock client successfully created!")
    print(bedrock_client._endpoint)
    return bedrock_client



    ## for aksk bedrock
def get_bedrock_aksk(secret_name='chatbot_bedrock', region_name = "us-west-2"):
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        # For a list of exceptions thrown, see
        # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = json.loads(get_secret_value_response['SecretString'])
    return secret['BEDROCK_ACCESS_KEY'],secret['BEDROCK_SECRET_KEY']

ACCESS_KEY, SECRET_KEY=get_bedrock_aksk()


#role based initial client#######
os.environ["AWS_DEFAULT_REGION"] = "us-west-2"  # E.g. "us-east-1"
os.environ["AWS_PROFILE"] = "default"
#os.environ["BEDROCK_ASSUME_ROLE"] = "arn:aws:iam::687912291502:role/service-role/AmazonSageMaker-ExecutionRole-20211013T113123"  # E.g. "arn:aws:..."
os.environ["AWS_ACCESS_KEY_ID"]=ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"]=SECRET_KEY


#新boto3 sdk只能session方式初始化bedrock
boto3_bedrock = get_bedrock_client(
    #assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)

parameters_bedrock = {
    "max_tokens_to_sample": 2048,
    #"temperature": 0.5,
    "temperature": 0,
    #"top_k": 250,
    #"top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

bedrock_llm = Bedrock(model_id="anthropic.claude-v2", client=boto3_bedrock, model_kwargs=parameters_bedrock)
###test the bedrock langchain integration###
#bedrock_llm.predict("Human:how do you describe LLM?\n"+
#           "Assistant:")