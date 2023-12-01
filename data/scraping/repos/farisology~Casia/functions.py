import re
import os
import json
import boto3
import cohere
import base64
import openai
import psycopg2
import pinecone
from time import perf_counter
from time import time, sleep
from datetime import datetime
from sqlalchemy import create_engine, text
from botocore.exceptions import ClientError
from .models import Chats, Posts
from sqlmodel import SQLModel, Session, select

vdb = pinecone.Index("testbed")


def get_secret(sname):

    secret_name = sname
    region_name = "ap-southeast-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
    else:
        # Decrypts secret using the associated KMS CMK.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            return secret
        else:
            decoded_binary_secret = base64.b64decode(
                get_secret_value_response['SecretBinary'])
            return decoded_binary_secret


casiadb_user = openai_key = json.loads(
    get_secret("casiadb_creds"))["username"]
casiadb_pass = openai_key = json.loads(
    get_secret("casiadb_creds"))["password"]

engine = create_engine(
    f'postgresql://{casiadb_user}:{casiadb_pass}@casiadb.cdmj2hx4hrmn.ap-southeast-1.rds.amazonaws.com/casia')

SQLModel.metadata.create_all(engine)


def write_chats(created_at_utc,
                model,
                question_length,
                question_text,
                answer_text,
                processing_time_seconds,
                endpoint_name):
    message = Chats(created_at_utc=created_at_utc,
                    model=model,
                    question_length=question_length,
                    question_text=question_text,
                    answer_text=answer_text,
                    processing_time_seconds=processing_time_seconds,
                    endpoint_name=endpoint_name)
    with Session(engine) as session:
        session.add(message)
        session.commit()


def get_articles():
    try:
        connection = psycopg2.connect(
            f'postgresql://{casiadb_user}:{casiadb_pass}@casiadb.cdmj2hx4hrmn.ap-southeast-1.rds.amazonaws.com/casia')
        cursor = connection.cursor()
        postgreSQL_select_Query = "select * from posts"
        cursor.execute(postgreSQL_select_Query)
        results = cursor.fetchall()
        post_obj = list()
        if len(results) > 1:
            for post in results:
                post_dict = dict()
                post_dict["id"] = post[0]
                post_dict["created_at"] = post[1]
                post_dict["title"] = post[2]
                post_dict["short_description"] = post[3]
                post_dict["description"] = post[4]
                post_dict["image_s3_uri"] = post[5]
                print(post_dict["title"])
                post_obj.append(post_dict)

        return post_obj
    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data from PostgreSQL", error)

    finally:
        # closing database connection.
        if connection:
            cursor.close()
            connection.close()
            print("PostgreSQL connection is closed")


def chadgpt_completion(question, context, model):
    max_retry = 5
    retry = 0
    preample = """You are casia a wise botanist with vast knowledge in gardening and great servitude to human well-being. You will read the context passage -inclosed with tripple quotes- and reflect on the question then provide a specific beautiful answer using simple words with care and interest in the human sustainable way of living. Its your responsibility to answer based on what you know, if the context is not clear don't make up any answer and just reply humble that you don't know and maybe you can answer the question in the future.

Passage:"""
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"{preample}"},
                    {"role": "assistant", "content": f"{context}"},
                    {"role": "user", "content": f"{question}"},
                ],
                temperature=0,
                top_p=1.0, max_tokens=300, frequency_penalty=0.0, presence_penalty=0.0,
            )
            text = response["choices"][0]["message"]["content"].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            filename = '%s_gpt3.txt' % time()
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
            # save_file('gpt3_logs/%s' % filename, preample +
            #           '\n\n==========\n\n' + text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3.5 turbo error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def gpt3_embedding(content, engine="text-embedding-ada-002"):
    # fix any UNICODE errors
    content = content.encode(encoding='ASCII', errors='ignore').decode()
    response = openai.Embedding.create(input=content, engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def cohere_interface(question, model, cohere_key, pinecone_key, openai_key):
    co = cohere.Client(cohere_key)
    openai.api_key = openai_key
    pinecone.init(api_key=pinecone_key, environment='us-east1-gcp')
    vdb = pinecone.Index("testbed")
    vector = gpt3_embedding(question)

    tic = perf_counter()
    results = vdb.query(vector=vector, top_k=1, include_values=False,
                        include_metadata=True, namespace="casia_kb")
    prompet = f"{question} based on the this information below:\n{results}"
    response = co.generate(
        model=model,
        prompt=prompet,
        max_tokens=500,
        temperature=0.2,
        k=0,
        stop_sequences=[],
        return_likelihoods='NONE')
    toc = perf_counter()
    ql = len(question.split())
    write_chats(datetime.now(),
                model,
                ql,
                question,
                response.generations[0].text,
                toc - tic,
                "ask_cohere")
    return response.generations[0].text


def openai_interface(question, model, openai_key, pinecone_key):
    openai.api_key = openai_key
    pinecone.init(api_key=pinecone_key, environment='us-east1-gcp')
    vdb = pinecone.Index("testbed")
    vector = gpt3_embedding(question)

    tic = perf_counter()
    results = vdb.query(vector=vector, top_k=3, include_values=False,
                        include_metadata=True, namespace="casia_kb")
    answer = chadgpt_completion(question, '\n'.join(
        [info["metadata"]["text"] for info in results["matches"]]), model)
    toc = perf_counter()
    ql = len(question.split())
    write_chats(datetime.now(),
                model,
                ql,
                question,
                answer,
                toc - tic,
                "ask_openai")

    return answer
