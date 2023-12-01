# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import os
import traceback
from typing import Optional, List 
import json
import boto3
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from pydantic import BaseModel

class SMEndpointEmbeddings(BaseModel, Embeddings):
    endpoint_name: str
        
    def embed_documents(
        self, texts: List[str], chunk_size: int = 64
    ) -> List[List[float]]:
        results = []
        for t in texts:
            response = self.embed_query(t)
            results.append(response)
        return results

    def embed_query(self, text: str) -> List[float]:
        payload = {'text_inputs': [text]}
        payload = json.dumps(payload).encode('utf-8')
        client = boto3.client("runtime.sagemaker")
        response = client.invoke_endpoint(EndpointName=self.endpoint_name, 
                                                ContentType='application/json',  
                                                Body=payload)
    
        model_predictions = json.loads(response['Body'].read())
        embedding = model_predictions['embedding'][0]
        return embedding

# Inputs: document id and s3 location of summary
def main():

    print("Task starting")
    endpoint_name = os.environ['endpoint']
    print(f"Endpoint: {endpoint_name}")
    table_name = os.environ['table']
    print(f"Table: {table_name}")
    region = os.environ['region']
    print(f"Region: {region}")
    docId = os.environ['docId']
    print(f"docId: {docId}")
    jobId = os.environ['jobId']
    print(f"jobId: {jobId}")
    bucket = os.environ['bucket']
    print(f"bucket: {bucket}")
    name = os.environ['name']
    print(f"name: {name}")
    mntpnt = os.environ['mountpoint']
    print(f"name: {mntpnt}")

    try:
        s3 = boto3.client('s3')
        doc_dir = os.path.join(mntpnt, docId)
        if not os.path.exists(doc_dir):
            os.mkdir(doc_dir)
        sum_dir = os.path.join(doc_dir, 'summary')
        if not os.path.exists(sum_dir):
            os.mkdir(sum_dir)
        sum_path = os.path.join(sum_dir, 'summary.txt')
        print(f"Downloading s3://{bucket}/{name} to {sum_path}")
        s3.download_file(bucket, name, sum_path)

        persist_directory = os.path.join(doc_dir, 'db')
        if not os.path.exists(persist_directory):
            os.mkdir(persist_directory)
        loader = TextLoader(sum_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,
                                                        chunk_overlap  = 0)
        texts = text_splitter.split_documents(documents)
        print(f"Number of splits: {len(texts)}")

        embeddings = SMEndpointEmbeddings(
            endpoint_name=endpoint_name,
        )
        vectordb = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
        vectordb.persist()

        ddb = boto3.resource('dynamodb', region_name=region)
        table = ddb.Table(table_name)
        table.update_item(
                Key = { "documentId": docId, "jobId": jobId },
                UpdateExpression = 'SET jobStatus = :jobstatusValue', 
                ExpressionAttributeValues = {
                    ':jobstatusValue': "Complete",
                }
            )

    except Exception as e:
        trc = traceback.format_exc()
        print(trc)
        print(str(e))

if __name__ == "__main__":
    main()
