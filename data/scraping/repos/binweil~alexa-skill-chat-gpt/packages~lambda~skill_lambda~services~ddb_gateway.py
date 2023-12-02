from datetime import datetime
from datetime import timedelta
import boto3
import os

from botocore.exceptions import ClientError

from constants.openai_constants import OpenAIConfig


class DynamoDBGateway:
    def __init__(self):
        self.ddb_client = boto3.client('dynamodb')
        self.table_name = os.getenv('DYNAMODB_TABLE_NAME')

    def put(self, customer_id, interaction_count=0, model=OpenAIConfig.GPT_MODEL_3_5.value):
        updated_ttl = (datetime.now() + timedelta(days=30)).strftime('%s')
        self.ddb_client.put_item(
            TableName=self.table_name,
            Item={
                "customer_id": {
                    'S': customer_id
                },
                "interaction_count": {
                    'N': str(interaction_count)
                },
                "model_setting": {
                    'S': model
                },
                "expire_ttl": {
                    'N': updated_ttl
                }
            }
        )

    def get(self, customer_id):
        try:
            response = self.ddb_client.get_item(
                TableName=self.table_name,
                Key={
                    'customer_id': {
                        'S': customer_id
                    }
                }
            )
            if "Item" in response:
                return {
                    "customer_id": response["Item"]["customer_id"]["S"],
                    "interaction_count": int(response["Item"]["interaction_count"]["N"]),
                    "model_setting": response["Item"]["model_setting"]["S"]
                }
            return None
        except ClientError as e:
            print(e.response["Error"]["Message"])
            return None

    def update_interaction_count(self, customer_id, interaction_count):
        updated_ttl = (datetime.now() + timedelta(days=30)).strftime('%s')
        expression = "set interaction_count=:r, expire_ttl=:p"
        response = self.ddb_client.update_item(
            TableName=self.table_name,
            Key={
                'customer_id': {
                    'S': customer_id
                },
            },
            UpdateExpression=expression,
            ExpressionAttributeValues={
                ':r': {
                    "N": str(interaction_count)
                },
                ":p": {
                    'N': updated_ttl
                }
            },
            ReturnValues="UPDATED_NEW")
        print(response)

    def update_model_setting(self, customer_id, model_setting):
        updated_ttl = (datetime.now() + timedelta(days=30)).strftime('%s')
        expression = "set model_setting=:r, expire_ttl=:p"
        response = self.ddb_client.update_item(
            TableName=self.table_name,
            Key={
                'customer_id': {
                    'S': customer_id
                }
            },
            UpdateExpression=expression,
            ExpressionAttributeValues={
                ':r': {
                    "S": model_setting
                },
                ":p": {
                    'N': updated_ttl
                }
            },
            ReturnValues="UPDATED_NEW")
        print(response)

