import json
import boto3
from langchain.llms.bedrock import Bedrock


def create_bedrock_client():
    
    sts_connection = boto3.client('sts')
    acct_b = sts_connection.assume_role(
        RoleArn="arn:aws:iam::<accountid>:role/LambdaBedrockCrossAccountRole",
        RoleSessionName="cross_acct_lambda"
    )

    ACCESS_KEY = acct_b['Credentials']['AccessKeyId']
    SECRET_KEY = acct_b['Credentials']['SecretAccessKey']
    SESSION_TOKEN = acct_b['Credentials']['SessionToken']

    bedrock_config = None

    # create service client using the assumed role credentials
    bedrock_client = boto3.client(
        service_name="bedrock",
        region_name='us-east-1',
        endpoint_url='https://bedrock.us-east-1.amazonaws.com',
        config=bedrock_config,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        aws_session_token=SESSION_TOKEN,
    )
    return bedrock_client

def decode_and_show(model_response: GenerationResponse) -> None:
    """
    Decodes and displays an image from SDXL output

    Args:
        model_response (GenerationResponse): The response object from the deployed SDXL model.

    Returns:
        None
    """
    image = model_response.artifacts[0].base64
    image_data = base64.b64decode(image.encode())
    image = Image.open(io.BytesIO(image_data))
    display(image)
