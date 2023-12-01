import boto3
import openai
import json
from dotenv import load_dotenv
from os import getenv

load_dotenv(verbose=True)           # Set operating system environment variables based on contents of .env file.

openai.api_key = getenv('OPEN_AI_KEY')

prompt = """Please create a new EC2 instance for me.
I want to use it as a Minecraft Bedrock server for up to 25 concurrent players. Please balance performance and cost.
I'd also like it to have a 1/4 TB of storage.
After its created, I want to be able to easily get rid of this instance using the AWS console."""

print(f"prompt={prompt}")

run_instances_function = {
    "name": "run_instances",
    "description": "Use the AWS API to create a new EC2 instance.",
    "parameters": {
        "type": "object",
        "properties": {
            "InstanceType": {
                "type": "string",
                "description": "The EC2 instance type. There is information about the options here, https://aws.amazon.com/ec2/instance-types/t3/",
                "enum": ["t3.nano", "t3.micro", "t3.small", "t3.medium", "t3.large", "t3.xlarge", "t3.2xlarge"]},
            "VolumeSize": {"type": "number",
                           "description": "The size of the volume, in GiBs."},
            "DisableApiTermination": {"type": "boolean",
                                      "description": "If you set this parameter to true, you can’t terminate the instance using the Amazon EC2 console, CLI, or API; otherwise, you can."}
                      },
        "required": ["InstanceType", "VolumeSize", "DisableApiTermination"]
                  }
              }

completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    temperature=float(getenv('TEMPERATURE')),
    messages=[{"role": "user", "content": prompt}],
    functions=[run_instances_function],
    function_call={"name": "run_instances"})

# print(completion)

reply_content = completion.choices[0].message

function_name = reply_content.to_dict()['function_call']['name']
function_json = reply_content.to_dict()['function_call']['arguments']
function_dict = json.loads(function_json)

print(f"\nGPT responded with, function_name={function_name}, function_dict={function_dict}")

if getenv('SKIP_AWS') == 'False':

    ec2_client = boto3.client('ec2', region_name=getenv('AWS_REGION_NAME'))

    response = ec2_client.run_instances(
        ImageId='ami-00b1c9efd33fda707',        # Amazon Linux 2 AMI ID.
        InstanceType=function_dict['InstanceType'],
        MinCount=1,
        MaxCount=1,
        BlockDeviceMappings=[
            {
                'DeviceName': '/dev/xvda',
                'Ebs': {
                    'VolumeSize': function_dict['VolumeSize'],
                    'VolumeType': 'gp2'         # General Purpose SSD (gp2) volume.
                }
            }
        ],
        DisableApiTermination=function_dict['DisableApiTermination']
    )

    instance_id = response['Instances'][0]['InstanceId']
    print(f"\nEC2 instance created, instance_id={instance_id}")
