
import boto3
from dotenv import load_dotenv
from langchain.tools import BaseTool

load_dotenv()

ec2_client = boto3.client('ec2')

desc = """use this tool when the user asks for information about EC2 AWS resources 
          This tool will act like a cloud architect with for AWS cloud resources.
          It will return the json formatted list of EC2 AWS resources.
"""

class awsEC2Tool(BaseTool):
    name = "AWS EC2 Tool"
    description = desc
    def _run(self):
        ec2_client = boto3.client('ec2')
        aws_response = ec2_client.describe_instances(
            # Filters = [
            #     {
            #         'Name': 'instance-state-name',
            #         'Values': ['running']
            #     }
            # ]
        )
        instances = []
        for reservation in aws_response['Reservations']:
            for instance in reservation['Instances']:
                instance_name = instance['Tags'][0]['Value']
                instance_id = instance['InstanceId']
                instance_type = instance['InstanceType']
                launch_time = instance['LaunchTime'].strftime("%Y-%m-%d %H:%M:%S")
                instances_key_pair_name = instance['KeyName']
                instance_monitoring_state = instance['Monitoring']
                instance_placement = instance['Placement']
                try:
                    private_ip_address = instance['PrivateIpAddress']
                except KeyError:
                    private_ip_address = None
                try:
                    public_ip_address = instance['PublicIpAddress']
                except KeyError:
                    public_ip_address = None
                public_dns_name = instance['PublicDnsName']
                instance_current_state = instance['State']['Name']
                instances.append(
                    {
                        'instance_name' : instance_name,
                        'instance_id': instance_id,
                        'instance_type': instance_type,
                        'launch_time' : launch_time,
                        'instance_key_pair_name' : instances_key_pair_name,
                        'instance_monitoring_state' : instance_monitoring_state,
                        'instance_placement' : instance_placement,
                        'private_ip_address' : private_ip_address,
                        'public_ip_address' : public_ip_address,
                        'public_dns_name' : public_dns_name,
                        'instance_current_state' : instance_current_state
                        
                    }
                )
        if len(instances) > 0:
            return instances
        else:
            return "There are no instances available on your AWS account."
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")
