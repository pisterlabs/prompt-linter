# GIL
# Generator vs iterator
# Higher order fun in python
# args vs kwargs
# dictionary name change
# copy vs deepcopy
# decorator and print arguments
# reverse of a list
# local vs global
# tuple unpacking
# modules in python
# .pyc files in python
# 

#Application ID AA00BGK461




# import requests
# # accountApiPath = f'/api/report/account?account_id={accountId}'
# novaApiPath = "https://8yzrfjei4l.execute-api.us-gov-west-1.amazonaws.com/scott/api/report/account?account_id=653363285927"
# evnoapikey='7fc86567-c1ab-4e88-8af8-d7b70c6283b1'
# novaApiKey="YqRV2Rrz1L7NWSX4njhdotnbLeoRkCH663oFecF4"
# headers = {"Content-Type": "application/json",
#            "Accept": "application/json",
#            "x-api-key": novaApiKey,
#            "evno-x-api-key": evnoapikey}
# response = requests.get(novaApiPath, headers=headers, verify=False)

# print(response)

# if response.status_code == 200:
#         novaData = response.json()
#         print(f'Processed Nova data: {novaData}')

# nova_url = "https://8yzrfjei4l.execute-api.us-gov-west-1.amazonaws.com/scott/api/report/account?account_id=653363285927"
# evnoapikey='7fc86567-c1ab-4e88-8af8-d7b70c6283b1'
# nova_api_key="YqRV2Rrz1L7NWSX4njhdotnbLeoRkCH663oFecF4"
# headers = {
# "Content-Type": "application/json",
# "Accept": "application/json",
# "x-api-key": nova_api_key,
# "evno-x-api-key": evnoapikey}

# response = requests.get(nova_url, headers=headers, verify=True)
# print("load metadata response ",response.json())


# from flask import Flask

# app = Flask(__name__)

# @app.route('/')
# def hello():
#         return "Hello world Selvam"

# orderdDict
   

# def main() :
#     A = "zzzX"
#     # "abc";//"zzzX";//"dBacaAA";
#     B = "zzzX"
#     # "ABC";//zzzX";//"caBdaaA";
#     count = 0
#     strLength = len(A)
#     letters =  dict()
#     noOfCharacterToBeChecked = 1
#     startIndex = 0
#     while (noOfCharacterToBeChecked <= strLength) :
#         lastIndex = startIndex + noOfCharacterToBeChecked
#         # First get the characters in HashMap for the noOfLtters to be checked
#         i = startIndex
#         while (i < lastIndex) :
#             letters[A[i]] = letters.get(A[i],0) + 1
#             i += 1
#         # compare the second string and increase count if matched
#         i = startIndex
#         while (i < lastIndex) :
#             if ((B[i] in letters.keys())) :
#                 letterCount = letters.get(B[i],0)
#                 if (letterCount > 1) :
#                     letters[B[i]] = letters.get(B[i],0) - 1
#                 else :
#                     del letters[B[i]]
#             else :
#                 break
#             i += 1
#         if letters :
#             count += 1
#         letters.clear()
#         # Check till last index
#         if (lastIndex < strLength) :
#             startIndex += 1
#         else :
#             # Once last index is reached increase noOfLetter to be checked and reset startIndex
#             noOfCharacterToBeChecked += 1
#             startIndex = 0
#     print(count, end ="")
   
# main()


# def solution(A: str, B: str) -> int:
#     def fragmentsCorrespond(A: str, B: str, posA: int, posB: int) -> bool:
#         # check if starting positions are the same
#         if (posA != posB):
#           return False

#         # create maps to count number of occurrences of each letter in fragments
#         mapA = {}
#         mapB = {}

#         # count occurrences of each letter in fragments of A and B
#         for i in range(posA, posA + fragmentLength):
#           chA = A[i]
#           chB = B[i]

#           mapA[chA] = mapA.get(chA, 0) + 1
#           mapB[chB] = mapB.get(chB, 0) + 1

#         # compare the maps
#         return mapA == mapB

#   # initialize count of corresponding fragments to 0
#     count = 0

#   # iterate over each fragment of A and B and check if they correspond
#   for i in range(len(A)):
#     for j in range(len(B)):
#       if fragmentsCorrespond(A, B, i, j):
#         count += 1

#   # return count of corresponding fragments
#   return count


# solution("dBacaAA", "caBdaaA")


# import boto3
import json
accountID = "706330793438"
assumeRole = "dcs-logging-plugin-role"


def get_aws_client(resourceType, accountId, awsRegion, roleName, sessionName, rolePath='/'):
    """
    This function Assumes role and returns a client

    Args:
        resourceType (string): Resource type to initilize (Ex: ec2, s3)
        accountId (string): Target account Id to assume role
        awsRegion (string): AWS region to initilize service
        roleName (string): Role name to assume
        sessionName (string): Assume role session name
        rolePath (string): Role Path, default = '/'

    Returns:
        serviceClient (botocore client): botocore resource client
    """
#     stsClient = boto3.client('sts')
#     try:
#         # Make Role ARN
#         if rolePath == '/':
#             roleArn = f'arn:aws-us-gov:iam::{accountId}:role/{roleName}'
#         else:
#             roleArn = f'arn:aws-us-gov:iam::{accountId}:role/{rolePath.lstrip("/").rstrip("/")}/{roleName}'

#         # Assume role
#         role = stsClient.assume_role(RoleArn=roleArn, RoleSessionName=sessionName)
#         accessKey = role['Credentials']['AccessKeyId']
#         secretKey = role['Credentials']['SecretAccessKey']
#         sessionToken = role['Credentials']['SessionToken']
#         serviceClient = boto3.client(resourceType, region_name=awsRegion,
#                                      aws_access_key_id=accessKey,
#                                      aws_secret_access_key=secretKey,
#                                      aws_session_token=sessionToken)
#         return serviceClient
#     except Exception as error:
#         print(f'Failed to assume the role for Account: {accountId}: {error}')
#         raise



# def get_orgId(accountID, assumeRole):
#     session = "helloworld"
#     # Retrieve Organization ID
#     orgClient = get_aws_client(
#         "organizations", accountID, "us-gov-west-1", assumeRole, session, "DCSLayer0"
#     )

#     print(orgClient.describe_organization()["Organization"]["Id"])

# get_orgId(accountID,assumeRole)

# Expertise + Requirement




# z
# dcs-aws-platform
# Service Hub vs Global ops

# Service hubs (3)

# AME VPC automation
# APAC
# EMA

# step functions deployed in us-east-1
# Inventory dynamo db table (vpc-automation)


# Digital gold Pros:

# 1. No security
# 2. No Making Charges
# 3. Physical Delivery
# 4. No minimum requirement

# MMTC
# SafeGold
# Augmont


# Cons
# 1. 3% GST 
# 2. Spread of 3 - 6% on selling 

# Challenges:
# ============
# No indefinite holding

# Capital gain tax

# These are not regulated by SEBI or RBI 



# SGB:

# NO GST
# 0 Charges 
# Rs 50 discount per gram
# Tax-free
# 2.5% interest p.a

# cons:
#  8 years lock-in 







# prd npd sbx sys  

# Dev --> 

# import qrcode
# upi://pay?pa=example@upi&pn=John+Doe&am=100&tn=Invoice+1234&cu=INR&tr=1234567890&mc=0123&aid=bank.example.upi

# ?pa=johndoe@upi&pn=John+Doe&am=100&tn=Test+Payment&cu=INR&tr=1234567890&url=https%3A%2F%2Fexample.com%2Fupi%2Fcallback&mc=5812&orgid=12345



# qr_img = qrcode.make("upi://pay?cu=INR&pa=SPHOO1414.09@CMSIDFC&pn=Sphoorthi Kutumbam&url=https%3A%2F%2F5d25-157-48-173-165.ngrok.io%2Fcallback")  
# # saving the image file  
# qr_img.save("callbackqr-img.jpg")

# from flask import Flask,render_template,request,send_from_directory

# app = Flask(__name__)

# @app.route('/',methods=["GET","POST"])
# def index():
#     return render_template("report_form.html")

# @app.route("/qr",methods=['GET',"POST"])
# def success():
#     return render_template("Thanks.html")

# if __name__ == "__main__":

#     app.run(host='0.0.0.0', port=81,debug=True)

# AWS-CloudOps&Platform.
# Azure-CloudOps&Platform.
# GCP-CloudOps&Platform
# Networking-CloudOps&Platform
# VDI-CloudOps&Platform.


# DCSSSMAccessRole  L0 account role
# EC2SSMAgentMonitoring L1 account role

# prd url

# L1 npd account its ops npd --> 920112648628
# Jenkins DSL (Domain Specific Language)  --> Job DSL pluggin
# Groovy Script
# shared libraries (to create multiple tools pipelines)

# import boto3

# client = boto3.client('ec2',region_name='us-east-1')
# response = client.describe_instances(
#     Filters=[
#         {
#             'Name': 'iam-instance-profile.arn',
#             'Values': [
#                 'arn:aws:iam::1234567890:instance-profile/MyProfile-ASDNSDLKJ',
#             ]
#         },
#     ]
# )

# 438304065432
# 848809407043

# Create job
# cron job * * * * * 
# create user with role based permissions
# fetch source code from Github and building it




# Stackset DenyITSLayer1PlatformModification

# account list  564 + 1169


# Nova query /api/report/

# account status active
# if account layer0 deployed as true

# account in account_list:

#     switch to that account and bring ec2 instance list having instance profile role in 21 regions


# else:
#     ignore  





# 848809407043 --> ITSGovernanceRole 
# 239182046413
# 129989725660 --> No ITsGovernanceRole
# 438304065432 --> No Switch
# 1. Console way  

# 2. Cloudformation way (Reach out to CSPOG team)

# 3. Using Boto3 

# 4. Create with bucket new policy + old policy

# 5. Test with Exception rule (Account, MSP, resource)


# Email cross check with secure as true

# af-south-1




# print(set(x))



# 626107101308 --> ResourceLogicalId:DeloitteVPCPeeringExceptionBoundaryPolicy, ResourceType:AWS::IAM::ManagedPolicy, 
# ResourceStatusReason:Policy arn:aws:iam::626107101308:policy/DeloitteVPCPeeringExceptionBoundaryPolicy
# does not exist or is not attachable. (Service: AmazonIdentityManagement; Status Code: 404; Error Code: NoSuchEntity; Request ID: 9670926e-479f-45e2-a626-0ad350e21b37; Proxy: null)

# 637918661773 --> stack deosn't exist



# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
# import openai
# openai.api_key = "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"
# audio_file= open("bhayya.mp3", "rb")
# transcript = openai.Audio.translate("whisper-1", audio_file)
# print(transcript)


# import requests
# import json

# # Replace with your own values
# ACCESS_TOKEN = 'EAARDEWiFyZAsBAPqsUVaEUgXT7EZCrvSCiKZB4HPjucDUwuK2ZA6h96TE9YsR5tf23V9ZCPo42w3lijs7x5KGNDJNHQWYj0EZAyXy2wieMBee27woueBZBknBDcengOT9ZAiOfN3SvaLbCoc7ZAXZCIHn8yb7LxwyNvdtxZBXCRnbux2hLCvbKXcF8o8yvt0DUjLlZASMZC4VvvidLgZDZD'
# SENDER_PHONE_NUMBER_ID = '110974595107295'
# WABA_ID = '103025609236513'
# VERIFY_TOKEN = 'SK2022Telangana'

# # Endpoint for sending messages
# ENDPOINT = f'https://api.chat-api.com/instance{WABA_ID}/message'

# # Message template for poll
# message = {
#     "chatId": "customer_phone_number",
#     "message": {
#         "type": "multi",
#         "body": [
#             {
#                 "type": "poll",
#                 "question": "Are you Practiced Prasthana Sadhana today?",
#                 "options": ["Yes", "No"]
#             },
#             {
#                 "type": "poll",
#                 "question": "If you are Practised along with family, how many members?",
#                 "options": ["2", "3", "4","5","6"]
#             },
#             {
#                 "type": "text",
#                 "text": "Thanks for your response. your feedback is hightly appreciated?"
#             }
#         ]
#     }
# }

# # Convert message to JSON
# json_message = json.dumps(message)

# # Send message
# response = requests.post(
#     url=ENDPOINT,
#     headers={
#         'Authorization': f'Bearer {ACCESS_TOKEN}',
#         'Content-Type': 'application/json'
#     },
#     data=json_message
# )

# # Print response
# print(response.text)

"""
# Fundamental data types (Immutable data types)

1. Int
2. Float
3. Boolean
4. String
5. Complex


# Collection/derived data tyeps
1. List
2. Tuple  -> immutable data type
3. Set
4. Dictionary


# Mutable vs Immutable

# In python everything is an object

"""

"""
1. Fundamental data types  (immutable data types)
   1. Integer
   2. float
   3. string
   4. boolean
   5. complex  --> 10+1j

2. Derived or collection data types
    1. list  //array
    2. set   //set
    3. tuple  (immutable)
    4. dictionary //map
"""
# a = 10
# a = 10.0
# a = '10'
# a = "10"
# a = '''10'''
# a = """10"""
# a = True
# a = False 
# a = None
# a = 10+1j


# print(a)
# print(type(a))



# a = []
# print(type(a))
# a = list()
# print(type(a))

# a = {}
# print(type(a))
# a = dict()
# print(type(a))


# a = ()

# print(type(a))

# a = tuple()
# print(type(a))

# a = set()
# print(type(a))



# a = {1,2,3,4,5}

# print(type(a))

# a = {1:1,2:2,3:3}
# print(type(a))

"""
5 types

1. static function 
2. Accept one arguments/parmaters
3. with default
4. without default
5. infinite arguments  using *


"""

"""
In python, everything is an object
every .py file is called module

python.org 

os 
json 
math
random 

1. Inbuilt modules
2. External/outside/third-party --> pypi store and search for modue and install it using pip xyz, flask, django, requests, 
3. custom- module
"""

# vpc_descriptions = ec2.describe_vpcs()
 
# for vpc in vpc_descriptions["Vpcs"]:
#     pass


# response = vpc_client.describe_network_interfaces(
#         Filters=[
#             {
#                 'Name': 'vpc-id',
#                 'Values': [
#                     "vpc-0a286f22859510125"
#                     # "vpc-e6ec509c"
#                 ]
#             }
#         ]
#     )


# print(response)



def create_stack_instances():
    stackset_name = 'MY_STACKSET_NAME'
    account_list = ['123456789012', '987654321098']

    # Define the operation preferences
    operation_preferences = {
        'FailureToleranceCount': 0,
        'MaxConcurrentCount': 3
    }

    # Create the stack instances
    response = cloudformation.create_stack_instances(
        StackSetName=stackset_name,
        Accounts=account_list,
        OperationPreferences=operation_preferences
    )

def remove_stack_instances():
    stackset_name = 'MY_STACKSET_NAME'
    account_list = ['123456789012', '987654321098']
    region_list = ['us-east-1', 'us-west-2']

    # Specify whether to retain the StackSet instances or not
    retain_stack = False

    # Remove the stack instances
    response = cloudformation.delete_stack_instances(
        StackSetName=stackset_name,
        Accounts=account_list,
        Regions=region_list,
        RetainStacks=retain_stack
    )


def create_new_stackset():
    stackset_name = 'MY_STACKSET_NAME'
    template_body = """
        {
            "Resources": {
                "MyEC2Instance": {
                    "Type": "AWS::EC2::Instance",
                    "Properties": {
                        "ImageId": "ami-0c55b159cbfafe1f0",
                        "InstanceType": "t2.micro"
                    }
                }
            }
        }
    """

    # Define the stackset parameters
    stackset_params = {
        'StackSetName': stackset_name,
        'TemplateBody': template_body,
        'Capabilities': ['CAPABILITY_IAM', 'CAPABILITY_NAMED_IAM'],
        'AdministrationRoleARN': 'arn:aws:iam::123456789012:role/CloudFormationStackSetAdministrationRole',
        'ExecutionRoleName': 'MyStackSetExecutionRole',
        'Description': 'My StackSet Description'
    }

    # Create the StackSet
    response = cloudformation.create_stack_set(**stackset_params)

    print(response)



new = ["087127721683",
"162310846309",
"162310846309",
"173479391558",
"185075825010",
"234019974018",
"309245159336",
"312249472578",
"440708338928",
"447099417955",
"320510073098",
"363235117348",
"373758253227",
"380906577769",
"386880889362",
"388634006099",
"413145654121",
"415134345932",
"168849861669",
"174295623751",
"184405994032",
"208421051549",
"226603863404",
"227841379693",
"228115329917",
"246585516170",
"247048948584",
"256706112543",
"272606000076",
"279743555714",
"287248264930",
"003866754167",
"009296111214",
"012680939556",
"054273887971",
"055840983865",
"066979683739",
"089587160124",
"107282117866",
"109997849406",
"117523786847",
"150316879294",
"159602377297",
"159861609278",
"162022282563",
"167393382127",
"492634300165",
"497315343129",
"506633848063",
"527490275647",
"544734843935",
"546138792195",
"583303984486",
"584630848692",
"586482989402",
"614092787540",
"616194183804",
"616864236654",
"625001204664",
"625125552135",
"626107101308",
"641938356393",
"643596949585",
"644141345393",
"644784173767",
"650018817833",
"652993727289",
"670161819556",
"716751674484",
"720257876889",
"724311065114",
"736530749364",
"753880925017",
"766245584008",
"785786920527",
"816919933351",
"837626969847",
"839326706163",
"858141963590",
"858526972147",
"858526972147",
"883784239155",
"910380970013",
"912681226107",
"931334696467",
"948310893572",
"948948595631",
"952135726105",
"955116103754",
"960956162241",
"975204971987",
"994920118214",
"997667771749",
"488502120284",
"488502120284",
"543492705153",
"553697376517",
"607851404322",
"607851404322",
"623899618944",
"683133891344",
"683133891344",
"844577878525",
"844577878525",
"854481054617",
"905676609443",
"918701091807",
"933831760677",
"933831760677",
"967972633959",
"997715449024"]





# 240044539051
# 780113007310
# 814255944148
# 309727501936
# 735624102611
# 354173411065
# 354173411065
# 707763339947
# 735624102611
# 778558837889
# 812790839692
# 819300110664


# {"162435009311":"AccountAdminRole",
# "163564840054":"CloudscriptValidatePrdCrossAccountRole",
# "201944482039":"EventRuleRole",
# "282181226940":"AccountAdminRole",
# "498306521888":"CreateSAMLProviderRole",
# "511898103173":"AccountSysAdminRole",
# "552923642514":"TenableIORole",
# "553697376517":"CloudscriptValidatePrdCrossAccountRole",
# "933831760677":"TenableIORole, "
# }


# x = {
#    "087549363555":"ResourceLogicalId:DenyModificationPolicyv2, ResourceType:AWS::IAM::ManagedPolicy, ResourceStatusReason:DenyITSLayer1PlatformModification already exists.",
#    "122578856689":"ResourceLogicalId:AccountDBAdminRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:AWS_122578856689_DBAdmin already exists in stack arn:aws:cloudformation:us-east-1:122578856689:stack/StackSet-DCSLayer0-AccountRoles-ITS-59ca39ac-f434-4e6e-be74-2201f0dfd94f/a44d62a0-2a3d-11ed-9d85-12027c3bee8d.",
#    "233127284560":"ResourceLogicalId:BackupRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:AWS_DCS_BackupAdmin already exists in stack arn:aws:cloudformation:us-east-1:233127284560:stack/StackSet-DCSLayer0-PersistentRoles-us-csd-consulting-1cfd9952-7078-482e-ade6-b2a5682b822f/2fa17210-09ab-11ec-80e5-0af5a9785083.",
#    "239823972708":"ResourceLogicalId:CloudscriptValidatePrdCrossAccountRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:cloudscript-validate already exists in stack arn:aws:cloudformation:us-east-1:239823972708:stack/CloudScriptValidation/236c3fe0-6b3c-11ed-a9a5-1200a9993c09.",
#    "240044539051":"ResourceLogicalId:CreateSAMLProviderRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:Create_SAML_Provider already exists in stack arn:aws:cloudformation:us-east-1:240044539051:stack/StackSet-DCSLayer0-AccountRoles-ITS-07e3c005-4e19-4418-9f1c-6e0d3db1d72a/e664f840-d6df-11eb-9f5b-0efa5da0ad91.",
#    "264944699463":"ResourceLogicalId:AccountSysAdminRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:AWS_264944699463_SysAdmin already exists in stack arn:aws:cloudformation:us-east-1:264944699463:stack/StackSet-DCSLayer0-AccountRoles-ITS-b9582f1b-cb9b-40d9-9e4a-23a054cfe321/f68b56c0-d61b-11eb-a42a-0ac763f17b41.",
#    "314448338500":"ResourceLogicalId:BackupRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:AWS_DCS_BackupAdmin already exists in stack arn:aws:cloudformation:us-east-1:314448338500:stack/StackSet-DCSLayer0-PersistentRoles-us-csd-consulting-2a9acd05-7b5b-4f6b-a2f4-c21b81722d26/0696a070-c1a9-11ea-997a-12ae8fb95b53.",
#    "342472205137":"ResourceLogicalId:AccountSysAdminRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:AWS_342472205137_SysAdmin already exists in stack arn:aws:cloudformation:us-east-1:342472205137:stack/StackSet-DCSLayer0-AccountRoles-ITS-b4eb8251-d174-4f2f-993f-ca0dd33246ff/9733ad20-0051-11ec-a51a-12f0216d258f.",
#    "370020434536":"ResourceLogicalId:BackupRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:AWS_DCS_BackupAdmin already exists in stack arn:aws:cloudformation:us-east-1:370020434536:stack/StackSet-DCSLayer0-PersistentRoles-us-csd-consulting-bf645dd2-ede5-4eee-a077-7cc352e78e06/77a78d30-c1a7-11ea-8614-1271b8f35690.",
#    "420525251589":"ResourceLogicalId:CreateSAMLProviderRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:Create_SAML_Provider already exists in stack arn:aws:cloudformation:us-east-1:420525251589:stack/StackSet-DCSLayer0-AccountRoles-ITS-26c2f6e4-02a9-49c7-8aa5-a6a53aa118d7/4cc5a8e0-d6e1-11eb-904f-0aa443d7245f.",
#    "447718312100":"ResourceLogicalId:BackupRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:AWS_DCS_BackupAdmin already exists in stack arn:aws:cloudformation:us-east-1:447718312100:stack/StackSet-DCSLayer0-PersistentRoles-us-csd-consulting-f7260703-7ab3-4741-8859-b11fcd963b79/57bc8d70-c1a9-11ea-8cf0-0a55488d44ba.",
#    "457026597750":"ResourceLogicalId:AccountReaderRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:AWS_457026597750_Reader already exists in stack arn:aws:cloudformation:us-east-1:457026597750:stack/StackSet-DCSLayer0-AccountRoles-ITS-12310312-fd86-47bc-aba3-487ee163cb5b/4d954740-d618-11eb-9376-0e229acdb0d3.",
#    "486653307662":"ResourceLogicalId:CreateSAMLProviderRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:Create_SAML_Provider already exists in stack arn:aws:cloudformation:us-east-1:486653307662:stack/StackSet-DCSLayer0-AccountRoles-ITS-ed55b420-08b3-450a-bb23-77418a3e05d4/394d91f0-2a3d-11ed-8f1b-0a49ba8114ab.",
#    "486724093467":"ResourceLogicalId:AccountSysAdminRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:AWS_486724093467_SysAdmin already exists in stack arn:aws:cloudformation:us-east-1:486724093467:stack/StackSet-DCSLayer0-AccountRoles-ITS-8ec6eaff-82aa-4da3-8d0e-71409404ecfe/72d7f6c0-d617-11eb-8735-0a87c278b5b5.",
#    "513764142827":"ResourceLogicalId:AccountDBAdminRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:AWS_513764142827_DBAdmin already exists in stack arn:aws:cloudformation:us-east-1:513764142827:stack/StackSet-DCSLayer0-AccountRoles-ITS-06c2ad64-3b98-4fad-bba5-3e22489943c1/b9d89440-ff74-11eb-adac-0af655d326e7.",
#    "517564158377":"ResourceLogicalId:CreateSAMLProviderRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:Create_SAML_Provider already exists in stack arn:aws:cloudformation:us-east-1:517564158377:stack/StackSet-DCSLayer0-AccountRoles-ITS-78834287-d539-41f2-80e8-441e4f5562c0/3b3b2750-d775-11eb-80cb-126643f85ad9.",
#    "543315275728":"ResourceLogicalId:AccountAdminRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:AWS_543315275728_Admin already exists in stack arn:aws:cloudformation:us-east-1:543315275728:stack/StackSet-DCSLayer0-AccountRoles-ITS-ad2fa428-683a-4feb-b4b9-b9de4a74618b/635137e0-d610-11eb-b7cd-0e688fb565d9.",
#    "551019182106":"ResourceLogicalId:AccountDBAdminRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:AWS_551019182106_DBAdmin already exists in stack arn:aws:cloudformation:us-east-1:551019182106:stack/StackSet-DCSLayer0-AccountRoles-ITS-1968b69e-06ba-4d0b-8b73-89ec5dbae91e/8b9c05b0-d76c-11eb-8210-0e69690dd155.",
#    "627069371650":"ResourceLogicalId:BackupRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:AWS_DCS_BackupAdmin already exists in stack arn:aws:cloudformation:us-east-1:627069371650:stack/StackSet-DCSLayer0-PersistentRoles-us-csd-consulting-6e5e906d-b71a-40c9-9d63-96e82a062216/6417d040-2a3c-11ed-b3d1-0e2f3ffeccf1.",
#    "654517000290":"ResourceLogicalId:AccountAdminRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:AWS_654517000290_Admin already exists in stack arn:aws:cloudformation:us-east-1:654517000290:stack/StackSet-DCSLayer0-AccountRoles-ITS-81d24dfa-436a-477d-b022-5f0ee343cfa3/72519350-d554-11eb-968e-0ec9de1de58d.",
#    "666800176692":"ResourceLogicalId:AccountAdminRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:AWS_666800176692_Admin already exists in stack arn:aws:cloudformation:us-east-1:666800176692:stack/StackSet-DCSLayer0-AccountRoles-ITS-4dc5caf3-0653-45f8-80aa-f30b9f57b18b/544da5f0-c06b-11ea-abf9-122e54527a47.",
#    "713018753281":"ResourceLogicalId:AccountReaderRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:AWS_713018753281_Reader already exists in stack arn:aws:cloudformation:us-east-1:713018753281:stack/StackSet-DCSLayer0-AccountRoles-ITS-44d3ae67-f150-43f5-a14d-c6f9d1a0d31f/0f6d6580-2a3e-11ed-a2ab-1208b06dceb3.",
#    "817952536107":"ResourceLogicalId:EventRuleRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:EventRuleRole already exists in stack arn:aws:cloudformation:us-east-1:817952536107:stack/StackSet-DCSLayer0-AccountRoles-ITS-f0cb3749-7793-4138-bb4d-63cf5081d73f/646afa70-d54e-11eb-9ef7-0ed02896373b.",
#    "889531594868":"ResourceLogicalId:AccountSysAdminRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:AWS_889531594868_SysAdmin already exists in stack arn:aws:cloudformation:us-east-1:889531594868:stack/StackSet-DCSLayer0-AccountRoles-ITS-0e0e0d38-ee52-4a4a-8662-a1339c2c49f6/b9cf49e0-d54d-11eb-9ac2-0a479dbf9bdf.",
#    "918223892161":"ResourceLogicalId:AccountReaderRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:AWS_918223892161_Reader already exists in stack arn:aws:cloudformation:us-east-1:918223892161:stack/StackSet-DCSLayer0-AccountRoles-ITS-93c3e96e-f0c2-4fa2-82ba-b5e19138ee9b/29feb030-d610-11eb-84a7-0a3049666ffd.",
#    "958991296092":"ResourceLogicalId:CreateSAMLProviderRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:Create_SAML_Provider already exists in stack arn:aws:cloudformation:us-east-1:958991296092:stack/StackSet-DCSLayer0-AccountRoles-ITS-10768bff-0154-4796-b5f7-543861e5001a/819c1e50-d6e1-11eb-af73-0a8bd56fe845.",
#    "965817487737":"ResourceLogicalId:CreateSAMLProviderRole, ResourceType:AWS::IAM::Role, ResourceStatusReason:Create_SAML_Provider already exists in stack arn:aws:cloudformation:us-east-1:965817487737:stack/StackSet-DCSLayer0-AccountRoles-ITS-87a3c974-285d-419a-acc7-2f50d9b29034/9b50f960-2a32-11ed-8b67-0a4728db477d.",


#    "210579071845":"Stack:arn:aws:cloudformation:us-east-1:210579071845:stack/StackSet-ITSLayer1-SPOKE-IAM-CLDTRL-NPD-1e9e4d30-d1ee-4630-8903-5e24a4afc993/4135f4a0-91fa-11ed-ab93-0e4a7eaa58e1 is in DELETE_FAILED state and can not be updated.",
#    "218515608600":"Stack:arn:aws:cloudformation:us-east-1:218515608600:stack/StackSet-ITSLayer1-SPOKE-IAM-CLDTRL-NPD-cc9b3980-01fa-49c0-8ef1-cc1bf0f1b1a8/b4194760-0843-11ed-9da7-0a1b4053af2d is in DELETE_FAILED state and can not be updated.",
#    "232162000477":"Stack:arn:aws:cloudformation:us-east-1:232162000477:stack/StackSet-ITSLayer1-SPOKE-IAM-CLDTRL-NPD-95c8472f-e07d-40c9-86ec-4a3ace9e495e/b482dea0-0843-11ed-980a-0a8dfe7862ef is in DELETE_FAILED state and can not be updated.",
#    "284328782567":"Stack:arn:aws:cloudformation:us-east-1:284328782567:stack/StackSet-ITSLayer1-SPOKE-IAM-CLDTRL-NPD-2bbff9ac-be5d-4823-858d-111e3f908c3a/b4567770-0843-11ed-938e-0a6c5ed0442d is in DELETE_FAILED state and can not be updated.",
#    "696360193991":"Stack:arn:aws:cloudformation:us-east-1:696360193991:stack/StackSet-ITSLayer1-SPOKE-IAM-CLDTRL-NPD-18965df6-8bbc-454a-8f77-2798c62d5e5e/c07e7a50-ce60-11ed-8922-0e543d69624d is in DELETE_FAILED state and can not be updated.",
#    "701862677052":"Stack:arn:aws:cloudformation:us-east-1:701862677052:stack/StackSet-ITSLayer1-SPOKE-IAM-CLDTRL-NPD-19dd644f-f9c3-43a4-981a-9fcbd39968ec/52b33020-6995-11ea-baa2-0ee5b74c6229 is in DELETE_FAILED state and can not be updated.",
#    "822566085433":"Stack:arn:aws:cloudformation:us-east-1:822566085433:stack/StackSet-ITSLayer1-SPOKE-IAM-CLDTRL-NPD-07786c1b-52aa-47da-9839-47143094c68e/c19a2380-ce60-11ed-b001-12c78aa37b85 is in DELETE_FAILED state and can not be updated.",
#    "479008266961":"Stack:arn:aws:cloudformation:us-east-1:479008266961:stack/StackSet-ITSLayer1-SPOKE-IAM-CLDTRL-NPD-32e21e44-97da-450f-9ba1-92c22659440c/b1fc9e20-7a95-11ea-9664-12301089d57f is in DELETE_FAILED state and can not be updated.",
# }






# sso_roles = {
# "201944482039":"EventRuleRole",
# "282181226940":"AccountAdminRole",
# "498306521888":"CreateSAMLProviderRole",
# "511898103173":"AccountSysAdminRole",
# "552923642514":"TenableIORole",
a = {'122578856689': 'AccountDBAdminRole', '240044539051': 'CreateSAMLProviderRole', '264944699463': 'AccountSysAdminRole', '342472205137': 'AccountSysAdminRole', '420525251589': 'CreateSAMLProviderRole', '457026597750': 'AccountReaderRole', '486653307662': 'CreateSAMLProviderRole', '486724093467': 'AccountSysAdminRole', '513764142827': 'AccountDBAdminRole', '517564158377': 'CreateSAMLProviderRole', '543315275728': 'AccountAdminRole', '551019182106': 'AccountDBAdminRole', '654517000290': 'AccountAdminRole', '666800176692': 'AccountAdminRole', '713018753281': 'AccountReaderRole', '817952536107': 'EventRuleRole', '889531594868': 'AccountSysAdminRole', '918223892161': 'AccountReaderRole', '958991296092': 'CreateSAMLProviderRole', '965817487737': 'CreateSAMLProviderRole'}


# delete_failed = {'210579071845': ' DELETE_FAILED state and can not be updated.', '218515608600': ' DELETE_FAILED state and can not be updated.', '232162000477': ' DELETE_FAILED state and can not be updated.', '284328782567': ' DELETE_FAILED state and can not be updated.', '696360193991': ' DELETE_FAILED state and can not be updated.', '701862677052': ' DELETE_FAILED state and can not be updated.', '822566085433': ' DELETE_FAILED state and can not be updated.', '479008266961': ' DELETE_FAILED state and can not be updated.'}

# others = {
# "553697376517":"CloudscriptValidatePrdCrossAccountRole",
# "163564840054":"CloudscriptValidatePrdCrossAccountRole", '087549363555': 'ResourceStatusReason:DenyITSLayer1PlatformModification already exists.', '239823972708': 'ResourceLogicalId:CloudscriptValidatePrdCrossAccountRole, '}






# https://github.com/Deloitte/dcs-aws-platform/blob/87564a4c80072e3bd321d9c3a9997a20533e459c/src/global_layer_0/cleanup_sandbox_account/functions/cleanup_sandbox.py

# {  
#    "StackSetName":"DCSLayer0-AccountRoles-DE-ITS",
#    "ParameterKey":["CreateSAMLProviderRole","HubAccountId"],
#    "ParameterValue":["LambdaCreateSAMLProvider","210240447631"]
# }


# npd_accounts = ['002470666470', '003259701703', '003372875111', '003866754167', '004629961456', '005241645787', '005333532456', '007707791333', '008530717892', '009296111214', '009348927679', '009464921414', '010553816281', '010753539596', '010817157965', '012095546580', '012680939556', '014296065027', '015257350768', '016937230970', '017432556837', '017456215898', '017848332886', '018060595034', '018649475058', '019347635701', '021000367750', '021365338423', '022169245411', '022618988199', '023146164982', '026194564518', '026427244252', '027393114528', '028533194311', '028568440897', '030974280115', '033014299223', '033246515043', '034322051670', '035483035615', '035778782477', '035825521573', '037355734743', '037388063884', '038059618311', '041149546493', '041479117925', '042115301607', '043262203499', '047685407929', '047807787807', '051469546478', '051724048437', '051799833889', '051994682883', '052346183910', '053070086514', '053859713267', '054273887971', '055064962277', '055840983865', '056630023496', '056863834668', '056895686976', '058611342228', '062130537180', '063353051663', '063376279610', '066979683739', '067154635066', '067350284291', '068757897087', '068900102323', '069580900393', '070036689193', '070660504732', '072706165844', '075458757601', '077240679103', '080656668290', '084407457870', '084582856213', '084970654249', '085773352584', '086573493719', '086574853384', '087549363555', '088220075168', '088563067471', '089587160124', '090446900710', '091115685892', '091284928336', '092327206231', '092944838799', '092960230608', '093441740690', '094299405254', '094827805839', '095139704753', '095611350051', '097020054916', '097107211992', '097475337748', '097795456520', '098000758385', '098556162587', '099325924796', '101177931550', '101796278814', '103896790399', '103917580387', '104590855980', '105723104798', '106456323084', '107282117866', '107831069545', '109997849406', '110557118724', '110613030397', '110633459663', '110912366499', '111535522048', '112558980097', '113176856871', '114535277687', '114727038176', '115154971638', '117523786847', '117725889170', '118974102263', '119203704260', '120152586068', '121964457821', '122578856689', '122994942344', '123155530402', '123906516738', '124524166810', '125237375797', '126523394397', '126667425912', '128731681412', '129330666742', '130149840165', '130220492427', '131610679465', '131660845369', '132027580256', '133143185602', '133550674684', '133762489993', '134679210625', '135185876313', '135285569980', '135755270796', '136473818407', '137885247948', '138233832997', '142332166808', '142646231261', '143247161605', '143711193266', '144475990439', '144627418504', '144635163526', '146003574029', '148094740375', '149089727609', '149148767097', '149477848214', '149878268794', '150316879294', '150767860677', '151434629482', '152223477882', '152898684328', '153685446838', '153714429732', '153772258784', '153951089943', '156487489205', '159602377297', '159861609278', '159921825654', '160327063738', '160339976030', '160934439473', '161391293302', '161394336152', '161636799113', '162022282563', '163564840054', '164398058260', '165159114599', '166101436673', '167187402137', '167226961519', '167393382127', '168441963426', '168849861669', '169515270762', '170140899813', '170394039252', '171762472359', '172067301824', '172832121763', '172863928423', '173538766175', '174295623751', '174961374134', '175091774301', '175924461164', '175930311790', '175956104120', '176996053919', '178790212740', '181613734321', '184405994032', '185020060020', '185284577515', '185671863613', '186861417013', '187528728268', '187879053795', '188529011465', '190367388287', '190642798438', '192614291797', '193361093508', '193854970547', '194695465174', '194928253959', '195566443221', '197821584623', '198092089923', '198271522709', '199099526724', '199360944383', '199570988326', '200228513821', '200543695574', '201823333330', '203158740621', '204053091687', '204630464347', '205063679632', '205707500197', '208421051549', '209441755774', '210264310230', '210579071845', '210632014336', '210798437721', '211101344176', '211508935776', '211593031182', '211600541199', '211695053472', '212271771810', '212875955720', '212921778948', '213911605817', '214979362845', '215611426819', '216542771521', '217360121513', '218515608600', '220730334052', '220933860874', '221112619429', '222001666987', '222020306244', '222531082059', '222624872065', '223441765792', '224244178360', '225658821723', '225805886800', '225832781273', '226603863404', '227841379693', '227937082667', '228115329917', '229679863806', '231716799145', '231972052922', '231974784973', '232162000477', '232448574943', '233127284560', '233256077443', '234772266870', '235513223743', '236248596881', '238557733178', '238820089267', '238970376046', '239205171373', '239823972708', '240044539051', '240715832728', '241396937347', '242199234071', '242753836132', '245250269795', '246197285750', '246585516170', '246698818014', '246704155817', '246938629297', '247048948584', '248014647110', '249285578332', '251072340318', '251641150286', '251985550364', '252077205715', '252323805874', '252365818154', '253933647263', '254263703118', '255798633318', '256489642696', '256706112543', '256867917700', '259692210398', '261934197954', '263128578697', '263648093447', '264619089984', '264629864623', '264944699463', '265756937744', '266010434788', '266396698525', '266955691786', '266980808703', '267316576554', '268522796693', '269475816966', '269734407326', '270950599001', '271937290370', '272298319762', '272606000076', '272851264686', '275309946303', '275380584232', '277167574796', '277615250575', '277922364851', '279462431622', '279743555714', '280522856839', '282053871296', '282272865295', '282752622212', '283503555640', '283520033493', '284328782567', '284760807629', '286814544202', '287248264930', '288228195021', '288952646112', '291704570552', '292210083552', '292767457610', '292995354700', '294658461048', '294745394203', '295562148805', '295998058776', '296351982715', '296916826127', '297486902890', '297947569761', '298561899742', '300801515163', '301605300944', '303364506186', '305890512688', '306182918611', '307036072367', '307036839104', '307133528178', '307492883559', '308360630103', '309416070659', '309727501936', '312807674804', '314285588528', '314448338500', '315370314208', '315724403350', '315952988300', '316021091064', '316461870015', '316625284770', '318783272265', '319338734333', '320199641064', '320510073098', '321267289192', '321307503701', '321376957942', '322601554261', '323232363111', '323817093338', '323911653333', '324789170316', '326755649394', '327639925109', '328061036419', '328235541546', '329106378841', '329783930457', '330374005851', '330459082734', '332607986083', '334734394521', '335893736669', '336512527178', '337910336443', '340826808143', '340942712854', '342122824633', '342163036816', '342416616976', '342472205137', '342788227825', '343248351138', '344161786867', '344653063612', '345404735839', '345889295625', '347229344523', '347913913427', '349091595040', '350229878820', '351943365646', '354173411065', '354366284558', '354716619065', '358132790115', '358508000257', '359252437790', '359695216436', '361804686539', '362629117613', '363220349094', '363235117348', '363727823241', '364513602188', '365111145171', '365379373575', '365522681417', '365655694970', '365664458300', '365753072011', '366779305333', '366862120963', '367409673491', '368276533222', '369102312058', '369770340971', '370020434536', '370908162876', '371826770560', '372203661552', '373510380303', '373758253227', '373943505089', '375028686807', '376460358145', '376614162583', '376839262288', '377602907415', '380737479390', '380769935574', '380906577769', '384921830497', '385078759766', '385397780492', '385570000518', '385644467038', '386372166532', '386423219226', '386880889362', '386951795368', '388634006099', '390217200304', '391931929196', '392479946920', '393624352404', '393663394648', '394326658915', '395562096909', '396547825296', '397709778342', '399553586947', '403053226718', '403398797571', '406236698787', '406850891275', '407294690266', '407922971056', '410993274172', '413145654121', '413418342606', '413582460477', '413990510913', '414665010361', '415134345932', '415381292284', '416842063547', '418363115654', '419451308158', '420191660637', '420525251589', '421953516327', '423930725445', '424467068503', '425607349574', '426743316697', '427183723235', '428677740062', '429242996634', '431231529702', '431903633673', '432089771844', '434331732926', '435453563095', '436310111307', '437813383155', '438050491659', '438720243151', '440737389801', '440856099504', '441012919436', '441189720969', '441340786764', '442073219544', '442716992213', '442888752927', '443249815747', '444895787476', '445066270436', '445421740248', '446334086509', '446521373869', '447033644314', '447344437245', '447579392870', '447718312100', '448362402750', '449510142326', '450542789652', '451268808790', '452055674543', '454708018951', '455771646044', '456082790337', '457026597750', '457326670944', '458178772538', '458599620270', '461075792974', '461639330540', '462532790342', '463479487416', '463680094014', '465638941851', '467325410942', '468323486041', '470306404467', '471501708651', '471936325112', '472167960556', '472274473037', '473012084470', '473144355006', '474547691682', '475182315323', '477730144921', '477760339841', '478518164695', '478577616413', '478925142388', '479008266961', '479017279804', '479521414473', '480022745122', '480208011542', '480475536219', '481315189088', '482815256356', '483069708168', '483078657402', '484163937975', '485042675174', '486238640158', '486528187221', '486653307662', '486724093467', '488283147041', '489537064020', '490290534637', '490455086580', '491271408797', '491642886103', '492634300165', '492906408022', '493857221609', '494101600798', '495648771014', '496900188711', '497315343129', '497647849039', '497934097495', '498536333823', '498711219760', '500008131718', '500482683615', '500980881293', '501765712544', '502327753374', '503351113243', '503741562442', '504762789322', '504857848042', '505828008366', '506633848063', '506696121897', '507297579984', '507458175641', '509386356242', '511051334879', '511507869156', '512171866621', '512382550508', '512889038796', '513764142827', '514036822646', '515808197273', '517564158377', '518664229968', '520846418629', '522328802007', '524275599438', '524658522586', '525103131843', '525174722646', '525729546547', '526450483356', '526529371479', '527319450606', '527490275647', '529277024000', '531268065622', '531329070867', '531569583280', '531659201158', '534534201911', '534846924117', '535700223023', '536017423151', '537671715099', '537849064119', '539278275488', '540791212970', '540957243980', '541144414411', '541833898623', '542172426746', '542562960121', '543315275728', '544133436431', '544510848374', '544734843935', '545298271467', '546138792195', '547043556405', '547250268408', '547500692192', '547509476297', '547934214792', '548659743877', '550446022751', '550812425828', '550882646259', '551019182106', '551587375342', '551799593302', '552291554780', '552668850488', '552903841159', '552923642514', '553514781467', '555869646217', '556525398552', '557955772651', '558610732496', '559651534567', '560024577736', '561320747596', '561546907559', '561971607098', '564066165061', '565133027204', '567514599659', '567664804420', '568533325689', '569594090084', '570696484411', '572023949181', '572961371493', '573292697388', '573313377656', '576524227815', '577914934418', '578066212474', '578316533664', '578660118980', '579155750082', '581047588350', '581775067842', '581985365737', '583190184710', '583303984486', '583345763035', '583608845417', '583989446449', '584630848692', '584974054351', '586482989402', '586805444392', '588773227252', '588809974228', '588949991295', '589727330895', '589892584776', '590506744596', '592564898892', '593190477397', '593391513284', '594239924270', '595353211380', '596896391463', '598716746971', '599187033811', '599569244858', '600142321150', '600424674755', '600555062417', '601290880738', '602644925125', '602706386284', '604033226315', '604318285507', '604707100243', '604723096881', '606511248153', '607575999558', '608583646195', '609468539593', '611120875422', '611316385221', '614092787540', '616194183804', '616614454022', '616864236654', '619153066025', '619171252003', '619768731561', '623701539211', '625001204664', '625125552135', '626107101308', '627069371650', '628157037413', '628625382433', '629026071019', '630697057306', '630958910218', '634691564906', '635675837571', '636712211284', '637454724728', '637839024236', '638334832935', '638365212721', '639536544457', '640234854210', '641441075861', '641938356393', '642264126148', '643596949585', '644036824268', '644141345393', '644784173767', '644849997084', '645182689106', '646023701073', '647087774510', '647885244255', '648585965371', '650018817833', '650149263190', '651491549966', '651509282451', '652125692913', '652254402111', '652437243948', '652943102365', '652993727289', '653060988075', '654075011442', '654174153064', '654517000290', '654872607431', '655572130710', '656483505048', '658062093174', '659789841918', '661045825192', '661881507687', '663321287581', '664233298234', '666800176692', '667027443162', '667485214512', '667662266003', '667723493591', '667900912763', '670161819556', '671629789513', '674058067202', '674583080303', '675248765338', '675771396901', '679245664611', '679578327390', '680660665682', '681628658461', '682033880221', '682654310819', '684560741695', '688405658638', '689468357458', '690318954491', '690966616852', '692721260310', '693542608313', '694345650500', '694674735136', '695382570272', '695506696950', '696344538400', '696360193991', '697312559295', '697642837822', '697872195742', '698031681033', '698776863250', '699486026393', '699736171795', '700210323933', '700777391571', '701275957923', '701862677052', '703183665370', '703674870110', '703906235327', '704025107641', '705043942076', '705182684277', '705494424374', '705566307488', '705854351142', '705967963585', '707012202561', '708437838039', '713018753281', '714967854867', '716751674484', '717502159111', '718832137337', '719283693773', '720257876889', '720366697201', '720724015417', '720745720430', '721284546788', '721369432361', '721523660704', '723175378301', '723788595012', '724311065114', '724343007644', '725260281549', '726327367093', '726750907807', '727456174053', '728061972801', '728483363736', '729469735401', '730031397153', '730275874014', '731923714466', '733804268833', '734998425718', '735837376890', '736530749364', '736713133364', '737141037132', '737192344670', '738731325101', '739672651720', '740053198814', '740123709034', '740166810819', '741274041958', '743027775537', '744946510523', '745576527420', '745686942713', '746080954153', '747329189279', '747602229289', '748278030583', '749115363610', '750227276023', '750369583115', '750808625910', '751142013127', '753455713898', '753880925017', '753904791510', '755216844945', '755312818847', '756414919982', '758763400630', '758843481727', '759209259634', '762478760418', '763186303361', '763269149154', '763639998517', '764490523317', '766245584008', '766533495919', '766554992149', '767507899515', '767813017116', '767848537480', '767857357979', '768955793546', '769986034361', '770381447961', '770600390810', '770817124419', '771557547238', '771645514271', '773334544016', '773467821022', '775595962189', '775762748442', '776173975841', '776440666163', '777601016429', '778075340046', '779529038689', '780113007310', '784441283169', '785183849335', '785786920527', '785817577328', '786289358306', '788093788772', '788247499336', '789476507634', '790874520702', '791968533226', '792374790622', '794501456704', '794569136533', '795085004402', '796402573673', '796591184867', '798831350617', '800436161211', '801074580788', '801334471752', '802825831136', '803115734520', '803466671685', '803611162529', '807269076110', '807394297581', '808028441788', '808458935611', '809498814746', '809887086749', '810358558433', '810686302806', '812263245315', '814255944148', '814566149193', '816813650017', '816919933351', '817952536107', '819418488580', '821346124874', '821816138949', '822274905021', '822566085433', '822955369949', '823523588426', '824349500053', '824943661652', '826919501912', '827687231818', '828059665819', '828170655941', '828471328199', '828707026805', '829296092291', '829331138793', '830458620727', '830960023363', '831213597264', '831428250184', '831795865367', '831811033311', '834357290412', '834501561769', '834806916487', '836073812853', '836346912461', '837289689095', '837626969847', '837956138358', '838910886206', '839326706163', '841003316687', '841764116136', '843058968294', '843385628534', '843968153990', '844702767392', '845115198660', '845663238523', '846230196886', '847198424597', '847293893566', '848219600572', '848793886259', '849045193397', '849379670310', '850795100153', '852472875055', '853713850743', '855484158442', '855777208844', '856019220660', '856464429809', '858141963590', '858312341223', '858526972147', '858633904165', '858857573117', '859828415022', '860178592278', '860680429784', '862094248680', '863953187386', '864080733145', '865408716471', '865654561421', '866203773640', '868521031821', '868874314265', '868978391936', '869371544050', '870562742571', '872335144391', '872425586473', '874896561479', '875314453109', '875752141512', '877180364429', '878348541116', '879472252061', '881051476853', '881451740496', '882194302238', '882722950271', '883454950117', '883784239155', '885718323853', '885924542129', '886015701061', '887119163707', '889531594868', '890591524509', '891047194812', '892095653963', '893007952058', '893441358747', '893649511657', '893725561486', '895081226716', '895114609877', '895796344347', '896284331846', '897265563406', '899956852123', '900397174018', '901519578274', '902077756344', '903662551929', '904898881486', '906924434830', '910380970013', '910997097838', '911444749492', '911593296289', '912681226107', '913412532161', '913593522947', '914462607210', '914550290266', '915366716420', '915431102062', '915814505567', '916086415886', '916378955819', '916396089121', '917303654463', '918223892161', '918704104394', '918824989007', '919367393195', '920078571165', '920815465273', '921134338319', '921400275636', '924069284639', '924294269964', '925017383941', '926372856456', '927753750267', '928599649920', '929708144712', '929888664800', '930328442017', '931334696467', '932207706514', '934472131788', '935855703025', '936639518272', '937767124938', '938247196952', '941819694106', '942547771823', '942973508166', '943200330016', '943654527978', '943942458547', '944713022036', '947692952239', '948310893572', '948572612712', '948948595631', '949050008497', '952135726105', '953636518551', '954695687190', '954738915509', '954742235303', '955038580167', '955116103754', '955621810722', '956058981677', '956885745546', '957304465555', '958991296092', '959629872806', '960737463278', '960956162241', '961972215724', '962383023396', '962612127814', '962639478670', '963363688713', '963365592956', '964131740148', '965266634360', '965817487737', '966309006358', '970120047911', '970224627637', '971295728977', '971552591868', '971557445175', '971836208243', '972656844555', '973653897100', '974255779466', '974936500263', '975204971987', '975817023078', '978553898267', '978671209663', '979212527518', '981805318608', '982693706514', '983350415204', '983728535136', '985410213753', '985676819172', '986012759135', '986038433815', '987148916101', '987825197801', '988601501448', '989150551987', '990148784831', '990223392234', '991022102975', '992200855368', '992908013717', '993130990278', '993836073940', '993932001969', '994090975690', '994920118214', '996010827026', '996637069404', '997205032839', '997667771749', '998732324712', '999954735781']
# prd_accounts = ['002455533990', '003788644068', '004420648591', '011342422378', '012046266255', '016101336017', '018472551420', '021191302697', '022791440266', '024517926231', '024838550089', '025625448852', '025648210413', '028002051339', '030263312215', '031043502080', '031573843522', '033065397244', '034307784415', '034526171824', '034536176640', '034749574334', '035784263114', '036093542835', '038144103111', '038270351176', '039233901334', '039929384510', '040046960803', '041081560071', '043330710293', '047209783566', '047749980333', '050877488444', '052819490290', '053082227562', '053392642336', '053444045268', '055182143107', '055266516288', '056412096258', '059272759707', '060465202534', '062387122580', '065911132104', '067447795879', '067555441682', '067804797237', '068183022879', '068315429392', '069962244880', '070717231209', '072038240142', '073208257581', '073546915641', '078769231505', '084367452833', '084383639626', '086102618356', '086301938642', '087127721683', '088258796410', '092606503522', '093916721094', '102145276445', '102302903403', '102771375180', '104624496682', '107393473960', '109577287868', '116222102902', '119753528152', '120311362244', '122600361133', '124841701563', '127513335444', '128264176259', '128346779966', '130203272373', '131750173436', '133277734215', '135381627630', '139100456334', '142803759194', '143312840871', '144544136090', '148392725318', '154843675742', '155305736938', '156327752252', '157812473613', '158347257464', '158684596895', '161395754509', '161775556864', '162310846309', '162435009311', '163564840054', '163918101077', '166449560162', '166556434326', '168131472534', '169912408263', '170178091998', '170302206609', '172383415443', '173316567229', '173373426076', '173479391558', '174145408811', '174188575348', '174278580167', '175472836528', '176443733777', '176539146345', '181395753476', '181864525870', '185075825010', '186574298919', '188831284171', '194751376404', '195616859884', '198162557634', '201547482655', '201944482039', '202120059134', '204193633058', '204385000248', '204485639007', '205394135495', '206793667976', '208073657936', '208626548723', '211070001969', '212149607166', '212773838067', '217413253555', '218167183912', '220288712633', '222023357012', '222762603060', '228052005984', '232144315514', '232698387549', '232930013947', '233632481017', '234019974018', '234848667271', '239459044292', '240344030626', '241671150056', '241693290537', '241696072805', '247012361096', '251058218532', '254313626187', '254471163821', '255311346772', '257915333181', '258860010574', '262259241401', '266527211573', '268527685757', '276287038763', '277946231630', '281981188021', '282181226940', '282623329273', '283711851771', '284383242428', '290092688251', '294297099413', '298370672248', '305213819503', '305317527087', '307670020531', '308350474287', '309245159336', '311642618236', '312249472578', '318724812262', '318902570495', '320383732799', '320947721229', '321026552774', '327370950832', '328754255717', '329503450035', '332205887294', '333186577928', '334144298543', '334352873287', '336222303362', '336827305173', '340021236041', '340179473373', '340860049954', '345414987291', '345595150906', '354362986338', '355214048684', '363191287345', '366986788235', '366987310429', '367544918569', '371132510028', '371209647698', '372988589375', '373124920883', '373337982383', '374125395891', '376587057974', '381173206319', '385159429807', '388830625952', '389222194029', '389613005727', '392047261520', '398358669351', '403327078303', '404335588036', '410871088857', '414878987508', '415076259107', '415096990548', '416380726237', '417276993801', '420924285556', '422660281210', '422716773809', '424023865363', '425844425297', '426187456338', '426325367657', '426843455707', '426909158075', '431768475246', '433112528293', '433860551753', '434820176994', '436950664991', '440708338928', '441977708642', '442279558925', '443482151530', '445544608757', '446543602545', '447099417955', '449696838726', '454243075918', '454878478860', '455295949501', '456012650670', '456716556996', '457633433589', '458712402796', '459741916297', '460579496107', '463748631505', '470892628663', '471101500246', '471764684541', '473250842200', '475312077285', '475679390152', '475704409833', '480207372766', '481383093164', '482709997193', '484797066964', '486318442788', '486687906253', '487825727312', '488502120284', '488791035777', '490723148876', '492182230036', '492809362056', '494419598111', '497228614851', '498306521888', '504206122956', '510920009915', '511898103173', '512593562953', '514641961978', '516386627390', '521756226937', '522610480753', '523122162952', '523471060002', '523890020650', '524666392392', '525684156106', '526851179788', '529102001334', '531885649505', '534939542028', '536825901754', '537669063398', '537943953334', '538115423334', '540404953683', '540656245819', '541061827746', '542083475947', '542734845675', '542906167192', '543492705153', '543691652517', '544206404384', '544536937525', '545240029200', '545560008065', '545994597414', '547578050363', '551757942034', '552923642514', '553526484992', '553697376517', '556116077166', '556759822564', '558067152068', '560596504822', '563511753176', '564649271459', '565010824322', '568396353920', '570051990040', '571208668853', '572027877266', '573859772831', '580375422572', '580411221872', '581130070053', '581328788659', '583106373569', '585514194047', '586329412044', '588607312191', '589202808364', '589327882798', '592578543696', '593780602920', '594796041417', '595716108428', '596414525875', '597373045724', '597442721103', '598578472115', '598783347326', '601947996617', '607851404322', '608959785178', '609608201737', '611480974551', '613809103576', '614751792119', '615475426900', '615864598532', '617366402669', '618143286284', '618222675128', '621577900889', '621848218806', '622372203913', '622913598307', '623899618944', '623940634196', '624603845543', '624925071076', '625466519857', '626553292796', '628943449194', '630033476802', '632621214384', '632954654882', '635786325450', '637376217363', '637928158385', '639369182738', '640259690623', '643987766248', '644969574939', '645286070021', '645683819147', '646717608788', '648080342736', '650644976608', '651409190172', '657107451157', '659069636583', '659576002760', '662929856130', '670111638441', '670404736065', '670999639077', '678835796394', '680069428997', '681044841647', '681356837841', '682655211644', '683133891344', '683173620724', '683340563767', '684757896148', '685107354181', '686387901187', '688120237422', '696141481732', '698551361437', '699789982634', '701438804970', '701970095679', '702011104433', '702064102213', '703829513590', '706151101526', '707763339947', '710230637982', '710465056204', '717090925392', '717151169743', '721294810136', '721462236188', '723606622699', '726012408397', '727600915853', '728610086968', '729269827574', '731082648661', '735624102611', '736819047443', '738852525135', '740123642997', '740153750092', '741012202819', '741732532483', '743862158735', '744048656618', '746177754969', '746203666711', '746805879165', '747229060202', '749257892087', '750378041276', '750802954404', '751720771238', '753874989666', '755090050495', '757031139434', '758360774458', '762594668164', '763232339158', '764177646170', '764230540928', '769553063621', '769985493362', '773559476152', '775302474189', '776522867075', '777973684917', '778558837889', '779420033654', '779558345599', '781297070656', '782998046294', '791422305768', '791959462774', '792126690708', '792645976151', '796155319658', '796937856387', '797231615719', '798052893433', '798485367562', '799867849121', '804269864808', '805039658749', '805522167547', '806037382723', '810613220646', '812790839692', '813055922763', '813558814731', '814582850999', '816348406744', '819300110664', '819716157095', '821372114949', '826429160863', '827001787104', '828290435030', '829955684988', '831323883328', '831587722162', '835173813168', '836686594237', '836809512233', '840214933274', '840365407675', '841168632450', '841208569552', '844577878525', '844677627151', '850702558675', '850791908078', '851787832323', '854481054617', '857578946837', '861990178126', '867168973137', '867327676348', '872800060950', '873697805464', '875635120922', '877574383758', '878153322966', '879286378355', '884435246449', '886409264347', '896914459532', '898090075122', '898483634510', '900406205346', '901022633716', '903186577214', '904005903356', '904168909248', '905676609443', '912103569023', '914628935504', '915503999640', '918013418617', '918701091807', '919235719473', '920037858723', '922274129305', '923052746422', '923931686599', '924473885417', '928604188985', '929126023852', '932346075053', '933831760677', '936096392668', '936117866411', '936363923026', '936680646698', '942265044478', '943971716662', '947872759685', '949955964069', '951802633484', '957132350705', '957518831560', '957627549507', '957734782020', '957966502020', '960841581879', '961541317843', '963879624041', '964426436180', '965416789171', '967227327126', '967972633959', '968301499415', '969413062254', '970156530737', '971424449433', '973896107202', '982104967400', '982190825179', '982935446751', '984281569836', '984625537202', '989504077972', '991428470003', '993001063892', '997715449024', '998663967865', '999355684375']

# npd_accounts.extend(prd_accounts)
# l1_accounts = npd_accounts


# l0_accounts = ['002455533990', '002470666470', '003259701703', '003866754167', '004629961456', '005241645787', '005333532456', '008530717892', '009296111214', '009464921414', '010373844257', '010553816281', '010753539596', '010817157965', '011342422378', '012046266255', '012095546580', '012680939556', '013724545051', '014296065027', '015257350768', '016101336017', '016937230970', '017432556837', '017456215898', '017848332886', '018060595034', '018472551420', '018649475058', '021191302697', '022618988199', '022663055756', '023146164982', '024517926231', '024838550089', '025502614753', '025625448852', '025648210413', '026194564518', '027393114528', '027804237073', '028002051339', '028568440897', '030263312215', '030974280115', '031043502080', '031573843522', '033014299223', '033246515043', '034307784415', '034749574334', '035657162383', '035778782477', '035784263114', '035825521573', '036611309112', '037355734743', '037388063884', '038059618311', '038144103111', '038270351176', '039233901334', '039929384510', '041081560071', '041149546493', '042115301607', '043262203499', '047681963590', '047685407929', '047807787807', '050111803238', '050877488444', '051469546478', '051724048437', '051799833889', '052346183910', '053070086514', '053392642336', '053859713267', '054273887971', '055064962277', '055182143107', '055266516288', '055840983865', '056895686976', '057239939437', '058611342228', '059272759707', '060465202534', '062130537180', '062637643455', '062790058996', '063353051663', '063376279610', '066569650784', '066979683739', '067154635066', '067555441682', '067804797237', '068183022879', '068315429392', '068757897087', '068900102323', '069962244880', '070036689193', '070660504732', '070717231209', '072038240142', '072706165844', '077240679103', '077365842699', '080656668290', '081577392665', '084383639626', '084407457870', '084582856213', '084707884338', '086573493719', '086574853384', '087127721683', '087549363555', '088220075168', '088258796410', '088563067471', '089402802109', '089587160124', '090446900710', '091115685892', '091284928336', '092606503522', '092944838799', '092960230608', '093441740690', '094552460557', '095611350051', '097020054916', '097107211992', '098556162587', '099189045841', '099325924796', '101177931550', '101470261478', '103896790399', '103917580387', '104624496682', '105723104798', '106456323084', '107282117866', '107393473960', '107831069545', '109577287868', '109997849406', '110557118724', '110633459663', '110912366499', '111535522048', '112558980097', '114535277687', '114724603768', '114727038176', '114858314951', '115154971638', '116452690986', '116909240956', '117523786847', '117725889170', '118974102263', '119203704260', '119373311443', '119529477927', '119753528152', '120152586068', '120379190565', '121652837644', '122600361133', '122616578453', '122674790828', '122994942344', '123686375669', '123906516738', '124524166810', '127095091672', '129330666742', '130203272373', '130220492427', '131372407359', '131660845369', '132027580256', '133143185602', '133550674684', '133762489993', '135185876313', '135381627630', '135755270796', '136473818407', '138233832997', '139100456334', '139914741522', '141578125655', '142332166808', '142646231261', '144544136090', '144627418504', '144635163526', '146003574029', '148094740375', '149477848214', '149878268794', '150316879294', '150767860677', '151434629482', '151577017449', '152223477882', '152692632441', '152898684328', '153125681988', '153685446838', '153714429732', '153951089943', '155305736938', '156327752252', '156487489205', '158347257464', '158684596895', '158830002625', '159602377297', '159861609278', '159921825654', '160133382852', '160339976030', '160934439473', '161391293302', '161394336152', '161395754509', '161636799113', '162022282563', '162310846309', '162435009311', '163564840054', '163918101077', '164398058260', '166223901165', '166556434326', '167393382127', '168131472534', '168849861669', '170140899813', '170178091998', '170302206609', '171762472359', '172067301824', '172296105687', '172383415443', '172832121763', '172863928423', '173316567229', '173479391558', '174188575348', '175091774301', '175924461164', '175930311790', '175956104120', '178790212740', '178798771574', '181395753476', '184405994032', '185020060020', '185075825010', '185284577515', '186574298919', '186861417013', '187528728268', '188529011465', '191346140709', '193361093508', '193854970547', '194751376404', '197821584623', '198092089923', '198162557634', '199570988326', '200228513821', '200543695574', '201547482655', '201823333330', '201944482039', '203622632891', '204193633058', '205063679632', '205394135495', '205707500197', '208073657936', '208421051549', '208626548723', '210264310230', '210579071845', '210798437721', '211070001969', '211101344176', '211593031182', '211600541199', '212149607166', '212773838067', '212875955720', '212921778948', '213911605817', '214979362845', '215611426819', '216542771521', '217360121513', '217413253555', '217463887104', '218167183912', '218515608600', '220288712633', '220933860874', '221112619429', '222023357012', '222531082059', '222762603060', '223441765792', '224244178360', '224286928353', '225658821723', '225832781273', '226603863404', '227841379693', '227937082667', '228052005984', '228115329917', '231716799145', '231972052922', '231974784973', '232162000477', '232698387549', '233127284560', '233256077443', '233632481017', '234019974018', '234848667271', '235513223743', '236244464571', '238557733178', '238820089267', '238970376046', '239205171373', '239459044292', '239823972708', '240044539051', '240210116683', '240344030626', '240715832728', '241671150056', '241693290537', '241696072805', '245250269795', '246197285750', '246585516170', '246698818014', '246938629297', '247015892866', '247048948584', '251058218532', '251072340318', '251985550364', '252323805874', '252365818154', '252702702909', '253026504241', '253034832134', '253123616265', '254263703118', '254471163821', '256489642696', '256706112543', '256867917700', '257173906640', '257915333181', '258860010574', '259692210398', '261934197954', '262259241401', '263128578697', '264944699463', '265756937744', '266396698525', '266527211573', '266955691786', '266980808703', '268522796693', '268527685757', '269475816966', '269734407326', '270950599001', '271937290370', '272298319762', '272606000076', '275380584232', '277167574796', '277946231630', '279462431622', '279743555714', '280522856839', '281981188021', '282181226940', '282272865295', '282623329273', '282752622212', '283503555640', '283520033493', '284328782567', '284383242428', '284760807629', '287248264930', '288228195021', '288952646112', '290092688251', '292210083552', '294297099413', '294454434120', '294745394203', '295562148805', '296916826127', '297486902890', '298370672248', '298561899742', '300801515163', '300825029455', '301923971814', '304084972275', '304576424523', '304844249827', '305132896877', '306182918611', '307036839104', '307133528178', '307492883559', '308360630103', '309245159336', '309416070659', '309727501936', '312249472578', '312807674804', '314285588528', '314297002272', '314448338500', '315370314208', '315952988300', '316461870015', '316625284770', '316658889291', '318724812262', '318902570495', '320199641064', '320510073098', '320947721229', '321267289192', '321376957942', '322885182098', '323911653333', '324789170316', '327370950832', '328235541546', '328741113609', '328754255717', '329783930457', '330248483315', '330374005851', '330459082734', '332205887294', '332607986083', '334734394521', '334873317999', '335893736669', '336222303362', '336385788515', '336827305173', '337910336443', '340826808143', '340860049954', '340942712854', '341813341134', '342122824633', '342416616976', '342451229934', '342472205137', '342788227825', '344161786867', '345404735839', '345414987291', '345693200111', '345889295625', '347913913427', '349082719353', '350045283196', '350229878820', '354362986338', '354716619065', '357362362136', '359252437790', '359797200114', '361804686539', '363191287345', '363235117348', '363727823241', '364513602188', '364980597341', '365111145171', '365322098407', '365655694970', '366779305333', '366862120963', '366986788235', '366987310429', '367409673491', '367544918569', '368276533222', '369102312058', '369770340971', '370020434536', '370908162876', '371826770560', '372203661552', '372886211720', '372988589375', '373124920883', '373758253227', '373943505089', '374125395891', '375028686807', '375127515215', '376072711816', '376587057974', '376614162583', '377602907415', '380906577769', '384921830497', '385078759766', '385159429807', '385203934575', '385397780492', '385644467038', '386372166532', '386423219226', '386880889362', '386951795368', '389613005727', '389956108244', '391931929196', '392047261520', '393663394648', '394326658915', '396547825296', '398358669351', '403053226718', '403327078303', '403398797571', '403878382125', '404335588036', '405743288520', '406236698787', '406850891275', '407512270482', '407922971056', '410871088857', '410993274172', '413145654121', '413418342606', '413990510913', '414344255365', '414665010361', '414878987508', '415076259107', '415096990548', '415134345932', '415898374611', '416380726237', '416842063547', '417276993801', '418363115654', '418978190712', '419451308158', '420525251589', '421953516327', '422660281210', '422716773809', '423930725445', '425607349574', '426325367657', '426743316697', '426909158075', '427590496635', '428677740062', '429242996634', '431231529702', '431768475246', '432089771844', '433112528293', '434331732926', '434820176994', '436310111307', '436950664991', '438050491659', '438720243151', '440737389801', '440856099504', '441010300410', '441012919436', '441340786764', '441977708642', '442279558925', '442888752927', '443249815747', '444895787476', '445066270436', '445452159007', '445544608757', '446327656647', '446334086509', '447099417955', '447344437245', '447718312100', '448096295602', '448362402750', '448437791421', '449510142326', '450542789652', '451268808790', '453914492133', '454243075918', '454708018951', '454878478860', '455295949501', '455771646044', '456012650670', '456082790337', '456893043309', '457026597750', '457633433589', '458178772538', '458451669361', '458599620270', '458712402796', '459741916297', '460579496107', '461075792974', '461639330540', '462532790342', '463479487416', '463680094014', '463748631505', '465638941851', '467325410942', '470306404467', '470892628663', '471101500246', '471764684541', '471936325112', '472167960556', '472274473037', '473144355006', '474303134594', '474547691682', '475312077285', '475679390152', '475704409833', '476144844511', '477503752563', '477760339841', '479008266961', '479058449553', '479201903159', '480022745122', '480207372766', '480208011542', '480475536219', '483069708168', '483078657402', '483555791534', '484163937975', '485171739360', '486238640158', '486318442788', '486528187221', '486653307662', '486724093467', '487075835117', '487825727312', '488283147041', '488502120284', '488730987161', '489537064020', '490290534637', '490723148876', '491271408797', '491642886103', '492182230036', '492634300165', '492809362056', '492906408022', '493857221609', '494101600798', '494419598111', '495648771014', '497315343129', '498306521888', '498536333823', '498711219760', '500482683615', '500980881293', '502327753374', '503351113243', '503741562442', '504206122956', '504762789322', '505828008366', '506633848063', '506696121897', '507297579984', '507458175641', '511898103173', '512171866621', '512296068723', '512382550508', '512436193157', '512593562953', '513764142827', '514036822646', '514641961978', '515808197273', '517564158377', '518664229968', '519581326750', '522328802007', '522610480753', '523544263548', '524275599438', '524658522586', '524725421296', '525103131843', '526002620700', '526450483356', '526529371479', '526851179788', '527490275647', '529102001334', '529277024000', '531268065622', '531329070867', '531569583280', '531659201158', '531885649505', '534534201911', '534846924117', '534939542028', '535108280501', '535700223023', '536825901754', '537669063398', '537671715099', '537849064119', '537943953334', '538115423334', '539864271063', '541061827746', '542083475947', '542172426746', '542734845675', '543315275728', '544133436431', '544510848374', '544536937525', '544734843935', '545240029200', '545994597414', '546138792195', '547043556405', '547509476297', '547578050363', '547934214792', '548659743877', '550812425828', '551019182106', '551757942034', '552291554780', '552903841159', '553514781467', '554385559122', '556525398552', '558610732496', '559651534567', '560024577736', '561320747596', '561546907559', '564066165061', '564649271459', '565115787062', '565133027204', '567664804420', '568396353920', '568533325689', '570696484411', '571208668853', '572027877266', '572336037108', '572961371493', '573292697388', '573313377656', '573859772831', '576524227815', '578066212474', '578316533664', '578660118980', '581047588350', '581130070053', '581328788659', '581775067842', '581985365737', '583106373569', '583190184710', '583303984486', '583345763035', '583608845417', '584630848692', '584974054351', '586329412044', '586482989402', '586805444392', '588159678948', '588259907349', '588607312191', '588773227252', '588809974228', '589327882798', '589727330895', '589892584776', '590032489899', '590356385337', '590506744596', '592564898892', '593780602920', '593991997540', '594239924270', '594796041417', '595353211380', '596414525875', '596896391463', '597373045724', '598559248149', '598578472115', '598716746971', '598783347326', '600636239655', '602644925125', '604318285507', '604723096881', '606511248153', '607575999558', '608959785178', '609468539593', '609608201737', '610215273417', '610387278800', '611120875422', '614092787540', '615475426900', '615839074094', '615864598532', '615986332594', '616194183804', '616614454022', '617366402669', '618143286284', '619153066025', '619245725626', '621401273440', '621848218806', '622372203913', '623701539211', '623899618944', '623940634196', '624925071076', '625001204664', '625125552135', '625402342851', '626107101308', '626553292796', '627069371650', '628157037413', '628943449194', '629026071019', '629644073792', '630020482417', '630958910218', '632338919882', '632954654882', '634691564906', '637376217363', '637454724728', '637839024236', '637928158385', '638334832935', '639369182738', '639536544457', '639729112487', '640234854210', '640259690623', '641441075861', '641938356393', '642675037886', '643596949585', '644036824268', '644141345393', '644784173767', '644849997084', '644969574939', '646717608788', '647087774510', '647885244255', '648585965371', '650018817833', '650644976608', '651509282451', '652125692913', '652394602738', '652437243948', '652993727289', '653060988075', '654075011442', '654174153064', '654434512251', '654517000290', '655641897893', '656256919338', '656483505048', '657107451157', '658062093174', '659576002760', '659789841918', '661045825192', '661881507687', '663321287581', '664233298234', '666800176692', '667027443162', '667723493591', '667900912763', '668314542521', '670161819556', '670404736065', '671629789513', '675248765338', '675771396901', '678835796394', '679245664611', '679578327390', '680069428997', '680660665682', '681044841647', '681356837841', '681628658461', '682654310819', '683133891344', '684560741695', '684757896148', '685107354181', '686387901187', '688405658638', '689468357458', '690318954491', '694345650500', '694674735136', '695382570272', '695506696950', '695614360589', '695927472067', '696141481732', '696344538400', '696360193991', '697312559295', '697642837822', '698551361437', '699486026393', '699736171795', '699789982634', '700777391571', '701438804970', '701970095679', '703183665370', '703829513590', '704025107641', '705494424374', '705854351142', '707216201787', '707230609171', '707763339947', '707967128746', '708349364971', '710465056204', '713018753281', '713962568261', '714967854867', '716751674484', '720257876889', '720745720430', '721294810136', '721462236188', '723175378301', '724311065114', '725260281549', '726012408397', '726062910967', '726750907807', '727600915853', '729269827574', '729469735401', '730031397153', '731082648661', '733734154655', '735624102611', '735837376890', '736530749364', '736819047443', '737192344670', '738731325101', '738852525135', '740123642997', '740153750092', '740166810819', '741012202819', '741732532483', '743027775537', '743862158735', '744946510523', '745576527420', '746177754969', '746805879165', '747229060202', '747329189279', '747334583968', '747424048911', '747602229289', '748031854237', '748278030583', '749876805881', '750369583115', '750802954404', '752327656508', '753874989666', '753880925017', '754775705348', '755090050495', '755216844945', '756991211222', '757031139434', '758360774458', '758843481727', '759209259634', '762594668164', '763269149154', '763639998517', '764177646170', '764490523317', '766245584008', '766533495919', '766554992149', '767507899515', '767813017116', '767848537480', '770600390810', '770817124419', '771557547238', '771615742381', '771645514271', '773467821022', '773484745606', '775595962189', '776440666163', '776522867075', '777601016429', '778075340046', '778558837889', '779529038689', '779558345599', '781297070656', '782998046294', '784441283169', '785786920527', '789476507634', '790481191661', '791968533226', '792126690708', '794501456704', '794569136533', '796402573673', '797231615719', '798052893433', '798485367562', '801074580788', '801334471752', '802825831136', '804269864808', '806542883926', '807394297581', '808458935611', '809221857512', '809498814746', '809887086749', '810358558433', '810686302806', '812519411640', '812790839692', '813055922763', '813558814731', '814566149193', '814582850999', '816919933351', '817888888251', '817952536107', '819300110664', '819418488580', '819716157095', '821085792879', '821346124874', '821372114949', '821816138949', '822274905021', '822566085433', '822955369949', '823523588426', '824349500053', '824943661652', '826429160863', '826919501912', '827687231818', '828059665819', '828290435030', '828471328199', '828707026805', '829331138793', '830458620727', '830960023363', '831213597264', '831323883328', '831428250184', '831811033311', '833267802779', '834357290412', '834501561769', '834806916487', '835173813168', '836073812853', '836346912461', '836686594237', '836809512233', '837289689095', '837626969847', '837703541926', '837956138358', '838910886206', '840214933274', '840365407675', '841003316687', '841168632450', '841764116136', '843504708241', '844577878525', '844677627151', '844702767392', '845115198660', '845663238523', '847198424597', '848219600572', '848793886259', '849045193397', '849379670310', '850702558675', '850795100153', '852472875055', '853713850743', '854481054617', '857578946837', '858141963590', '858526972147', '858857573117', '860178592278', '860680429784', '862094248680', '863953187386', '864080733145', '865408716471', '865654561421', '866203773640', '867168973137', '867327676348', '867752219685', '868521031821', '868874314265', '869064364163', '869371544050', '870031545560', '870562742571', '871102085030', '871472515762', '872800060950', '874896561479', '875314453109', '875752141512', '875895939522', '877180364429', '877574383758', '878348541116', '879286378355', '880642969338', '881051476853', '881451740496', '882722950271', '883454950117', '883606944081', '883784239155', '884435246449', '884481293184', '886409264347', '886659585567', '887119163707', '887126066436', '889531594868', '890591524509', '891047194812', '892095653963', '893007952058', '893649511657', '893725561486', '894369697281', '895081226716', '895796344347', '896284331846', '896914459532', '897265563406', '898090075122', '899956852123', '901022633716', '902077756344', '903186577214', '904168909248', '904898881486', '905676609443', '910380970013', '910997097838', '912103569023', '912681226107', '912751028336', '913412532161', '913593522947', '914462607210', '914628935504', '915416998552', '915814505567', '916396089121', '917303654463', '918013418617', '918223892161', '918701091807', '918704104394', '919235719473', '919367393195', '920037858723', '920078571165', '920815465273', '921084973823', '922274129305', '923052746422', '923931686599', '924069284639', '924294269964', '925017383941', '926372856456', '927358699734', '927753750267', '928599649920', '929126023852', '929708144712', '929862073750', '930328442017', '931334696467', '932207706514', '933831760677', '936096392668', '936117866411', '936363923026', '936639518272', '936680646698', '937511639188', '937767124938', '937804906182', '938247196952', '940473554828', '941819694106', '942973508166', '943160461060', '943654527978', '944713022036', '947872759685', '948310893572', '948948595631', '951802633484', '952135726105', '954539648113', '954695687190', '954738915509', '954742235303', '954983321707', '955038580167', '955116103754', '955621810722', '957132350705', '957304465555', '957306922246', '957518831560', '957734782020', '957966502020', '958991296092', '959629872806', '960162230649', '960956162241', '961527323227', '961541317843', '961972215724', '963879624041', '964131740148', '964426436180', '964462942334', '965266634360', '965817487737', '966309006358', '967480132046', '967972633959', '968301499415', '969413062254', '970120047911', '970156530737', '971295728977', '971552591868', '971557445175', '971836208243', '972656844555', '973653897100', '974255779466', '974936500263', '975204971987', '978553898267', '979212527518', '982935446751', '984281569836', '984625537202', '985410213753', '986038433815', '987142698092', '987148916101', '989150551987', '990148784831', '990223392234', '991022102975', '991428470003', '992200855368', '992908013717', '993932001969', '994324121608', '994837143917', '994920118214', '996010827026', '996637069404', '997205032839', '997667771749', '998732324712', '999355684375', '999954735781']

# l0_not_in_l1 = []

# for x in l0_accounts:
#     if x not in l1_accounts:
#         l0_not_in_l1.append(x)

# print(l0_not_in_l1)

# l1_not_in_l0 = [ '007707791333', '009348927679', '019347635701', '021000367750', '021365338423', '022169245411', '028533194311', '034322051670', '035483035615', '041479117925', '051994682883', '056630023496', '056863834668', '067350284291', '069580900393', '075458757601', '084970654249', '085773352584', '092327206231', '094299405254', '094827805839', '095139704753', '097475337748', '097795456520', '098000758385', '101796278814', '104590855980', '110613030397', '113176856871', '121964457821', '122578856689', '123155530402', '125237375797', '126523394397', '126667425912', '128731681412', '130149840165', '131610679465', '134679210625', '135285569980', '137885247948', '143247161605', '143711193266', '144475990439', '149089727609', '149148767097', '153772258784', '160327063738', '165159114599', '166101436673', '167187402137', '167226961519', '168441963426', '169515270762', '170394039252', '173538766175', '174295623751', '174961374134', '176996053919', '181613734321', '185671863613', '187879053795', '190367388287', '190642798438', '192614291797', '194695465174', '194928253959', '195566443221', '198271522709', '199099526724', '199360944383', '203158740621', '204053091687', '204630464347', '209441755774', '210632014336', '211508935776', '211695053472', '212271771810', '220730334052', '222001666987', '222020306244', '222624872065', '225805886800', '229679863806', '232448574943', '234772266870', '236248596881', '241396937347', '242199234071', '242753836132', '246704155817', '248014647110', '249285578332', '251641150286', '252077205715', '253933647263', '255798633318', '263648093447', '264619089984', '264629864623', '266010434788', '267316576554', '272851264686', '275309946303', '277615250575', '277922364851', '282053871296', '286814544202', '291704570552', '292767457610', '292995354700', '294658461048', '295998058776', '296351982715', '297947569761', '301605300944', '303364506186', '305890512688', '307036072367', '315724403350', '316021091064', '318783272265', '319338734333', '321307503701', '322601554261', '323232363111', '323817093338', '326755649394', '327639925109', '328061036419', '329106378841', '336512527178', '342163036816', '343248351138', '344653063612', '347229344523', '349091595040', '351943365646', '354173411065', '354366284558', '358132790115', '358508000257', '359695216436', '362629117613', '363220349094', '365379373575', '365522681417', '365664458300', '365753072011', '373510380303', '376460358145', '376839262288', '380737479390', '380769935574', '385570000518', '388634006099', '390217200304', '392479946920', '393624352404', '395562096909', '397709778342', '399553586947', '407294690266', '413582460477', '415381292284', '420191660637', '424467068503', '427183723235', '431903633673', '435453563095', '437813383155', '441189720969', '442073219544', '442716992213', '445421740248', '446521373869', '447033644314', '447579392870', '452055674543', '457326670944', '468323486041', '471501708651', '473012084470', '475182315323', '477730144921', '478518164695', '478577616413', '478925142388', '479017279804', '479521414473', '481315189088', '482815256356', '485042675174', '490455086580', '496900188711', '497647849039', '497934097495', '500008131718', '501765712544', '504857848042', '509386356242', '511051334879', '511507869156', '512889038796', '520846418629', '525174722646', '525729546547', '527319450606', '536017423151', '539278275488', '540791212970', '540957243980', '541144414411', '541833898623', '542562960121', '545298271467', '547250268408', '547500692192', '550446022751', '550882646259', '551587375342', '551799593302', '552668850488', '552923642514', '555869646217', '557955772651', '561971607098', '567514599659', '569594090084', '572023949181', '577914934418', '579155750082', '583989446449', '588949991295', '593190477397', '593391513284', '599187033811', '599569244858', '600142321150', '600424674755', '600555062417', '601290880738', '602706386284', '604033226315', '604707100243', '608583646195', '611316385221', '616864236654', '619171252003', '619768731561', '628625382433', '630697057306', '635675837571', '636712211284', '638365212721', '642264126148', '645182689106', '646023701073', '650149263190', '651491549966', '652254402111', '652943102365', '654872607431', '655572130710', '667485214512', '667662266003', '674058067202', '674583080303', '682033880221', '690966616852', '692721260310', '693542608313', '697872195742', '698031681033', '698776863250', '700210323933', '701275957923', '701862677052', '703674870110', '703906235327', '705043942076', '705182684277', '705566307488', '705967963585', '707012202561', '708437838039', '717502159111', '718832137337', '719283693773', '720366697201', '720724015417', '721284546788', '721369432361', '721523660704', '723788595012', '724343007644', '726327367093', '727456174053', '728061972801', '728483363736', '730275874014', '731923714466', '733804268833', '734998425718', '736713133364', '737141037132', '739672651720', '740053198814', '740123709034', '741274041958', '745686942713', '746080954153', '749115363610', '750227276023', '750808625910', '751142013127', '753455713898', '753904791510', '755312818847', '756414919982', '758763400630', '762478760418', '763186303361', '767857357979', '768955793546', '769986034361', '770381447961', '773334544016', '775762748442', '776173975841', '780113007310', '785183849335', '785817577328', '786289358306', '788093788772', '788247499336', '790874520702', '792374790622', '795085004402', '796591184867', '798831350617', '800436161211', '803115734520', '803466671685', '803611162529', '807269076110', '808028441788', '812263245315', '814255944148', '816813650017', '828170655941', '829296092291', '831795865367', '839326706163', '843058968294', '843385628534', '843968153990', '846230196886', '847293893566', '855484158442', '855777208844', '856019220660', '856464429809', '858312341223', '858633904165', '859828415022', '868978391936', '872335144391', '872425586473', '879472252061', '882194302238', '885718323853', '885924542129', '886015701061', '893441358747', '895114609877', '900397174018', '901519578274', '903662551929', '906924434830', '911444749492', '911593296289', '914550290266', '915366716420', '915431102062', '916086415886', '916378955819', '918824989007', '921134338319', '921400275636', '929888664800', '934472131788', '935855703025', '942547771823', '943200330016', '943942458547', '947692952239', '948572612712', '949050008497', '953636518551', '956058981677', '956885745546', '960737463278', '962383023396', '962612127814', '962639478670', '963363688713', '963365592956', '970224627637', '975817023078', '978671209663', '981805318608', '982693706514', '983350415204', '983728535136', '985676819172', '986012759135', '987825197801', '988601501448', '993130990278', '993836073940', '994090975690', '003788644068', '004420648591', '022791440266', '033065397244', '034526171824', '034536176640', '036093542835', '040046960803', '043330710293', '047209783566', '047749980333', '052819490290', '053082227562', '053444045268', '056412096258', '062387122580', '065911132104', '067447795879', '073208257581', '073546915641', '078769231505', '084367452833', '086102618356', '086301938642', '093916721094', '102145276445', '102302903403', '102771375180', '116222102902', '120311362244', '124841701563', '127513335444', '128264176259', '128346779966', '131750173436', '133277734215', '142803759194', '143312840871', '148392725318', '154843675742', '157812473613', '161775556864', '166449560162', '169912408263', '173373426076', '174145408811', '174278580167', '175472836528', '176443733777', '176539146345', '181864525870', '188831284171', '195616859884', '202120059134', '204385000248', '204485639007', '206793667976', '232144315514', '232930013947', '247012361096', '254313626187', '255311346772', '276287038763', '283711851771', '305213819503', '305317527087', '307670020531', '308350474287', '311642618236', '320383732799', '321026552774', '329503450035', '333186577928', '334144298543', '334352873287', '340021236041', '340179473373', '345595150906', '355214048684', '371132510028', '371209647698', '373337982383', '381173206319', '388830625952', '389222194029', '420924285556', '424023865363', '425844425297', '426187456338', '426843455707', '433860551753', '440708338928', '443482151530', '446543602545', '449696838726', '456716556996', '473250842200', '481383093164', '482709997193', '484797066964', '486687906253', '488791035777', '497228614851', '510920009915', '516386627390', '521756226937', '523122162952', '523471060002', '523890020650', '524666392392', '525684156106', '540404953683', '540656245819', '542906167192', '543492705153', '543691652517', '544206404384', '545560008065', '552923642514', '553526484992', '553697376517', '556116077166', '556759822564', '558067152068', '560596504822', '563511753176', '565010824322', '570051990040', '580375422572', '580411221872', '585514194047', '589202808364', '592578543696', '595716108428', '597442721103', '601947996617', '607851404322', '611480974551', '613809103576', '614751792119', '618222675128', '621577900889', '622913598307', '624603845543', '625466519857', '630033476802', '632621214384', '635786325450', '643987766248', '645286070021', '645683819147', '648080342736', '651409190172', '659069636583', '662929856130', '670111638441', '670999639077', '682655211644', '683173620724', '683340563767', '688120237422', '702011104433', '702064102213', '706151101526', '710230637982', '717090925392', '717151169743', '723606622699', '728610086968', '744048656618', '746203666711', '749257892087', '750378041276', '751720771238', '763232339158', '764230540928', '769553063621', '769985493362', '773559476152', '775302474189', '777973684917', '779420033654', '791422305768', '791959462774', '792645976151', '796155319658', '796937856387', '799867849121', '805039658749', '805522167547', '806037382723', '810613220646', '816348406744', '827001787104', '829955684988', '831587722162', '841208569552', '850791908078', '851787832323', '861990178126', '873697805464', '875635120922', '878153322966', '898483634510', '900406205346', '904005903356', '915503999640', '924473885417', '928604188985', '932346075053', '942265044478', '943971716662', '949955964069', '957627549507', '960841581879', '965416789171', '967227327126', '971424449433', '973896107202', '982104967400', '982190825179', '989504077972', '993001063892', '997715449024', '998663967865']

# l0_not_in_l1 = ['010373844257', '013724545051', '022663055756', '025502614753', '027804237073', '035657162383', '036611309112', '047681963590', '050111803238', '057239939437', '062637643455', '062790058996', '066569650784', '077365842699', '081577392665', '084707884338', '089402802109', '094552460557', '099189045841', '101470261478', '114724603768', '114858314951', '116452690986', '116909240956', '119373311443', '119529477927', '120379190565', '121652837644', '122616578453', '122674790828', '123686375669', '127095091672', '131372407359', '139914741522', '141578125655', '151577017449', '152692632441', '153125681988', '158830002625', '160133382852', '166223901165', '172296105687', '178798771574', '191346140709', '203622632891', '217463887104', '224286928353', '236244464571', '240210116683', '247015892866', '252702702909', '253026504241', '253034832134', '253123616265', '257173906640', '294454434120', '300825029455', '301923971814', '304084972275', '304576424523', '304844249827', '305132896877', '314297002272', '316658889291', '322885182098', '328741113609', '330248483315', '334873317999', '336385788515', '341813341134', '342451229934', '345693200111', '349082719353', '350045283196', '357362362136', '359797200114', '364980597341', '365322098407', '372886211720', '375127515215', '376072711816', '385203934575', '389956108244', '403878382125', '405743288520', '407512270482', '414344255365', '415898374611', '418978190712', '427590496635', '441010300410', '445452159007', '446327656647', '448096295602', '448437791421', '453914492133', '456893043309', '458451669361', '474303134594', '476144844511', '477503752563', '479058449553', '479201903159', '483555791534', '485171739360', '487075835117', '488730987161', '512296068723', '512436193157', '519581326750', '523544263548', '524725421296', '526002620700', '535108280501', '539864271063', '554385559122', '565115787062', '572336037108', '588159678948', '588259907349', '590032489899', '590356385337', '593991997540', '598559248149', '600636239655', '610215273417', '610387278800', '615839074094', '615986332594', '619245725626', '621401273440', '625402342851', '629644073792', '630020482417', '632338919882', '639729112487', '642675037886', '652394602738', '654434512251', '655641897893', '656256919338', '668314542521', '695614360589', '695927472067', '707216201787', '707230609171', '707967128746', '708349364971', '713962568261', '726062910967', '733734154655', '747334583968', '747424048911', '748031854237', '749876805881', '752327656508', '754775705348', '756991211222', '771615742381', '773484745606', '790481191661', '806542883926', '809221857512', '812519411640', '817888888251', '821085792879', '833267802779', '837703541926', '843504708241', '867752219685', '869064364163', '870031545560', '871102085030', '871472515762', '875895939522', '880642969338', '883606944081', '884481293184', '886659585567', '887126066436', '894369697281', '912751028336', '915416998552', '921084973823', '927358699734', '929862073750', '937511639188', '937804906182', '940473554828', '943160461060', '954539648113', '954983321707', '957306922246', '960162230649', '961527323227', '964462942334', '967480132046', '987142698092', '994324121608', '994837143917']



# import openai
# openai.api_key = "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"
# audio_file= open("noise.opus", "rb")
# denoised_audio = openai.Audio.denoise(audio_file)
# # transcript = openai.Audio.translate("whisper-1", audio_file)
# print(transcript)





ITSLayer1-CENTRIFY-PLTFRM-PRD - 69



'ITS': ['163564840054', '173479391558', '174188575348', '201547482655', '239459044292', 
'318724812262', '366987310429', '367544918569', '371132510028', '392047261520', 
'426325367657', '441977708642', '475679390152', '487825727312', '490723148876', 
'492809362056', '538115423334', '545994597414', '552923642514', '568396353920', 
'680069428997', '726012408397', '746805879165', '778558837889', '903186577214', 
'920037858723'],
 
{'us-csd-consulting': ['039233901334', '087127721683', '163918101077', '266527211573', 
'389613005727', '463748631505', '637376217363', '685107354181', '779558345599', 
'947872759685', '951802633484', '961541317843'], 
 'us-das': ['186574298919', '188831284171', '204193633058', '583106373569', 
            '586329412044', '702011104433', '791422305768', '914628935504', 
            '936680646698'], 
 'us-csd-advisory': ['374125395891', '471764684541', '511898103173', '529102001334', 
                    '543492705153', '558067152068', '572027877266', '607851404322', 
                    '829955684988', '923931686599'],
'DCS-SYS': ['420924285556', '580375422572'], 
'us-popena': ['764230540928'], 
'us-csd-tax': ['936363923026', '991428470003']}


371132510028

ITSLayer1-CENTRIFY-PLTFRM - 393


{'ITS': ['002470666470', '017848332886', '021365338423', '033014299223', 
'051469546478', '092944838799', '152898684328', '213911605817', '235513223743', 
'270950599001', '329783930457', '345889295625', '347913913427', '365379373575', 
'369770340971', '377602907415', '386423219226', '407922971056', '418363115654', 
'446521373869', '454708018951', '465638941851', '522328802007', '616614454022', 
'634691564906', '667900912763', '716751674484', '731923714466', '758843481727', 
'802825831136', '822274905021', '870562742571', '878012893259', '906880862652', 
'926372856456', '954695687190', '971295728977', '971552591868', '971557445175', 
'985410213753'], 


'us-csd-advisory': ['084407457870', '121964457821', '190367388287', '388634006099', 
'486653307662', '507458175641', '509386356242', '698031681033', '701862677052', 
'713018753281', '794501456704', '839326706163', '848793886259', '929888664800', 
'956058981677', '982693706514'], 

us-csd-advisory failed: 794501456704
its failed:             878012893259

'us-csd-consulting': ['015257350768', '017432556837', '018060595034', '047685407929', 
'054273887971', '055064962277', '091284928336', '111535522048', '136473818407', 
'144627418504', '194928253959', '268522796693', '271937290370', '287248264930', 
'288952646112', '298561899742', '320199641064', '324789170316', '362629117613', 
'370020434536', '386951795368', '419451308158', '432089771844', '448362402750', 
'450542789652', '457026597750', '470306404467', '479008266961', '531569583280', 
'540791212970', '590506744596', '664233298234', '666800176692', '700777391571', 
'729469735401', '801334471752', '834357290412', '850795100153', '868874314265', 
'897265563406', '899956852123', '918824989007', '919367393195'],

'us-das': ['037355734743', '211695053472', '321267289192', '366779305333', 
'373943505089', '583345763035', '650149263190', '694345650500', '785817577328',
 '858526972147', '916396089121'],

'DCS-SYS': ['210240447631', '354379781696', '553697376517', '656553435578', 
'699432268424', '947692952239'], 

'us-csd-tax': ['130220492427', '675771396901', '845115198660'], 
'us-cbcap': ['212271771810', '616864236654']}
# ==================================================================================


ITSLayer1-SPOKE-IAM-CLDTRL-PRD ==> 538

# Deleted accounts - 179
# ['003788644068', '004420648591', '022791440266', '033065397244', '034526171824', '034536176640', '036093542835', '040046960803', '043330710293', '047209783566', '047749980333', '052819490290', '053444045268', '056412096258', '062387122580', '067447795879', '069962244880', '073208257581', '073546915641', '078769231505', '084367452833', '086102618356', '086301938642', '102145276445', '102302903403', '102771375180', '116222102902', '120311362244', '124841701563', '127513335444', '128264176259', '128346779966', '131750173436', '133277734215', '142803759194', '143312840871', '148392725318', '154843675742', '169912408263', '173373426076', '174145408811', '174278580167', '175472836528', '176443733777', '176539146345', '181864525870', '195616859884', '202120059134', '204385000248', '206793667976', '232930013947', '247012361096', '254313626187', '255311346772', '276287038763', '283711851771', '305213819503', '305317527087', '307670020531', '308350474287', '311642618236', '320383732799', '321026552774', '329503450035', '333186577928', '334144298543', '334352873287', '340021236041', '340179473373', '355214048684', '371209647698', '373337982383', '388830625952', '389222194029', '424023865363', '425844425297', '426187456338', '426843455707', '433860551753', '443482151530', '446543602545', '449696838726', '456716556996', '471101500246', '473250842200', '482709997193', '484797066964', '486687906253', '488791035777', '497228614851', '510920009915', '521756226937', '523122162952', '523890020650', '524666392392', '525684156106', '537943953334', '540404953683', '542906167192', '544206404384', '553526484992', '556116077166', '560596504822', '563511753176', '565010824322', '580411221872', '585514194047', '589202808364', '592578543696', '595716108428', '596414525875', '601947996617', '613809103576', '614751792119', '621577900889', '624603845543', '630033476802', '632621214384', '635786325450', '643987766248', '645286070021', '645683819147', '648080342736', '651409190172', '659069636583', '670111638441', '670999639077', '682655211644', '683173620724', '683340563767', '702064102213', '706151101526', '710230637982', '717151169743', '723606622699', '728610086968', '744048656618', '746203666711', '750378041276', '751720771238', '763232339158', '773559476152', '775302474189', '777973684917', '779420033654', '792645976151', '796155319658', '796937856387', '799867849121', '805039658749', '805522167547', '806037382723', '810613220646', '816348406744', '827001787104', '841208569552', '850791908078', '851787832323', '861990178126', '873697805464', '875635120922', '878153322966', '900406205346', '904005903356', '915503999640', '922274129305', '928604188985', '942265044478', '943971716662', '949955964069', '957627549507', '965416789171', '967227327126', '971424449433', '973896107202', '982104967400', '982190825179', '993001063892', '998663967865']

Active accounts - 354

[ '012046266255', '016101336017', '018472551420', '021191302697', '024517926231', '024838550089', '025625448852', '025648210413', '028002051339', '030263312215', '031043502080', '031573843522', '035784263114', '038144103111', '038270351176', '039233901334', '039929384510', '041081560071', '050877488444', '053392642336', '055182143107', '055266516288', '059272759707', '060465202534', '067555441682', '067804797237', '068183022879', '068315429392', '070717231209', '072038240142', '084383639626', '087127721683', '088258796410', '092606503522', '104624496682', '107393473960', '109577287868', '119753528152', '122600361133', '130203272373', '135381627630', '139100456334', '144544136090', '155305736938', '156327752252', '158347257464', '158684596895', '161395754509', '162310846309', '162435009311', '163564840054', '163918101077', '166556434326', '168131472534', '170178091998', '170302206609', '172383415443', '173316567229', '173479391558', '174188575348', '175742142056', '181395753476', '185075825010', '186574298919', '194751376404', '198162557634', '201547482655', '201944482039', '204193633058', '205394135495', '208073657936', '208626548723', '211070001969', '212149607166', '212773838067', '217413253555', '218167183912', '220288712633', '222023357012', '222762603060', '228052005984', '232698387549', '233632481017', '234019974018', '234848667271', '239459044292', '240344030626', '241671150056', '241693290537', '241696072805', '251058218532', '254471163821', '257915333181', '258860010574', '262259241401', '266527211573', '268527685757', '277946231630', '281981188021', '282181226940', '282623329273', '284383242428', '290092688251', '294297099413', '298370672248', '309245159336', '312249472578', '318724812262', '318902570495', '320947721229', '327370950832', '328754255717', '332205887294', '336222303362', '336827305173', '340860049954', '345414987291', '354362986338', '363191287345', '366986788235', '366987310429', '367544918569', '371132510028', '372988589375', '373124920883', '374125395891', '376587057974', '385159429807', '389613005727', '392047261520', '395672011172', '398358669351', '403327078303', '404335588036', '410871088857', '414878987508', '415076259107', '415096990548', '416380726237', '417276993801', '422660281210', '422716773809', '426325367657', '426909158075', '431768475246', '433112528293', '434820176994', '436950664991', '441977708642', '442279558925', '445544608757', '447099417955', '447333052098', '454243075918', '454878478860', '455295949501', '456012650670', '457633433589', '458712402796', '459741916297', '460579496107', '463748631505', '470892628663', '471764684541', '475312077285', '475679390152', '475704409833', '480207372766', '486318442788', '487825727312', '488502120284', '490723148876', '492182230036', '492809362056', '494419598111', '498306521888', '504206122956', '511898103173', '512593562953', '514641961978', '522610480753', '526851179788', '529102001334', '531885649505', '534939542028', '536825901754', '537669063398', '538115423334', '541061827746', '542083475947', '542734845675', '544536937525', '545240029200', '545994597414', '547578050363', '551757942034', '564649271459', '568396353920', '571208668853', '572027877266', '573859772831', '581130070053', '581328788659', '583106373569', '586329412044', '588607312191', '589327882798', '593780602920', '594796041417', '597373045724', '598578472115', '598783347326', '608959785178', '609608201737', '615475426900', '615864598532', '617366402669', '618143286284', '621848218806', '622372203913', '623899618944', '623940634196', '624925071076', '626553292796', '628943449194', '632954654882', '637376217363', '637928158385', '639369182738', '640259690623', '644969574939', '646717608788', '650644976608', '657107451157', '659576002760', '664785693086', '670404736065', '678835796394', '680069428997', '681044841647', '681356837841', '683133891344', '684757896148', '685107354181', '686387901187', '696141481732', '698551361437', '699789982634', '701438804970', '701970095679', '703829513590', '707763339947', '710465056204', '721294810136', '721462236188', '726012408397', '727600915853', '729269827574', '731082648661', '735624102611', '736819047443', '738852525135', '740123642997', '740153750092', '741012202819', '741732532483', '743862158735', '746177754969', '747229060202', '750802954404', '753874989666', '755090050495', '757031139434', '758360774458', '762594668164', '764177646170', '776522867075', '778558837889', '779558345599', '781297070656', '782998046294', '792126690708', '792958435369', '797231615719', '798052893433', '798485367562', '804269864808', '812790839692', '813055922763', '813558814731', '814582850999', '819300110664', '819716157095', '821372114949', '826429160863', '828290435030', '831323883328', '835173813168', '836686594237', '836809512233', '840214933274', '840365407675', '841168632450', '844577878525', '844677627151', '850702558675', '854481054617', '857578946837', '867168973137', '867327676348', '872800060950', '877574383758', '879286378355', '884435246449', '886409264347', '896914459532', '898090075122', '901022633716', '903186577214', '904168909248', '905676609443', '912103569023', '914628935504', '918013418617', '918701091807', '919235719473', '920037858723', '923052746422', '923931686599', '929126023852', '933831760677', '936096392668', '936117866411', '936363923026', '936680646698', '947872759685', '951802633484', '957132350705', '957518831560', '957734782020', '957966502020', '961541317843', '963879624041', '964426436180', '967972633959', '968301499415', '969413062254', '970156530737', '982935446751', '984281569836', '984625537202', '991428470003', '999355684375']





ITSLayer1-SPOKE-IAM-CLDTRL-NPD ==> 1236

Active accounts 843
['003866754167', '003896315085', '004629961456', '005241645787', '005333532456', '008530717892', '009296111214', '009464921414', '010553816281', '010753539596', '010817157965', '012095546580', '012680939556', '014296065027', '015257350768', '016937230970', '017432556837', '017456215898', '017848332886', '018060595034', '018649475058', '022618988199', '023146164982', '026194564518', '027393114528', '028568440897', '030974280115', '031007339471', '033014299223', '033246515043', '035778782477', '035825521573', '037355734743', '037388063884', '038059618311', '041149546493', '042115301607', '043262203499', '047685407929', '047807787807', '051469546478', '051724048437', '051799833889', '052346183910', '053070086514', '054273887971', '055064962277', '055798325468', '055840983865', '056895686976', '058611342228', '062130537180', '063353051663', '063376279610', '066979683739', '067154635066', '068757897087', '068900102323', '070036689193', '070660504732', '072706165844', '073146838139', '077240679103', '080656668290', '084407457870', '084582856213', '086573493719', '086574853384', '087549363555', '088220075168', '088475525986', '088563067471', '089587160124', '090446900710', '091115685892', '091284928336', '092944838799', '092960230608', '093441740690', '095611350051', '097020054916', '097107211992', '098556162587', '099325924796', '101177931550', '103896790399', '103917580387', '105723104798', '106456323084', '107282117866', '107831069545', '109640153986', '109997849406', '110557118724', '110633459663', '110912366499', '111535522048', '112558980097', '114535277687', '114727038176', '115154971638', '117523786847', '117725889170', '118974102263', '119203704260', '120152586068', '122994942344', '123906516738', '124524166810', '129330666742', '130220492427', '131660845369', '132027580256', '133143185602', '133550674684', '133762489993', '135185876313', '135755270796', '136473818407', '138233832997', '142332166808', '142646231261', '144627418504', '144635163526', '146003574029', '148094740375', '149477848214', '149878268794', '150316879294', '150767860677', '151434629482', '152223477882', '152898684328', '153592066989', '153685446838', '153714429732', '153951089943', '159602377297', '159861609278', '159921825654', '160339976030', '160934439473', '161391293302', '161394336152', '161636799113', '162022282563', '163564840054', '164398058260', '167393382127', '168849861669', '170140899813', '171762472359', '172067301824', '172832121763', '172863928423', '175091774301', '175924461164', '175930311790', '175956104120', '178790212740', '184405994032', '185020060020', '185284577515', '186861417013', '187528728268', '188529011465', '193361093508', '193412282603', '193854970547', '197821584623', '198092089923', '199570988326', '200228513821', '200543695574', '201823333330', '205063679632', '205707500197', '208421051549', '210264310230', '210579071845', '210798437721', '211101344176', '211593031182', '211600541199', '212921778948', '213911605817', '214979362845', '215611426819', '216542771521', '217360121513', '218515608600', '220933860874', '221292848939', '222531082059', '223441765792', '224244178360', '225658821723', '225832781273', '226603863404', '227841379693', '227937082667', '228115329917', '231716799145', '231972052922', '231974784973', '232162000477', '233127284560', '233256077443', '235513223743', '237366858765', '238557733178', '238820089267', '238970376046', '239205171373', '239537226757', '239823972708', '240044539051', '245250269795', '246197285750', '246585516170', '246698818014', '246938629297', '247048948584', '251072340318', '251985550364', '252323805874', '252365818154', '254263703118', '256489642696', '256706112543', '256867917700', '259692210398', '261934197954', '263128578697', '264944699463', '266396698525', '266955691786', '266980808703', '268522796693', '269475816966', '269734407326', '270950599001', '271937290370', '272606000076', '275380584232', '277167574796', '279462431622', '279743555714', '280522856839', '282272865295', '282752622212', '283503555640', '283520033493', '284328782567', '284760807629', '286058688319', '287248264930', '288228195021', '288952646112', '292210083552', '294745394203', '295562148805', '296916826127', '297486902890', '298561899742', '300801515163', '306182918611', '307036839104', '307133528178', '307141129713', '307492883559', '308360630103', '309416070659', '309727501936', '312807674804', '314285588528', '314448338500', '315370314208', '315952988300', '316461870015', '316625284770', '320199641064', '320510073098', '321267289192', '321376957942', '323911653333', '324789170316', '328235541546', '328705181099', '329783930457', '330374005851', '330459082734', '332607986083', '334734394521', '335893736669', '337910336443', '340826808143', '340942712854', '342122824633', '342416616976', '342472205137', '342788227825', '344161786867', '345404735839', '345889295625', '347913913427', '354716619065', '359252437790', '361804686539', '363235117348', '363727823241', '364513602188', '365111145171', '365655694970', '366087731140', '366779305333', '366862120963', '367409673491', '368276533222', '369770340971', '370020434536', '372203661552', '373758253227', '373943505089', '374367240544', '375028686807', '376614162583', '376756837763', '377602907415', '380906577769', '384921830497', '385078759766', '385397780492', '385644467038', '386372166532', '386423219226', '386880889362', '386951795368', '391931929196', '394326658915', '396547825296', '403053226718', '403398797571', '406236698787', '406850891275', '407922971056', '410993274172', '413145654121', '413418342606', '413990510913', '415134345932', '416842063547', '418363115654', '419451308158', '420525251589', '421953516327', '423930725445', '425607349574', '426743316697', '428677740062', '429242996634', '431231529702', '432089771844', '434331732926', '436310111307', '438050491659', '438720243151', '440737389801', '440856099504', '441012919436', '441340786764', '442888752927', '443249815747', '444895787476', '445066270436', '446334086509', '447344437245', '447718312100', '448309204918', '448362402750', '449510142326', '450542789652', '451268808790', '453919366994', '454708018951', '455771646044', '456082790337', '457026597750', '458178772538', '458599620270', '461075792974', '461639330540', '462532790342', '463479487416', '463680094014', '465638941851', '467325410942', '470306404467', '471936325112', '472167960556', '472274473037', '473144355006', '474547691682', '477760339841', '479008266961', '480022745122', '480208011542', '480475536219', '483069708168', '483078657402', '484163937975', '486238640158', '486528187221', '486653307662', '486724093467', '488283147041', '489537064020', '490290534637', '491271408797', '491642886103', '492634300165', '492906408022', '493857221609', '494101600798', '495693388953', '497315343129', '498536333823', '498711219760', '500482683615', '500980881293', '502327753374', '503351113243', '503741562442', '504762789322', '505828008366', '506633848063', '506696121897', '507297579984', '507458175641', '512171866621', '512382550508', '513764142827', '514036822646', '515808197273', '515993956937', '517564158377', '518664229968', '522328802007', '524275599438', '524658522586', '525103131843', '526450483356', '526529371479', '527490275647', '529005549134', '529277024000', '529976655406', '531268065622', '531329070867', '531569583280', '531659201158', '534534201911', '534846924117', '535700223023', '537671715099', '537849064119', '542172426746', '543315275728', '544133436431', '544510848374', '544734843935', '546138792195', '547043556405', '547509476297', '547934214792', '548659743877', '550812425828', '551019182106', '552291554780', '552903841159', '553514781467', '556525398552', '558610732496', '559651534567', '560024577736', '561320747596', '561546907559', '564066165061', '565133027204', '567664804420', '568533325689', '570696484411', '572961371493', '573292697388', '573313377656', '576524227815', '578066212474', '578316533664', '578660118980', '578941406071', '581047588350', '581775067842', '581985365737', '583190184710', '583303984486', '583345763035', '583608845417', '584199283902', '584630848692', '584974054351', '586482989402', '586805444392', '588773227252', '589727330895', '589892584776', '590506744596', '592564898892', '592597120121', '594239924270', '595353211380', '596896391463', '598716746971', '601615387933', '602644925125', '604318285507', '604723096881', '606511248153', '609468539593', '611120875422', '614092787540', '616194183804', '616614454022', '619153066025', '623701539211', '625001204664', '625125552135', '626107101308', '627069371650', '628157037413', '629026071019', '630958910218', '634691564906', '637454724728', '637839024236', '638334832935', '639536544457', '640234854210', '641441075861', '641938356393', '643596949585', '644036824268', '644141345393', '644784173767', '644849997084', '647087774510', '647885244255', '648585965371', '650018817833', '651509282451', '652125692913', '652437243948', '652993727289', '653060988075', '654075011442', '654174153064', '654517000290', '656483505048', '658062093174', '659789841918', '661045825192', '661881507687', '663321287581', '666800176692', '667027443162', '667723493591', '667900912763', '670161819556', '671629789513', '675248765338', '675771396901', '677167808438', '679245664611', '679578327390', '680162656405', '680660665682', '681628658461', '682654310819', '684560741695', '688405658638', '689468357458', '690318954491', '694345650500', '694674735136', '695382570272', '695506696950', '696344538400', '696360193991', '697312559295', '697642837822', '699486026393', '699736171795', '700777391571', '703183665370', '704025107641', '705494424374', '705854351142', '712231335908', '713018753281', '714967854867', '716751674484', '720257876889', '720745720430', '721880670703', '723175378301', '724311065114', '725260281549', '726750907807', '729469735401', '730031397153', '735837376890', '736530749364', '737192344670', '738731325101', '739295643645', '740166810819', '743027775537', '744946510523', '745576527420', '747329189279', '747602229289', '748278030583', '750369583115', '753880925017', '755216844945', '758843481727', '759209259634', '763269149154', '763639998517', '764490523317', '766245584008', '766533495919', '766554992149', '767507899515', '767813017116', '767848537480', '770600390810', '770817124419', '771557547238', '771645514271', '773318300432', '773467821022', '775595962189', '776440666163', '777601016429', '778075340046', '779529038689', '784441283169', '785786920527', '789476507634', '791968533226', '794501456704', '794569136533', '796402573673', '801074580788', '801334471752', '802825831136', '807394297581', '808458935611', '809498814746', '809887086749', '810358558433', '810686302806', '814566149193', '816919933351', '817952536107', '819418488580', '821346124874', '821816138949', '822274905021', '822566085433', '822955369949', '823523588426', '824349500053', '824943661652', '826919501912', '827687231818', '828059665819', '828471328199', '828707026805', '829331138793', '830458620727', '830960023363', '831213597264', '831428250184', '831811033311', '834357290412', '834501561769', '834806916487', '836073812853', '836346912461', '837289689095', '837626969847', '837956138358', '838910886206', '841003316687', '841307960404', '841764116136', '844702767392', '845115198660', '845663238523', '847198424597', '848219600572', '848793886259', '849045193397', '849379670310', '850795100153', '852472875055', '853713850743', '858141963590', '858526972147', '858857573117', '860178592278', '860680429784', '862094248680', '862140534462', '863953187386', '864080733145', '865408716471', '865654561421', '866203773640', '868521031821', '868874314265', '869371544050', '870562742571', '871834102260', '874896561479', '875314453109', '875752141512', '877180364429', '881051476853', '881451740496', '882722950271', '883454950117', '883784239155', '886674288950', '887119163707', '889531594868', '890591524509', '891047194812', '892095653963', '893007952058', '893649511657', '893725561486', '895081226716', '895796344347', '896284331846', '897265563406', '899956852123', '902077756344', '904898881486', '910380970013', '910997097838', '912681226107', '913412532161', '913593522947', '914462607210', '915814505567', '916396089121', '916943954658', '917232550581', '917303654463', '918223892161', '918704104394', '919367393195', '920078571165', '920815465273', '924069284639', '924235240336', '924294269964', '925017383941', '926372856456', '927753750267', '928599649920', '929708144712', '930328442017', '931334696467', '932207706514', '936639518272', '937767124938', '938247196952', '941819694106', '942973508166', '943654527978', '944713022036', '948310893572', '948837575101', '948948595631', '952135726105', '954695687190', '954738915509', '954742235303', '955038580167', '955116103754', '955621810722', '957304465555', '958991296092', '959629872806', '960102767444', '960956162241', '961972215724', '964131740148', '964530035547', '965266634360', '965817487737', '966309006358', '970120047911', '971295728977', '971552591868', '971557445175', '971836208243', '973653897100', '974255779466', '974936500263', '975204971987', '978553898267', '979212527518', '985410213753', '986038433815', '987148916101', '989150551987', '990148784831', '990223392234', '991022102975', '992200855368', '992908013717', '993017681120', '993932001969', '994920118214', '996010827026', '997205032839', '997667771749', '998732324712', '999954735781']


# Deleted Accounts  - 391

# ['003372875111', '007707791333', '009348927679', '019347635701', '021000367750', '022169245411', '028533194311', '034322051670', '035483035615', '041479117925', '053859713267', '056630023496', '056863834668', '067350284291', '069580900393', '075458757601', '085773352584', '092327206231', '094299405254', '094827805839', '095139704753', '097475337748', '098000758385', '101796278814', '104590855980', '110613030397', '113176856871', '122578856689', '123155530402', '125237375797', '126523394397', '126667425912', '128731681412', '130149840165', '131610679465', '134679210625', '135285569980', '137885247948', '143247161605', '143711193266', '144475990439', '149089727609', '149148767097', '153772258784', '156487489205', '160327063738', '165159114599', '166101436673', '167187402137', '167226961519', '168441963426', '169515270762', '170394039252', '173538766175', '174961374134', '176996053919', '181613734321', '185671863613', '187879053795', '190642798438', '192614291797', '194695465174', '195566443221', '198271522709', '199099526724', '199360944383', '203158740621', '204053091687', '204630464347', '209441755774', '210632014336', '211508935776', '212875955720', '220730334052', '221112619429', '222001666987', '222020306244', '222624872065', '225805886800', '229679863806', '232448574943', '234772266870', '236248596881', '240715832728', '242199234071', '242753836132', '246704155817', '248014647110', '249285578332', '251641150286', '252077205715', '253933647263', '255798633318', '263648093447', '264619089984', '264629864623', '265756937744', '266010434788', '267316576554', '272298319762', '272851264686', '275309946303', '277615250575', '277922364851', '282053871296', '291704570552', '292767457610', '292995354700', '294658461048', '295998058776', '296351982715', '297947569761', '301605300944', '303364506186', '305890512688', '307036072367', '316021091064', '318783272265', '319338734333', '321307503701', '322601554261', '323232363111', '323817093338', '326755649394', '328061036419', '329106378841', '336512527178', '342163036816', '343248351138', '344653063612', '347229344523', '349091595040', '350229878820', '351943365646', '354173411065', '354366284558', '358132790115', '358508000257', '359695216436', '365522681417', '365664458300', '365753072011', '369102312058', '370908162876', '371826770560', '373510380303', '376460358145', '376839262288', '380218571053', '380737479390', '380769935574', '385570000518', '390217200304', '392479946920', '393624352404', '393663394648', '395562096909', '397709778342', '399553586947', '407294690266', '413582460477', '414665010361', '415381292284', '420191660637', '427183723235', '431903633673', '435453563095', '441189720969', '442073219544', '442716992213', '445421740248', '447033644314', '447579392870', '452055674543', '457326670944', '468323486041', '471501708651', '473012084470', '475182315323', '477730144921', '478518164695', '478577616413', '478925142388', '479017279804', '479521414473', '481315189088', '482815256356', '485042675174', '490455086580', '495648771014', '497647849039', '497934097495', '500008131718', '501765712544', '504857848042', '511051334879', '511507869156', '512889038796', '520846418629', '525174722646', '525729546547', '527319450606', '536017423151', '539278275488', '540957243980', '541144414411', '541833898623', '542562960121', '545298271467', '547250268408', '547500692192', '550446022751', '551587375342', '551799593302', '552668850488', '555869646217', '557955772651', '561971607098', '567514599659', '572023949181', '577914934418', '579155750082', '583989446449', '588809974228', '588949991295', '593190477397', '593391513284', '599187033811', '599569244858', '600142321150', '600424674755', '600555062417', '601290880738', '602706386284', '604033226315', '604707100243', '607575999558', '608583646195', '611316385221', '619171252003', '619768731561', '628625382433', '630697057306', '635675837571', '636712211284', '638365212721', '642264126148', '645182689106', '646023701073', '651491549966', '652254402111', '652943102365', '654872607431', '655572130710', '664233298234', '667662266003', '674058067202', '674583080303', '682033880221', '690966616852', '692721260310', '693542608313', '697872195742', '698776863250', '700210323933', '701275957923', '703674870110', '703906235327', '705043942076', '705182684277', '705967963585', '707012202561', '708437838039', '717502159111', '718832137337', '719283693773', '720366697201', '720724015417', '721284546788', '721369432361', '721523660704', '723788595012', '724343007644', '726327367093', '728061972801', '728483363736', '730275874014', '733804268833', '734998425718', '736713133364', '737141037132', '739672651720', '740053198814', '740123709034', '741274041958', '745686942713', '746080954153', '749115363610', '750227276023', '750808625910', '751142013127', '755312818847', '756414919982', '758763400630', '762478760418', '763186303361', '767857357979', '768955793546', '769986034361', '770381447961', '773334544016', '775762748442', '776173975841', '780113007310', '786289358306', '788093788772', '790874520702', '792374790622', '795085004402', '796591184867', '798831350617', '800436161211', '803115734520', '803466671685', '803611162529', '807269076110', '808028441788', '812263245315', '814255944148', '829296092291', '831795865367', '843058968294', '843385628534', '843968153990', '846230196886', '847293893566', '855777208844', '856019220660', '856464429809', '858312341223', '858633904165', '859828415022', '868978391936', '872335144391', '872425586473', '878348541116', '879472252061', '882194302238', '885718323853', '885924542129', '893441358747', '895114609877', '900397174018', '901519578274', '906924434830', '914550290266', '915366716420', '915431102062', '916086415886', '916378955819', '921134338319', '921400275636', '934472131788', '935855703025', '942547771823', '943200330016', '943942458547', '948572612712', '949050008497', '953636518551', '956885745546', '960737463278', '962383023396', '962612127814', '962639478670', '963363688713', '963365592956', '970224627637', '972656844555', '975817023078', '978671209663', '981805318608', '983728535136', '985676819172', '986012759135', '987825197801', '988601501448', '993130990278', '993836073940', '994090975690', '996637069404']




