import openai
import sys



file_path = sys.argv[1]
target_path = sys.argv[2]

with open(file_path, 'r') as file:
        codebase = file.read()


system_prompt = """
    You are a Caltech graduated senior cloud infrastructure engineer manager who specializes in understanding complex systems 
    and writing infrastructure code for deploying services on Microsoft Azure. 
    Your job is to take any codebase and explain it to your colleagues in such way that they exactly know what to do when it comes to infrastructure as code. 
    You are given an entire codebase and give your bullet points. 
    Make sure you quote exact names of the environment variables required: variable ('varibale name') Give your explanation in the following format:\n
    Runtime: {name of runtime}\n(if ports are needed) Port: {port number}\n 
"""
prompt = codebase

response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": codebase
            }
        ]
    )
print(response.choices[0].message.content)



system_prompt = """
    You are a cloud engineer, your role is to write sound pulumi for an azure app service in typescript based on the requirements. Use european regions, make sure the runtime is correct. Make all systems public.
    Make sure to use azure_native.web.WebApp and azure_native.web.WebAppApplicationSettings
    Use siteConfig: {
        alwaysOn: false,
        nodeVersion: <runtime>,
        linuxFxVersion: <runtime version>,
      }
    Make sure to include the required env variables into the app service configuration.
    Prefix all names with: ef
    Make sure all names are very short
    Make sure the pulumi exports the app service name as apiAppName and resource group name as apiResourceGroupName.
    If the requirements include references to storage accounts or databases, make sure to include their creation in the pulumi, including anything else they might need (e.g permissions or containers).
    Make sure there is no comments and that the output is valid pulumi typescript.
    Keep in mind this is the way you create storage connection strings:
    const storageAccountKeys = pulumi
        .all([storageAccount.name, resourceGroup.name])
        .apply(async ([accountName, resourceGroupName]) => {
        return await azure_native.storage.listStorageAccountKeys({
            accountName,
            resourceGroupName,
        });
     });
    STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=" +
      storageAccount.name +
      ";AccountKey=" +
      storageAccountKeys.keys[0].value +
      ";EndpointSuffix=core.windows.net",
    """
prompt = "Requirements: " + response.choices[0].message.content

response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )




iac = response.choices[0].message.content.replace("```typescript", "").replace("```", "")

with open(target_path, "w+") as file:
    file.write(iac)
