from azure.identity import AzureCliCredential, ChainedTokenCredential, ManagedIdentityCredential, EnvironmentCredential
import openai

credential = ChainedTokenCredential(ManagedIdentityCredential(), EnvironmentCredential(), AzureCliCredential())
access_token = credential.get_token("https://cognitiveservices.azure.com/.default")


openai.api_base = "https://<<your azure open ai resource name>>.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_type = "azure_ad"
openai.api_key = access_token.token

response = openai.ChatCompletion.create(

    engine="gpt-35-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
        {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
        {"role": "user", "content": "Do other Azure AI services support this too?"}
    ]
)

print(response)
