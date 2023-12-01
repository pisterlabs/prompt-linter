from azure.storage.blob import BlobServiceClient, ContainerClient
from langchain.document_loaders import AzureBlobStorageContainerLoader
import os 
# Replace these placeholders with your actual connection string and container name
connection_string = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
container_name = "test" 

# define a function to use AzureBlobStorageContainerLoader
def azureLoader(conn_str=connection_string, container=container_name):
    print("connect to azure blob storage")
    loader = AzureBlobStorageContainerLoader(conn_str, container)
    return loader.load()


