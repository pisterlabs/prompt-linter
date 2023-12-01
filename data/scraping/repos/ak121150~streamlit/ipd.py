import streamlit as st
from azure.storage.blob import BlobServiceClient
import http.client, urllib.request, urllib.parse, urllib.error, base64
import openai

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time

openai.api_type = "azure"
openai.api_base = "https://openai897.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key="f70495c8df474a09ba8c84db5a1132d0"
'''
Authenticate
Authenticates your credentials and creates a client.
'''
subscription_key = '6fd68fa46db64faea278fcf53c1779a1'
endpoint = "https://stdcou.cognitiveservices.azure.com/"

def upload_image_to_blob(image_file):
    # Connect to Azure Blob Storage
    blob_service_client = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=blobopenai234;AccountKey=V4QARsrZPBK6pCbNOqx0e98J1UfVQlkTMUVs9IHWCAlMSHq/PXobKpFaNBvCJfvz5NFzgDhhsC1y+AStRabrJg==;EndpointSuffix=core.windows.net")

    # Create a container (if it doesn't exist)
    container_name = "blob1234"
    container_client = blob_service_client.get_container_client(container_name)
    if not container_client.exists():
        container_client.create_container()

    # Upload the image file to the blob containerblob_client.upload_blob(image_file, overwrite=True)
    blob_name = image_file.name
    blob_client = container_client.get_blob_client(blob_name)


    # Return the URL of the uploaded image
    blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{blob_name}"
    return blob_url

# Streamlit app code
st.title("Upload and Store Image in Blob")

computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

uploaded_files = st.file_uploader("Choose multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
blob_urls = []
final=''
if uploaded_files is not None:
    for uploaded_file in uploaded_files:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Upload the image to Azure Blob Storage
        blob_url = upload_image_to_blob(uploaded_file)
        blob_urls.append(blob_url)

        # Display the blob URL
        st.markdown(f"**Blob URL:** {blob_url}")
        print(blob_url)

        print("===== Read File - remote =====")
        # Call API with URL and raw response (allows you to get the operation location)
        read_response = computervision_client.read(blob_url, raw=True)

        # Get the operation location (URL with an ID at the end) from the response
        read_operation_location = read_response.headers["Operation-Location"]
        # Grab the ID from the URL
        operation_id = read_operation_location.split("/")[-1]

        # Call the "GET" API and wait for it to retrieve the results
        while True:
            read_result = computervision_client.get_read_result(operation_id)
            if read_result.status not in ['notStarted', 'running']:
                break
            time.sleep(1)

        response_text = ""
        # Print the detected text, line by line
        if read_result.status == OperationStatusCodes.succeeded:
            for text_result in read_result.analyze_result.read_results:
                for line in text_result.lines:
                    final=final+line.text
                    # st.markdown(line.text)
        final=final+'\n \n \n \n'



st.markdown(final)


# response = openai.Completion.create(

#      engine="openai45",
#      prompt=final,
#      temperature=1,
#      max_tokens=100,
#      top_p=0.5,
#      frequency_penalty=0,
#      presence_penalty=0,
#      stop=None
#  )

# # # Print the response from OpenAI
# if response['choices'][0]['role'] == 'assistant':
#      st.markdown(response['choices'][0]['content'])
# print("End of Computer Vision quickstart.")





