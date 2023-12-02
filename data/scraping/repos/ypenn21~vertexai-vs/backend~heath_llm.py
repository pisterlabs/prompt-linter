import vertexai
from vertexai.preview.language_models import ChatModel, InputOutputTextPair
from typing import Dict
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import HumanMessage, SystemMessage

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

def predict_llm_health(
    project_id: str,
    prompt: str,
    health_instance
    ) :
    vertexai.init(project=project_id, location="us-central1")
    
    chat_model =  ChatVertexAI()
    #chat_model = ChatModel.from_pretrained("chat-bison@001")
    parameters = {
         "temperature": 0.2,
         "max_output_tokens": 256,
         "top_p": 0.8,
         "top_k": 40
    }
    #chat = chat_model.start_chat(
    template=("""You are most recognized personal medical expert specialized in Cardinal, heart disease,diabetes and other general health areas. You provide answers and advices based on following background information provided:
Age: {age}
Gender: {gender} 
Weight:  {weight}kg
Height: {height}cm
Fasting Glucose Level: {blood_glucose_level} mg/dL
Smoke: {smoking_history}
Diabetes Risks(0-1): {diabetes_risk}
Heart Disease history: {heart_disease}
Blood Pressure(H/L): {blood_pressure_h}/{blood_pressure_l}
Alcohol: {alcohol}
Cholesterol Level: {cholesterol}
Provide answers based on information available from most recognized resources such as websites:
https://diabetes.org/diabetes
https://www.heart.org/
https://www.cdc.gov/heartdisease/index.htm
https://www.cdc.gov/diabetes/library/features/diabetes-and-heart.html#:~:text=Eat%20fewer%20processed%20foods%20(such,your%20triglycerides%20and%20blood%20sugar.
For questions outside nutrition, medical and health areas, just say: I am health AI assistant, I can not answer question outside of this domain. Try again!
"""
    )   
    #template = (
    #"You are a helpful assistant that translates {input_language} to {output_language}."
    #)
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{prompt}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
       [system_message_prompt, human_message_prompt]
    )

# get a chat completion from the formatted messages
    response=chat_model(
      chat_prompt.format_prompt(
        age=health_instance["age"],
        gender=health_instance["gender"],
        weight=health_instance["weight"],
        height=health_instance["height"],
        blood_glucose_level=health_instance["blood_glucose_level"],
        smoking_history=health_instance["smoking_history"],
        heart_disease=health_instance["heart_disease"],
        blood_pressure_h=health_instance["blood_pressure_h"],
        blood_pressure_l=health_instance["blood_pressure_l"],
        diabetes_risk=health_instance["diabetes_risk"],
        alcohol=health_instance["alcohol"],
        cholesterol=health_instance["cholesterol"],
        prompt=prompt
      ).to_messages()
    )
   
    return response

def predict_health(
    project: str,
    endpoint_id: str,
    instance_dict: Dict,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # for more info on the instance schema, please use get_model_sample.py
    # and look at the yaml found in instance_schema_uri
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/tabular_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    return predictions
    #for prediction in predictions:
    #    print(" prediction:", dict(prediction))
