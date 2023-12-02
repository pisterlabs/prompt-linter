import streamlit as st
from langchain.llms import Clarifai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

PAT = st.secrets["clarifai_pat"]

@st.cache_resource(show_spinner=False)
def moderate_input(text):

    USER_ID = 'clarifai'
    APP_ID = 'main'
    MODEL_ID = 'moderation-multilingual-text-classification'
    MODEL_VERSION_ID = '79c2248564b0465bb96265e0c239352b'

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (('authorization', 'Key ' + PAT),)

    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,  # The userDataObject is created in the overview and is required when using a PAT
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,  # This is optional. Defaults to the latest model version
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(
                            raw=text
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        print(post_model_outputs_response.status)
        raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)

    # Since we have one input, one output will exist here
    output = post_model_outputs_response.outputs[0]

    for concept in output.data.concepts:
        if concept.value > 0.7:
            return True, concept.name, concept.value
    
    return [False, None, None]

@st.cache_resource(show_spinner=False)
def query_gpt4(text_input):

    USER_ID = 'openai'
    APP_ID = 'chat-completion'
    MODEL_ID = 'GPT-4'

    # MODEL_VERSION_ID = 'ad16eda6ac054796bf9f348ab6733c72'

    # channel = ClarifaiChannel.get_grpc_channel()
    # stub = service_pb2_grpc.V2Stub(channel)

    # metadata = (('authorization', 'Key ' + PAT),)

    # userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

    # post_model_outputs_response = stub.PostModelOutputs(
    # service_pb2.PostModelOutputsRequest(
    #     user_app_id=userDataObject,  # The userDataObject is created in the overview and is required when using a PAT
    #     model_id=MODEL_ID,
    #     version_id=MODEL_VERSION_ID,  # This is optional. Defaults to the latest model version
    #     inputs=[
    #         resources_pb2.Input(
    #             data=resources_pb2.Data(
    #                 text=resources_pb2.Text(
    #                     raw=prompt
    #                 )
    #             )
    #         )
    #     ]
    # ),
    # metadata=metadata
    # )

    # if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
    #     print(post_model_outputs_response.status)
    #     raise Exception(f"Post model outputs failed, status: {post_model_outputs_response.status.description}")

    # output = post_model_outputs_response.outputs[0]

    # reply = output.data.text.raw


    clarifai_llm = Clarifai(pat=PAT, user_id=USER_ID, app_id=APP_ID, model_id=MODEL_ID)

    template = """Question: {prompt}
    ...
    ... Answer: Let's think step by step."""

    prompt = PromptTemplate(template=template, input_variables=["prompt"])

    llm_chain = LLMChain(prompt=prompt, llm=clarifai_llm)

    reply = llm_chain.run(text_input)

    return reply

@st.cache_resource(show_spinner=False)
def query_SDXL(prompt):
        
    USER_ID = "stability-ai"
    APP_ID = "stable-diffusion-2"
    MODEL_ID = "stable-diffusion-xl"
    MODEL_VERSION_ID = "0c919cc1edfc455dbc96207753f178d7"

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (("authorization", "Key " + PAT),)

    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,  # The userDataObject is created in the overview and is required when using a PAT
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,  # This is optional. Defaults to the latest model version
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        text=resources_pb2.Text(
                            raw=prompt
                        )
                    )
                )
            ],
        ),
        metadata=metadata,
    )
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        print(post_model_outputs_response.status)
        raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)

    # Since we have one input, one output will exist here
    return post_model_outputs_response.outputs[0].data.image