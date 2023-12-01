import json
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler


class _ContentHandler_GLM(LLMContentHandler):
    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps({'inputs': prompt, 'parameters': model_kwargs, 'history': []})
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json['response']
    

class SageMakerLLM():

    def __init__(self,
                 SageMakerEndpointName,
                 AWSRegion,
                 LLMType='GLM-6b',
                 LLMArgs={'top_p': 0.45, 'temperature': 0.2}):

        self.LLM_ENDPOINT_NAME = SageMakerEndpointName
        self.REGION = AWSRegion

        self.modelArgs = LLMArgs

        self.LlmHandler = _ContentHandler_GLM()
        if LLMType == 'GLM-6b':
            self.LlmHandler = _ContentHandler_GLM()
        # elif
        #     TODO
        else:
            self.LlmHandler = _ContentHandler_GLM()


        self.sm_llm=SagemakerEndpoint(
                endpoint_name=self.LLM_ENDPOINT_NAME,
                region_name=self.REGION,
                model_kwargs=self.modelArgs,
                content_handler=self.LlmHandler
            )
    
    
    def get_llm(self):
        return self.sm_llm        