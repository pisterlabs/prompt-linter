import os
from langchain.llms.bedrock import Bedrock
from utils import bedrock
from langchain.prompts import PromptTemplate 


os.environ["AWS_DEFAULT_REGION"] = "us-east-1"  
os.environ["AWS_PROFILE"] = "zigi-bedrock"

boto3_bedrock = bedrock.get_bedrock_client(
    # IAM User에 Bedrock에 대한 권한이 없이 Role을 Assume하는 경우
    # assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None), 
    region=os.environ.get("AWS_DEFAULT_REGION", None)
)

inference_modifier = {'max_tokens_to_sample':4096, 
                      "temperature":0.5,
                      "top_k":250,
                      "top_p":1,
                      "stop_sequences": ["\n\nHuman"]
                     }

textgen_llm = Bedrock(model_id = "anthropic.claude-v2",
                    client = boto3_bedrock, 
                    model_kwargs = inference_modifier
                    )

prompt_template = """
Human: {subject}에 대한 주제로 {category}를(을) 한글로 작성해주세요 
Assistant:"""

multi_var_prompt = PromptTemplate(
    input_variables=["subject", "category"], template=prompt_template
)

prompt = multi_var_prompt.format(subject="가을", category="시")

response = textgen_llm(prompt)
result = response[response.index('\n')+1:]

print(f"프롬프트 : {prompt}")
print(f"생성 문장 : {result}")
