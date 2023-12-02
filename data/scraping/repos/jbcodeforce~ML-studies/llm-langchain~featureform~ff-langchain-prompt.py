import featureform as ff
from utils import bedrock,  print_ww
from langchain.llms import Bedrock
from langchain.prompts import PromptTemplate, StringPromptTemplate
from langchain.chains import LLMChain



featureform_client = ff.Client(insecure=True)
aws_bedrock_client = bedrock.get_bedrock_client()

template = """Given the amount a user spends on average per transaction, let them know if they are a high roller. Otherwise, make a silly joke about chickens at the end to make them feel better

Here are the user's stats:
Average Amount per Transaction: ${avg_transaction}

Your response:"""
prompt = PromptTemplate.from_template(template)

class FeatureformPromptTemplate(StringPromptTemplate):
    def format(self, **kwargs) -> str:
        user_id = kwargs.pop("user_id")
        avg_transaction = featureform_client.features(["avg_transactions"], {"user": user_id})
        kwargs["avg_transaction"]=avg_transaction
        return prompt.format(**kwargs)
    
prompt_template = FeatureformPromptTemplate(input_variables=["user_id"])
llm= Bedrock(
                client=aws_bedrock_client,
                model_id="anthropic.claude-v1"
            )   
chain = LLMChain(llm=llm, prompt=prompt_template)
print_ww(chain.run("C1410926"))