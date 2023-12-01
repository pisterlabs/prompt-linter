'''
Example of using feature store with langchain and llm.
The goal is to write a note to a specific driver regarding their up-to-date statistics.
The basic idea is to call a feature store from inside a prompt template 
to retrieve values that are then formatted into the prompt.
Run the feast-start.py script to populate the feature store.
'''
import sys
from langchain.prompts import PromptTemplate, StringPromptTemplate
from feast import FeatureStore

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from bedrock.utils import bedrock, print_ww
import os
from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock

# Load the feature store
feast_repo_path = "./my_feature_repo/feature_repo"
store = FeatureStore(repo_path=feast_repo_path)

'''
Define a prompt template that uses the feature store to get the driver's stats.
StringPromptTemplate exposes the format method, returning a prompt.
Create a new model by parsing and validating input data from keyword arguments.
'''
class FeastPromptTemplate(StringPromptTemplate):
    
    template = """Given the driver's up to date stats, write them note relaying those stats to them.
If they have a conversation rate above .5, give them a compliment. Otherwise, make a silly joke about chickens at the end to make them feel better

Here are the drivers stats:
Conversation rate: {conv_rate}
Acceptance rate: {acc_rate}
Average Daily Trips: {avg_daily_trips}

Your response:"""
    prompt = PromptTemplate.from_template(template)

    def format(self, **kwargs) -> str:
        driver_id = kwargs.pop("driver_id")
        feature_vector = store.get_online_features(
            features=[
                "driver_hourly_stats:conv_rate",
                "driver_hourly_stats:acc_rate",
                "driver_hourly_stats:avg_daily_trips",
            ],
            entity_rows=[{"driver_id": driver_id}],  # cloud be a list of driver_ids
        ).to_dict()
        kwargs["conv_rate"] = feature_vector["conv_rate"][0]
        kwargs["acc_rate"] = feature_vector["acc_rate"][0]
        kwargs["avg_daily_trips"] = feature_vector["avg_daily_trips"][0]
        return self.prompt.format(**kwargs)
    


def buildBedrockClient():
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
    os.environ["BEDROCK_ENDPOINT_URL"] = "https://bedrock." + os.environ["AWS_DEFAULT_REGION"] + ".amazonaws.com/"

    return  bedrock.get_bedrock_client(
        assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
        endpoint_url=os.environ.get("BEDROCK_ENDPOINT_URL", None),
        region=os.environ.get("AWS_DEFAULT_REGION", None))


if __name__ == "__main__":
    print("initialize the chain with titan llm from AWS bedrock")
    print("Be sure to have session token as env variable")
    bedrock_client= buildBedrockClient()
    titan_llm = Bedrock(model_id="amazon.titan-tg1-large", client=bedrock_client)
    prompt_template = FeastPromptTemplate(input_variables=["driver_id"])

    chain = LLMChain(llm=titan_llm, prompt=prompt_template)
    # run has positional arguments or keyword arguments, 
    print(chain.run(1001))