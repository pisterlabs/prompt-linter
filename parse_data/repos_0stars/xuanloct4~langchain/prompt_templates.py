import environment

from feast import FeatureStore

# You may need to update the path depending on where you stored it
feast_repo_path = "./my_feature_repo/feature_repo/"
store = FeatureStore(repo_path=feast_repo_path)

from langchain.prompts import PromptTemplate, StringPromptTemplate
template = """Given the driver's up to date stats, write them note relaying those stats to them.
If they have a conversation rate above .5, give them a compliment. Otherwise, make a silly joke about chickens at the end to make them feel better

Here are the drivers stats:
Conversation rate: {conv_rate}
Acceptance rate: {acc_rate}
Average Daily Trips: {avg_daily_trips}

Your response:"""
prompt = PromptTemplate.from_template(template)

class FeastPromptTemplate(StringPromptTemplate):
    
    def format(self, **kwargs) -> str:
        driver_id = kwargs.pop("driver_id")
        feature_vector = store.get_online_features(
            features=[
                'driver_hourly_stats:conv_rate',
                'driver_hourly_stats:acc_rate',
                'driver_hourly_stats:avg_daily_trips'
            ],
            entity_rows=[{"driver_id": driver_id}]
        ).to_dict()
        kwargs["conv_rate"] = feature_vector["conv_rate"][0]
        kwargs["acc_rate"] = feature_vector["acc_rate"][0]
        kwargs["avg_daily_trips"] = feature_vector["avg_daily_trips"][0]
        return prompt.format(**kwargs)
prompt_template = FeastPromptTemplate(input_variables=["driver_id"])
# print(prompt_template.format(driver_id=1001))

# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
# chain = LLMChain(llm=ChatOpenAI(), prompt=prompt_template)
# print(chain.run(1001))




import inspect

def get_source_code(function_name):
    # Get the source code of the function
    return inspect.getsource(function_name)

from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, validator


class FunctionExplainerPromptTemplate(StringPromptTemplate, BaseModel):
    """ A custom prompt template that takes in the function name as input, and formats the prompt template to provide the source code of the function. """

    @validator("input_variables")
    def validate_input_variables(cls, v):
        """ Validate that the input variables are correct. """
        if len(v) != 1 or "function_name" not in v:
            raise ValueError("function_name must be the only input_variable.")
        return v

    def format(self, **kwargs) -> str:
        # Get the source code of the function
        source_code = get_source_code(kwargs["function_name"])

        # Generate the prompt to be sent to the language model
        prompt = f"""
        Given the function name and source code, generate an English language explanation of the function.
        Function Name: {kwargs["function_name"].__name__}
        Source Code:
        {source_code}
        Explanation:
        """
        return prompt
    
    def _prompt_type(self):
        return "function-explainer"

fn_explainer = FunctionExplainerPromptTemplate(input_variables=["function_name"])

# Generate a prompt for the function "get_source_code"
prompt = fn_explainer.format(function_name=get_source_code)
print(prompt)

# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
# chain = LLMChain(llm=ChatOpenAI(), prompt=fn_explainer)
# print(chain.run(get_source_code))
