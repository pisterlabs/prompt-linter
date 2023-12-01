import pandas as pd
import os
from datasets import load_dataset
from huggingface_hub import Repository
from typing import Any

# import llm chain
from langchain import LLMChain

# load env variables
FORCE_PUSH = os.getenv("FORCE_PUSH", "0") == "1"
HF_TOKEN = os.getenv("HF_TOKEN", None)

def loadResponses():
    #check if file exists
    if os.path.isfile('responses/response.csv'):
        df = pd.read_csv('responses/response.csv')
    else:
        df = pd.DataFrame(columns=['response'])
    return df
        
def saveResponsetoDataset(response):
    # load the dataset
    repo = Repository(local_dir="responses", clone_from="nazneen/rlhf", use_auth_token=HF_TOKEN, repo_type="dataset")

    repo.git_pull()

    df = loadResponses()
    #concat response to dataset
    df = pd.concat([df, pd.DataFrame({'response': [response]})])
    # save the dataset
    df.to_csv('responses/response.csv', index=False)

    # with repo.commit(commit_message="LLM response", blocking = False):
    #     repo.git_add("responses/response.csv")
    #     repo.git_add(auto_lfs_track=True)
    repo.git_add("/Users/nazneenrajani/workspace/rlhf/rlhf_langchain/rlhf_langchain/responses/response.csv")
    repo.git_commit(commit_message="LLM response")
    repo.git_push()


# create a wraper that receives a class called LLMChain or LLM and adda functionality when the function predict is called
class RLHFLLMChain(LLMChain):
    # override predict function
    def predict(self, **kwargs: Any) -> str:
        """Format prompt with kwargs and pass to LLM.
        Args:
            **kwargs: Keys to pass to prompt template.
        Returns:
            Completion from LLM.
        """
        # add new functionality
        # get the prediction
        response = self(kwargs)[self.output_key]
        # save the prediction
        saveResponsetoDataset(response)
        # return the prediction
        return response