import os
import boto3
import pandas as pd
from llms.azure_llms import create_llm
#from langchain.agents import create_csv_agent
from agents.modified_langchain.csv.base import create_csv_agent
from langchain.agents.agent_types import AgentType
from chains.pandas_multi_prompt import PandasMultiPromptChain

# llm = create_llm(temp=0)

# llm.request_timeout = 30

# print("Creating CSV Agent.")
# combined_data = os.path.join("data", "combined.csv")
#df = pd.read_csv(combined_data)


# mitre_dir = "../data"
# mitre_data = [os.path.join(mitre_dir, fn) for fn in ["software.csv", "groups.csv", "mitigations.csv"]]
# if not os.path.exists(combined_data):
#     combined_df = None
#     keys = ['TID', 'Technique Name']
#     for fp in mitre_data:
#         next_df = pd.read_csv(fp)
#         if combined_df is not None:
#             combined_df = combined_df.merge(
#                 next_df,
#                 on=['TID', 'Technique Name'],
#                 how='outer'
#             )
#         else:
#             combined_df = next_df
    
#     combined_df.to_csv(combined_data)


# Waiting on Secret Key info for Amazon.
def download_files(bucket_name="team5.2-mitre", data_dir="data", files=[]):
    s3_client = boto3.client('s3')
    os.makedirs(data_dir, exist_ok=True)
    for fn in files:
        fp = os.path.join(data_dir, fn)
        if not os.path.exists(fp):
            print(f"Downloading {fn}")
            s3_client.download_file(bucket_name, fn, fp)

download_files(files=["combined.csv"])

def get_mitre_agent(use_memory=False):
    llm = create_llm(temp=0)
    llm.request_timeout = 30
    print("Creating CSV Agent.")
    combined_data = os.path.join("data", "combined.csv")
    
    # Use a selection of different rows from the data in the prompt
    # instead of just the header (has lots of repetition).
    df_rows = [[0,2000,4000,89000,123000]]

    csv_agent = create_csv_agent(
        llm,
        combined_data,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        df_rows=df_rows,
        include_df_in_prompt = None,
        input_variables = ["df_content", "input", "agent_scratchpad"],
        use_memory=use_memory
    )

    print("Finished Creating CSV Agent.")

    return csv_agent