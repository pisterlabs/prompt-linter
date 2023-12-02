import langchain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate, load_prompt
import wandb
from wandb.integration.langchain import WandbTracer
import openai
import streamlit as st

def generate_prd_v1(new_feature, new_feature_desc, wandb_name):
    wandb.login(key=st.secrets["WANDB_API_KEY"])

    wandb.init(
        project="generate_simple_prd",
        config={
            "model": "gpt-3.5-turbo",
            "temperature": 0
        },
        entity="arihantsheth",
        name=wandb_name,
    )

    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=st.secrets["OPENAI_API_KEY"])
    prompt_template = load_prompt("prompt_templates/generate_prd_template_v1.json") # For deployment
    # prompt_template = load_prompt("../prompt_templates/generate_prd_template_v1.json") # For local testing

    prompt = prompt_template.format(new_feature=new_feature, new_feature_desc=new_feature_desc)

    try:
        output = llm(prompt, callbacks=[WandbTracer()])
    except openai.error.AuthenticationError as e:
        print("OpenAI unknown authentication error")
        print(e.json_body)
        print(e.headers)
        return

    # with open(f"./generated_prds/{new_feature}_prd_v1.md", "w") as f: # For deployment
    # # with open(f"../generated_prds/{new_feature}_prd_v1.md", "w") as f: # For local testing
    #     f.write(output)

    wandb.finish()
    return output