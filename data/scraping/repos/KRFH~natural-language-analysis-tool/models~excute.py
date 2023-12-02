# LangChain
from langchain.llms import OpenAI
from langchain.agents import initialize_agent

# langchaintools
from models.langchaintools import mltools
from models.langchaintools import preprocessingtools as ppt

from layouts import (
    preprocessed_result_layouts,
    created_dataset_layouts,
    learning_result_layouts,
    inference_result_layouts,
)
from flask import session


def run_mltools(query: str, target: str, num_class: int):
    # Toolの設定
    tools = [
        mltools.LgbmtrainTool(),
        mltools.LgbminferenceTool(),
        ppt.DropColumnTool(),
        ppt.OnehotEncodingTool(),
        ppt.LabelEncodingTool(),
        ppt.TargetEncodingTool(),
        ppt.Fill0Tool(),
        ppt.FillMeansTool(),
        ppt.FillMedianTool(),
        ppt.MakeDatasetTool(),
        # PreprocessingTool(),
    ]
    # 通常のLangChainの設定
    llm = OpenAI(temperature=0, openai_api_key=session["api_key"])
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

    prompt = """
        {input_}一度toolを使ったら必ず終了してください．
        目的変数は{target_}です．
        """.format(
        input_=query, target_=target
    )

    results = agent.run(prompt)
    layouts = []

    if ("deleted" in results) or ("encod" in results) or ("fill" in results):
        layouts = preprocessed_result_layouts(results)

    if "dataset" in results:
        layouts = created_dataset_layouts(results)

    if "learning" in results:
        layouts = learning_result_layouts(results)
    if "inference" in results:
        layouts = inference_result_layouts(results, num_class, target)

    print(f"results:{results}")

    return layouts
