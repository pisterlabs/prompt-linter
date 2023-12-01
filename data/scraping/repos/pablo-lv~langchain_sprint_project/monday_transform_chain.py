import json
import warnings

warnings.filterwarnings('ignore')

from langchain.chains import TransformChain

from logger_setup import log


def transform_func(inputs: dict) -> dict:
    log.info("Transforming inputs")
    inputs = inputs["item"]
    inputs = json.loads(inputs)
    title = inputs["title"]
    priority = inputs["priority"]
    description = inputs["description"]
    return {"output_dict": {"title": title, "text": description, "priority": priority}}


def build_transform_chain():
    # Transform Chain
    transform_chain = TransformChain(
        input_variables=["item"], output_variables=["output_dict"], transform=transform_func
    )

    return transform_chain
