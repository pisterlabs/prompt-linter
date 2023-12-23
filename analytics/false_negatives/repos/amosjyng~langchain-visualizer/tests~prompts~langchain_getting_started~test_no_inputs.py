import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import vcr_langchain as vcr
from langchain import PromptTemplate
from langchain.llms import OpenAI

# ========================== Start of langchain example code ==========================
# https://langchain.readthedocs.io/en/latest/modules/prompts/getting_started.html


# An example prompt with no input variables
no_input_prompt = PromptTemplate(input_variables=[], template="Tell me a joke.")
prompt = no_input_prompt.format()


# ================================== Execute example ==================================


def test_prompt():
    assert prompt.parts == ("Tell me a joke.",)


@vcr.use_cassette()
async def no_inputs_prompt_demo():
    agent = OpenAI(model_name="text-ada-001", temperature=0)
    return agent(prompt)


def test_llm_usage_succeeds():
    """
    Check that it works like a regular prompt.
    Also, record playback for easy visualization.
    """
    result = asyncio.get_event_loop().run_until_complete(no_inputs_prompt_demo())
    assert result.strip().startswith("Why did the chicken cross the road?")


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(no_inputs_prompt_demo)
