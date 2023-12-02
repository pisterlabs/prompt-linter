import logging
import os

import openai
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from xpuls.mlmonitor.langchain.decorators.map_xpuls_project import MapXpulsProject
from xpuls.mlmonitor.langchain.decorators.telemetry_override_labels import TelemetryOverrideLabels
from xpuls.mlmonitor.langchain.instrument import LangchainTelemetry

logger = logging.getLogger(__name__)

openai.api_key="sk-td6piaoyjv4nw1j6s6vd8t2xldq6xns"
openai.api_base="https://mockgpt.wiremockapi.cloud/v1"

os.environ["OPENAI_API_BASE"] = "https://mockgpt.wiremockapi.cloud/v1"
os.environ["OPENAI_API_KEY"] = "sk-td6piaoyjv4nw1j6s6vd8t2xldq6xns"


# Set this to enable Advanced prompt tracing with server
# os.environ["XPULSAI_TRACING_ENABLED"] = "false"
os.environ["XPULSAI_TRACING_ENABLED"] = "false"

default_labels = {"system": "openai-ln-test", "agent_name": "fallback_value"}

LangchainTelemetry(
    default_labels=default_labels,
    xpuls_host_url="http://localhost:8000"
).auto_instrument()

memory = ConversationBufferMemory(memory_key="chat_history")
chat_model = ChatOpenAI(
    deployment_name="gpt35turbo",
    model_name="gpt-35-turbo",
    temperature=0
)


@TelemetryOverrideLabels(agent_name="chat_agent_alpha")
@MapXpulsProject(project_id="default")  # Get Project ID from console
def run_openai_agent():
    agent = initialize_agent(llm=chat_model,
                             verbose=True,
                             tools=[],
                             agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                             memory=memory,
                             # handle_parsing_errors="Check your output and make sure it conforms!",
                             return_intermediate_steps=False,
                             agent_executor_kwargs={"extra_prompt_messages": "test"})

    try:
        res = agent.run("You are to behave as a think tank to answer the asked question in most creative way,"
                        " ensure to NOT be abusive or racist, you should validate your response w.r.t to validity "
                        "in practical world before giving final answer" +
                        f"\nQuestion: How does nature work?, is balance of life true? \n")
    except ValueError as e:
        res = str(e)
        if not res.startswith("Could not parse LLM output: `"):
            raise e
        logger.error(f" Got ValueError: {e}")
        res = res.removeprefix("Could not parse LLM output: `").removesuffix("`")

    return res
