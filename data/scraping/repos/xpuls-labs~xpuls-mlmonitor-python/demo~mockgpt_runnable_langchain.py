import logging
import os

import openai
from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate

from xpuls.mlmonitor.langchain.decorators.map_xpuls_project import MapXpulsProject
from xpuls.mlmonitor.langchain.decorators.telemetry_override_labels import TelemetryOverrideLabels
from xpuls.mlmonitor.langchain.instrument import LangchainTelemetry

logger = logging.getLogger(__name__)

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_URL")
os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_URL")
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
openai.api_version = "2023-03-15-preview"

# Set this to enable Advanced prompt tracing with server
# os.environ["XPULSAI_TRACING_ENABLED"] = "false"
os.environ["XPULSAI_TRACING_ENABLED"] = "false"

default_labels = {"system": "openai-ln-test", "agent_name": "fallback_value"}

LangchainTelemetry(
    default_labels=default_labels,
    xpuls_host_url="http://localhost:8000"
).auto_instrument()

chat_model = AzureChatOpenAI(
    deployment_name="gpt35turbo",
    model_name="gpt-35-turbo",
    temperature=0
)


@TelemetryOverrideLabels(agent_name="chat_agent_alpha")
@MapXpulsProject(project_id="default")  # Get Project ID from console
def run_openai_agent():
    prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
    chain = prompt | chat_model
    try:
        res = chain.invoke({"foo": "bears"})
    except ValueError as e:
        res = str(e)
        if not res.startswith("Could not parse LLM output: `"):
            raise e
        logger.error(f" Got ValueError: {e}")
        res = res.removeprefix("Could not parse LLM output: `").removesuffix("`")

    return res
