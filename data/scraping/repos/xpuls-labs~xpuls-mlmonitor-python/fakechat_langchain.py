import logging
import os

import openai
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI, FakeListChatModel
from langchain.memory import ConversationBufferMemory

from xpuls.mlmonitor.langchain.decorators.map_xpuls_project import MapXpulsProject
from xpuls.mlmonitor.langchain.decorators.telemetry_override_labels import TelemetryOverrideLabels
from xpuls.mlmonitor.langchain.instrument import LangchainTelemetry

logger = logging.getLogger(__name__)


# Set this to enable Advanced prompt tracing with server
# os.environ["XPULSAI_TRACING_ENABLED"] = "false"
os.environ["XPULSAI_TRACING_ENABLED"] = "false"

default_labels = {"system": "openai-ln-test", "agent_name": "fallback_value"}

LangchainTelemetry(
    default_labels=default_labels,
    xpuls_host_url="http://localhost:8000"
).auto_instrument()

memory = ConversationBufferMemory(memory_key="chat_history")
chat_model = FakeListChatModel(
    responses=[
        "The Symphony of Ecosystems: Nature works through a delicate symphony of ecosystems, where every organism plays a critical role. The balance of life is like a finely tuned orchestra, with each species contributing to the overall harmony of the environment.",
        "Nature's Web of Interdependence: In nature, everything is interconnected. The balance of life is a web of interdependence, where the survival of one species often hinges on the well-being of another.",
        "The Dynamic Dance of Evolution: Nature operates through the dance of evolution, constantly adapting and evolving. The balance of life is not static but a dynamic equilibrium, ever-changing and adapting to new challenges.",
        "The Cycle of Renewal: Nature functions through cycles of growth, decay, and renewal. This cycle ensures the balance of life, as new life emerges from the old, maintaining a continuous flow of energy and resources.        ",
        "The Balance of Opposites: Nature maintains balance through the interplay of opposites – day and night, predator and prey, growth and decay. This balance is the essence of life, creating a harmonious and sustainable world.",
        "The Gaia Hypothesis: Like the Gaia Hypothesis, nature can be seen as a self-regulating entity, where the balance of life is maintained through complex interactions among living organisms and their environment.",
        "Chaos and Order in Natural Systems: Nature operates on a spectrum from chaos to order. The balance of life lies in this spectrum, where seemingly chaotic natural events lead to the emergence of complex, ordered systems.",
        "The Ripple Effect of Actions: In nature, every action has a ripple effect. The balance of life is a testament to how small changes in one part of the ecosystem can have far-reaching impacts.   ",
        "Nature as a Teacher: Nature works as the greatest teacher, showing us the importance of balance. The balance of life is a lesson in coexistence, sustainability, and respect for all living things",
        "The Mosaic of Biodiversity: Nature thrives on biodiversity. The balance of life is like a mosaic, where the variety of species creates a resilient and vibrant ecosystem."
    ]
)


@TelemetryOverrideLabels(agent_name="chat_agent_alpha")
@MapXpulsProject(project_id="default")  # Get Project ID from console
def run_fakechat_agent():
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
