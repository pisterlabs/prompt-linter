import re
import sys
import logging
import pytz
from datetime import datetime
from langchain.llms import Cohere
from moments.agent import Agent
from moments.moment import Moment, Self, Context, Rejected

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format="%(levelname)s | %(message)s"
)
LOG = logging.getLogger(__name__)


llm = Cohere()


class LlmCohereAgent(Agent):
    def before(self: "LlmCohereAgent", moment: Moment):
        # Add context if not already present.
        is_context_added = False
        for occurrence in moment.occurrences:
            if isinstance(occurrence, Context):
                is_context_added = True
                break

        if not is_context_added:
            tz = pytz.timezone("America/Los_Angeles")
            current_time = datetime.now(tz)
            moment.occurrences.append(
                Context('```time: "' + current_time.strftime("%I:%M %p") + '"```')
            )

    def do(self: "LlmCohereAgent", moment: Moment):
        # Add final "Self:" for agent to speak.
        prompt = str(moment) + "Self: "
        # Complete with langchain
        response = llm(prompt.strip())
        line = response.split("\n")[0].strip()
        print(f"-->{line}<--")
        if not re.match(r"^Self:\s+(\((.*)\)\s+)?\"(.+)\"$", line):
            # Try to fix it
            if not line.startswith('"'):
                line = '"' + line
            if not line.endswith('"'):
                line = line + '"'
            if not line.startswith("Self: "):
                line = "Self: " + line

        # Get the rejected
        response = llm(prompt.strip())
        rejected_line = response.split("\n")[0].strip()
        rejected_line = rejected_line.replace("Self: ", "Rejected: ")
        if not re.match(r"^Self:\s+(\((.*)\)\s+)?\"(.+)\"$", rejected_line):
            # Try to fix it
            if not rejected_line.startswith('"'):
                rejected_line = '"' + rejected_line
            if not rejected_line.endswith('"'):
                rejected_line = rejected_line + '"'
            if not rejected_line.startswith("Rejected: "):
                rejected_line = "Rejected: " + rejected_line

        # First, add the rejected
        print(f"-->{rejected_line}<--")
        moment.occurrences.append(Rejected.parse(rejected_line))
        # Then add actual response
        moment.occurrences.append(Self.parse(line))

    def after(self: "LlmCohereAgent", moment: Moment):
        pass
