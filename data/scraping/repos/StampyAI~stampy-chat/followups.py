import requests
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.pydantic_v1 import Extra

from stampy_chat import logging

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.4 # bit of a shot in the dark - play with this later
MAX_FOLLOWUPS = 3

@dataclass
class Followup:
    text: str
    pageid: str
    score: float

# do a search like this:
# https://nlp.stampy.ai/api/search?query=what%20is%20agi

def search_authored(query: str):
    return multisearch_authored([query])


def get_followups(query):
    if not query.strip():
        return []

    url = 'https://nlp.stampy.ai/api/search?query=' + quote(query)
    response = requests.get(url).json()
    return [Followup(entry['title'], entry['pageid'], entry['score']) for entry in response]


# search with multiple queries, combine results
def multisearch_authored(queries: List[str]):
    # sort the followups from lowest to highest score
    followups = [entry for query in queries for entry in get_followups(query)]
    followups = sorted(followups, key=lambda entry: entry.score)

    # Remove any duplicates by making a map from the pageids. This should result in highest scored entry being used
    followups = {entry.pageid: entry for entry in followups if entry.score > SIMILARITY_THRESHOLD}

    # Get the first `MAX_FOLLOWUPS`
    followups = sorted(followups.values(), reverse=True, key=lambda e: e.score)
    followups = list(followups)[:MAX_FOLLOWUPS]

    if logger.is_debug():
        logger.debug(" ------------------------------ suggested followups: -----------------------------")
        for followup in followups:
            if followup.score > SIMILARITY_THRESHOLD:
                logger.debug(f'{followup.score:.2f} - suggested to user')
            else:
                logger.debug(f'{followup.score:.2f} - not suggested')
            logger.debug(followup.text)
            logger.debug(followup.pageid)
            logger.debug('')

    return followups


class StampyChain(Chain):
    """Add followup questions to the output."""

    output_key: str = "followups"

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        return ['query', 'text']

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _callback(self, run_manager, method, value):
        """Call any callbacks handlers with the given `method`.

        Langchain doesn't really have a way to extend handlers with custom methods, hence the mucking around here...
        """
        if not run_manager:
            return

        for handler in run_manager.handlers:
            if func := getattr(handler, method, None):
                func(value)

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, List[Dict[str, Any]]]:
        self._callback(run_manager, 'on_followups_start', inputs)

        followups = multisearch_authored([inputs[key] for key in self.input_keys if inputs.get(key)])
        followups = list(map(asdict, followups))

        self._callback(run_manager, 'on_followups_end', followups)
        return {self.output_key: followups}

    async def _acall(self, *args, **kwargs) -> Dict[str, List[Dict[str, Any]]]:
        return self._call(*args, **kwargs)

    @property
    def _chain_type(self) -> str:
        return "stampy_chain"
