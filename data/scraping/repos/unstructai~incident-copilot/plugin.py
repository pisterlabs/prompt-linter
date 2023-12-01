"""
.. Added by Kishore Jalleda
.. full list of modifications at https://github.com/unstructai
.. copyright: (c) 2023 Kishore Jalleda
.. author:: Kishore Jalleda <kjalleda@gmail.com>
"""
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from dispatch.plugins.bases import ArtificialIntelligencePlugin
from dispatch.decorators import apply, counter, timer
import logging

from .config import AnthropicConfiguration

from ._version import __version__

log = logging.getLogger(__name__)
anthropic = Anthropic()


@apply(counter, exclude=["__init__"])
@apply(timer, exclude=["__init__"])
class AnthropicPlugin(ArtificialIntelligencePlugin):
    title = "Anthropic Plugin - Generative Artificial Intelligence"
    slug = "anthropic-artificial-intelligence"
    description = "Uses Anthropic's platform to allow users to ask questions in natural language."
    version = __version__

    author = "Unstruct AI"
    author_url = "htpps://github.com/unstructai"

    def __init__(self):
        self.configuration_schema = AnthropicConfiguration

    def ask(self, prompt: str) -> dict:
        """Ask a question to the Anthropic API."""
        completion = anthropic.completions.create(
            model=self.configuration.model,
            max_tokens_to_sample=self.configuration.max_tokens_to_sample,
            prompt=HUMAN_PROMPT + prompt + AI_PROMPT,
        )

        return completion.completion
