import abc
import logging
import time

import cohere
import openai

logger = logging.getLogger(__name__)


def create(args):
    if args.backend == "openai":
        openai.api_key = args.openai.api_key
        return OpenAI(args)
    else:
        return Cohere(args)


class Backend(abc.ABC):
    POSSIBLE_EXCEPTIONS = (Exception,)
    RATE_LIMIT = 1.1  # Rate limit, in seconds

    def __init__(self, args):
        self.args = args

    def generate(self, ex, max_requests=5, cooldown=10):
        time.sleep(self.RATE_LIMIT)
        n_requests = 0
        while n_requests < max_requests:
            try:
                res = self._generate(ex)
            except self.POSSIBLE_EXCEPTIONS as e:
                logging.warning("Encountered %s, retrying", e)
                time.sleep(cooldown)
                n_requests += 1
            else:
                res = self._postprocess(res)
                return res
        raise RuntimeError(f"Could not get answer for {ex} after {n_requests} requests")

    @abc.abstractmethod
    def _generate(self, ex):
        raise NotImplementedError

    def _postprocess(self, res):
        return res


class OpenAI(Backend):
    POSSIBLE_EXCEPTIONS = (openai.error.RateLimitError,)

    def _generate(self, ex):
        return openai.Completion.create(
            engine=self.args.openai.engine,
            prompt=ex["prompt"],
            max_tokens=ex["scratch_len"] + self.args.data.token_buffer,
            logprobs=5,
            temperature=0.0,
            stop="X",
        )["choices"][0]


class Cohere(Backend):
    POSSIBLE_EXCEPTIONS = (cohere.error.CohereError,)
    END_SEQUENCE = ".\n\n"

    def __init__(self, args):
        super().__init__(args)
        self.co = cohere.Client(self.args.cohere.api_key)

    def _generate(self, ex):
        return self.co.generate(
            model=self.args.cohere.model,
            prompt=ex,
            max_tokens=200,
            temperature=0.0,
            stop_sequences=[self.END_SEQUENCE],
            num_generations=1,
        ).generations[0]

    def _postprocess(self, res):
        """Massage into OpenAI format."""
        # Remove trailing sequence
        text = res.text
        if text.endswith(self.END_SEQUENCE):
            text = text[:-len(self.END_SEQUENCE)]
        return text
