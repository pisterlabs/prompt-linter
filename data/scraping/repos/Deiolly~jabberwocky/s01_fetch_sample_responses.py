"""Make a bunch of API calls and save sample GPT responses. This is useful for
testing, particularly with paid backends, where we want to repeatedly test our
functions on a variety of different parameter configurations without spending
a lot. Should hopefully only need to run this once.

Note: this is currently only for gooseai/openai, but might be nice to
eventually expand it to use any query_function. Even though others are free,
could be a good way to avoid the messy ad-hoc querying I've used so far during
development.
"""

from jabberwocky.openai_utils import GPTBackend
import openai

from htools.cli import fire, module_docstring
from htools.core import save


gpt = GPTBackend()
txts = ['Yesterday was', 'How many']


@module_docstring
def main(backend='gooseai'):
    """Currently tests combinations of 3 different scenarios:
    1. Single prompt vs multiple prompts (np)
    2. Single completion per prompt vs. multiple completions (nc)
    3. Streaming mode vs static responses (streamed responses are converted to
    lists since we can't easily pickle generators)

    The resulting dict is pickled to data/misc. As of 4/10/22, we have 8 keys
    (3 parameters ^ 2 possible values = 8) and keys are a tuple of 3 booleans
    in specifying whether a query used multiple prompts, whether it requested
    multiple completions, and whether it was in streaming mode. For example:

    # Get sample response for multiple inputs, multiple outputs,
    # non-streaming mode. Think of indexing as data[np, nc, stream].
    data = load('data/misc/gooseai_sample_responses.pkl')
    data[True, True, False)
    """
    if backend not in ('gooseai', 'openai'):
        raise NotImplementedError(
            f'This script does not currently support backend={backend}.'
        )

    gpt.switch(backend)

    # Key: (multi_in, multi_out, stream)
    responses = {}
    for multi_in in (True, False):
        for multi_out in (True, False):
            for stream in (True, False):
                prompt = txts if multi_in else txts[0]
                nc = 1 + multi_out
                print(prompt, nc, stream)
                res = openai.Completion.create(
                    prompt=prompt,
                    engine=GPTBackend.engine(0),
                    max_tokens=3,
                    logprobs=3,
                    n=nc,
                    stream=stream
                )
                if stream: res = list(res)
                responses[multi_in, multi_out, stream] = res
    save(responses, f'data/misc/{backend}_sample_responses.pkl')
    return responses


if __name__ == '__main__':
    fire.Fire(main)
