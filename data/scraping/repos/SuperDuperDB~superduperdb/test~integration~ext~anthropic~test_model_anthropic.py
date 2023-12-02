import pytest
import vcr

from superduperdb.ext.anthropic import AnthropicCompletions

CASSETTE_DIR = 'test/integration/ext/anthropic/cassettes'


@pytest.mark.skip(reason="API is not publically available yet")
@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_completions.yaml',
    filter_headers=['authorization'],
)
def test_completions():
    e = AnthropicCompletions(model='claude-2', prompt='Hello, {context}')
    resp = e.predict('', one=True, context=['world!'])

    assert isinstance(resp, str)


@pytest.mark.skip(reason="API is not publically available yet")
@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_batch_completions.yaml',
    filter_headers=['authorization'],
)
def test_batch_completions():
    e = AnthropicCompletions(model='claude-2')
    resp = e.predict(['Hello, world!'], one=False)

    assert isinstance(resp, list)
    assert isinstance(resp[0], str)


@pytest.mark.skip(reason="API is not publically available yet")
@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_completions_async.yaml',
    filter_headers=['authorization'],
)
@pytest.mark.asyncio
async def test_completions_async():
    e = AnthropicCompletions(model='claude-2', prompt='Hello, {context}')
    resp = await e.apredict('', one=True, context=['world!'])

    assert isinstance(resp, str)


@pytest.mark.skip(reason="API is not publically available yet")
@vcr.use_cassette(
    f'{CASSETTE_DIR}/test_batch_completions_async.yaml',
    filter_headers=['authorization'],
)
@pytest.mark.asyncio
async def test_batch_completions_async():
    e = AnthropicCompletions(model='claude-2')
    resp = await e.apredict(['Hello, world!'], one=False)

    assert isinstance(resp, list)
    assert isinstance(resp[0], str)
