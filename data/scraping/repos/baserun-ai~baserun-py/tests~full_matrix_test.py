import concurrent

import pytest
from anthropic import AsyncAnthropic, Anthropic, HUMAN_PROMPT, AI_PROMPT
from openai import OpenAI, AsyncOpenAI

import baserun


@pytest.fixture
def client():
    return OpenAI()


@pytest.fixture
def async_client():
    return AsyncOpenAI()


@pytest.mark.asyncio
async def test_paris_async(async_client):
    response = await async_client.completions.create(
        model="text-davinci-003",
        temperature=0.7,
        prompt="What are three activities to do in Paris?",
    )

    baserun.evals.includes(
        "Includes Eiffel Tower", response.choices[0].text, ["Eiffel Tower"]
    )
    baserun.evals.not_includes(
        "AI Language check", response.choices[0].text, ["AI Language"]
    )
    baserun.evals.model_graded_security("Malicious", response.choices[0].text)


@pytest.mark.asyncio
async def test_egypt_async_stream(async_client):
    response = await async_client.completions.create(
        model="text-davinci-003",
        temperature=0.7,
        prompt="What are three activities to do in Egypt?",
        stream=True,
    )

    async for chunk in response:
        print(chunk.choices[0].text)


@pytest.mark.asyncio
async def test_egypt_async(async_client):
    response = await async_client.completions.create(
        model="text-davinci-003",
        temperature=0.7,
        prompt="What are three activities to do in Egypt?",
    )

    baserun.evals.includes("Includes Pyramid", response.choices[0].text, ["Pyramid"])
    baserun.evals.not_includes(
        "AI Language check", response.choices[0].text, ["AI Language"]
    )
    baserun.evals.model_graded_security("Malicious", response.choices[0].text)
    baserun.evals.model_graded_fact(
        "Fact",
        "tell me about france",
        "eiffel tower is nice",
        response.choices[0].text,
    )


def test_paris_sync_stream(client):
    response = client.completions.create(
        model="text-davinci-003",
        temperature=0.7,
        prompt="What are three activities to do in Paris?",
        stream=True,
    )

    for chunk in response:
        print(chunk.choices[0].text)


def test_paris_sync(client):
    response = client.completions.create(
        model="text-davinci-003",
        temperature=0.7,
        prompt="What are three activities to do in Paris?",
    )

    baserun.log("Paris", response.choices[0].text)


@pytest.mark.asyncio
async def test_paris_chat_async_stream(async_client):
    response = await async_client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": "What are three activities to do in Paris?",
            }
        ],
        stream=True,
    )

    async for chunk in response:
        print(chunk.choices[0].delta)


@pytest.mark.asyncio
async def test_paris_chat_async(async_client):
    response = await async_client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": "What are three activities to do in Paris?",
            }
        ],
    )

    baserun.log("Paris", response.choices[0].message.content)


@pytest.mark.asyncio
async def test_madrid_chat_async(async_client):
    response = await async_client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": "What are three activities to do in Paris?",
            }
        ],
    )
    baserun.log("Madrid", response.choices[0].message.content)


def test_paris_chat_sync_stream(client):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": "What are three activities to do in Paris?",
            }
        ],
        stream=True,
    )

    for chunk in response:
        print(chunk.choices[0].delta)


def test_paris_chat_sync(client):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": "What are three activities to do in Paris?",
            }
        ],
    )

    baserun.evals.includes(
        "Includes Eiffel Tower",
        response.choices[0].message.content,
        ["Eiffel Tower"],
    )
    baserun.evals.not_includes(
        "AI Language check",
        response.choices[0].message.content,
        ["AI Language"],
    )
    baserun.evals.model_graded_security(
        "Malicious", response.choices[0].message.content
    )


def test_egypt_chat_sync(client):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": "What are three activities to do in Egypt?",
            }
        ],
    )

    baserun.evals.includes(
        "Includes Pyramid", response.choices[0].message.content, ["Pyramid"]
    )
    baserun.evals.not_includes(
        "AI Language check",
        response.choices[0].message.content,
        ["AI Language"],
    )
    baserun.evals.model_graded_security(
        "Malicious", response.choices[0].message.content
    )


@pytest.mark.parametrize(
    "place,expected", [("Paris", "Eiffel Tower"), ("Rome", "Colosseum")]
)
def test_trip(place, expected):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=[
            {"role": "user", "content": f"What are three activities to do in {place}?"}
        ],
    )

    assert expected in response.choices[0].message.content


def test_dogs_anthropic_sync():
    anthropic = Anthropic()
    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT} How many toes do dogs have?{AI_PROMPT}",
    )
    baserun.log("Dogs", completion.completion)


def test_dogs_anthropic_sync_stream():
    anthropic = Anthropic()
    stream = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT} How many toes do dogs have?{AI_PROMPT}",
        stream=True,
    )

    for data in stream:
        print(data.completion)


@pytest.mark.asyncio
async def test_dogs_anthropic_async():
    anthropic = AsyncAnthropic()
    completion = await anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT} How many toes do dogs have?{AI_PROMPT}",
    )
    baserun.log("Async Dogs", completion.completion)


@pytest.mark.asyncio
async def test_dogs_anthropic_async_stream():
    anthropic = AsyncAnthropic()
    stream = await anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=300,
        prompt=f"{HUMAN_PROMPT} How many toes do dogs have?{AI_PROMPT}",
        stream=True,
    )

    async for data in stream:
        print(data.completion)


@pytest.mark.asyncio
async def test_function_call(async_client):
    await async_client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in Paris?",
            }
        ],
        functions=[
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        ],
    )


@pytest.mark.asyncio
async def test_function_call_stream(async_client):
    response = await async_client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in Paris?",
            }
        ],
        functions=[
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        ],
        stream=True,
    )

    async for chunk in response:
        print(chunk.choices[0].delta)


@pytest.mark.asyncio
async def test_function_call_message(async_client):
    await async_client.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0.7,
        messages=[
            {
                "role": "user",
                "content": "What's the weather like in Paris?",
            },
            {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "get_current_weather",
                    "arguments": '{"location": "Paris"}',
                },
            },
            {
                "role": "function",
                "name": "get_current_weather",
                "content": '{"temperature": 22, "unit": "celsius", "description": "Sunny"}',
            },
        ],
        functions=[
            {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            }
        ],
    )


def test_threads():
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_to_fn = {
            executor.submit(fn): fn
            for fn in [
                baserun.thread_wrapper(test_paris_chat_sync),
                baserun.thread_wrapper(test_egypt_chat_sync),
            ]
        }
        for future in concurrent.futures.as_completed(future_to_fn):
            response = future_to_fn[future]
            try:
                data = future.result()
            except Exception as exc:
                print("%r generated an exception: %s" % (response, exc))
            else:
                print("%r response is %s" % (response, data))
