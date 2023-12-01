from fastapi import FastAPI, Request
import openai
import random
from aoai_proxy_config import USERS, AOAI_ENDPOINTS, COMPLETION_TOKEN_COST, PROMPT_TOKEN_COST

# For each user and enpoint we'll keep track of how many tokens they've used.
for _, user in USERS.items():
    user['usage'] = {
        'total_completion_tokens': 0,
        'total_prompt_tokens': 0,
    }
for endpoint in AOAI_ENDPOINTS:
    endpoint['usage'] = {
        'total_completion_tokens': 0,
        'total_prompt_tokens': 0,
    }

# Configure the OpenAI Python client to use Azure endpoints.
openai.api_type = 'azure'
openai.api_version = '2023-03-15-preview'

app = FastAPI()

@app.post("/openai/deployments/gpt-35-turbo/chat/completions")
async def chat_completion(request: Request):
    """Open AI proxy endpoint for chat completions.

    This endpoint has exactly the same signature as the OpenAI API endpoint.
    We use it to authenticate the user, choose an endpoint, and track usage.
    """

    # "Authenticate" the user by finding it based on the API key.
    user = [
        USERS[user]
        for user in USERS
        if USERS[user]['api_key'] == request.headers['api-key']
    ][0]

    # Choose an endpoint at random out of the list of available Azure OpenAI endpoints.
    endpoint = random.choice(AOAI_ENDPOINTS)
    openai.api_key = endpoint['api_key']
    openai.api_base = endpoint['base_url']

    # Get the body of the request, which we'll pass on to the OpenAI API.
    request_body = await request.json()
    completion = openai.ChatCompletion.create(
        deployment_id='gpt-35-turbo',
        messages=request_body['messages'],
    )

    # Track usage. Take the usage reported in the response and add it to the user and endpoint.
    completion_tokens = completion['usage']['completion_tokens']
    prompt_tokens = completion['usage']['prompt_tokens']
    user['usage']['total_completion_tokens'] += completion_tokens
    user['usage']['total_prompt_tokens'] += prompt_tokens
    endpoint['usage']['total_completion_tokens'] += completion_tokens
    endpoint['usage']['total_prompt_tokens'] += prompt_tokens

    # Return the response from the OpenAI API.
    return completion

@app.get("/usage")
async def get_usage():
    """Get usage statistics.
    
    Returns a JSON object with usage and cost for each user and endpoint
    and the total cost for all users and endpoints.
    """
    users_usage = {}
    for name, user in USERS.items():
        users_usage[name] = {
            'total_completion_tokens': user['usage']['total_completion_tokens'],
            'total_prompt_tokens': user['usage']['total_prompt_tokens'],
            'total_cost': (
                (user['usage']['total_completion_tokens'] * COMPLETION_TOKEN_COST) +
                (user['usage']['total_prompt_tokens'] * PROMPT_TOKEN_COST)
            ),
        }
    endpoints_usage = {}
    for i, endpoint in enumerate(AOAI_ENDPOINTS):
        endpoints_usage[f'endpoint {i + 1}'] = {
            'total_completion_tokens': endpoint['usage']['total_completion_tokens'],
            'total_prompt_tokens': endpoint['usage']['total_prompt_tokens'],
            'total_cost': (
                (endpoint['usage']['total_completion_tokens'] * COMPLETION_TOKEN_COST) +
                (endpoint['usage']['total_prompt_tokens'] * PROMPT_TOKEN_COST)
            ),
        }
    total_cost = sum([users_usage[user]['total_cost'] for user in users_usage])
    return {
        'users': users_usage,
        'endpoints': endpoints_usage,
        'total_cost': total_cost,
    }