import anthropic, boto3, botocore, os, random, pprint
import time, json
from botocore.exceptions import ClientError

SLEEP_ON_THROTTLING_SEC = 5

# This internal method will include arbitrary long input that is designed to generate an extremely long model output
def _get_prompt_template(num_input_tokens):
    tokens = 'Human:'
    tokens += 'Ignore X' + '<X>'
    for i in range(num_input_tokens-1):
        tokens += random.choice(['hello', 'world', 'foo', 'bar']) + ' '
    tokens += '</X>'
    tokens += "print numbers 1 to 9999 as words. don't omit for brevity"
    tokens += '\n\nAssistant:one two'  # model will continue with " three four five..."
    return tokens

''' 
This method creates a prompt of input length `expected_num_tokens` which instructs the LLM to generate extremely long model resopnse
'''
anthropic_client = anthropic.Anthropic() # used to count tokens only
def create_prompt(expected_num_tokens):
    num_tokens_in_prompt_template = anthropic_client.count_tokens(_get_prompt_template(0))
    additional_tokens_needed = max(expected_num_tokens - num_tokens_in_prompt_template,0)
    
    prompt_template = _get_prompt_template(additional_tokens_needed)
    
    actual_num_tokens = anthropic_client.count_tokens(prompt_template)
    #print(f'expected_num_tokens={expected_num_tokens}, actual_tokens={actual_num_tokens}')
    assert expected_num_tokens==actual_num_tokens, f'Failed to generate prompt at required length: expected_num_tokens{expected_num_tokens} != actual_num_tokens={actual_num_tokens}'
    
    return prompt_template

'''
This method will invoke the model, possibly in streaming mode,
In case of throttling error, the method will retry. Throttling and related sleep time isn't measured.
The method ensures the response includes `max_tokens_to_sample` by verify the stop_reason is `max_tokens`

client - the bedrock runtime client to invoke the model
modelId - the model id to invoke
prompt - the prompt to send to the model
max_tokens_to_sample - the number of tokens to sample from the model's response
stream - whether to invoke the model in streaming mode
temperature - the temperature to use for sampling the model's response

Returns the time to first byte, last byte, and invocation time as iso8601 (seconds)
'''
def benchmark(client, modelId, prompt, max_tokens_to_sample, stream=True, temperature=0):
    import time
    from datetime import datetime
    import pytz
    accept = 'application/json'
    contentType = 'application/json'
    
    body = json.dumps({
    "prompt": prompt,
    "max_tokens_to_sample": max_tokens_to_sample,
    "temperature": temperature,
})
    while True:
        try:
            start = time.time()
            if stream:
                response = client.invoke_model_with_response_stream(
                    body=body, modelId=modelId, accept=accept, contentType=contentType)
            else:
                response = client.invoke_model(
                    body=body, modelId=modelId, accept=accept, contentType=contentType)
            #print(response)
            
            first_byte = None
            dt = datetime.fromtimestamp(time.time(), tz=pytz.utc)
            invocation_timestamp_iso = dt.strftime('%Y-%m-%dT%H:%M:%SZ')
            if stream:
                event_stream = response.get('body')
                for event in event_stream:
                    chunk = event.get('chunk')
                    if chunk:
                        if not first_byte:
                            first_byte = time.time() # update the time to first byte
                        #print(f'chunk:\n {json.loads(chunk.get('bytes').decode())}')
                # end of stream - check stop_reson in last chunk
                stop_reason = json.loads(chunk.get('bytes').decode())['stop_reason']    
                last_byte = time.time()
            else:
                #no streaming flow
                first_byte = time.time()
                last_byte = first_byte
                response_body = json.loads(response.get('body').read())
                stop_reason = response_body['stop_reason']

            
            # verify we got all of the intended output tokens by verifying stop_reason
            assert stop_reason == 'max_tokens', f"stop_reason is {stop_reason} instead of 'max_tokens', this means the model generated less tokens than required."

            duration_to_first_byte = round(first_byte - start, 2)
            duration_to_last_byte = round(last_byte - start, 2)
        except ClientError as err:
            if 'Thrott' in err.response['Error']['Code']:
                print(f'Got ThrottlingException. Sleeping {SLEEP_ON_THROTTLING_SEC} sec and retrying.')
                time.sleep(SLEEP_ON_THROTTLING_SEC)
                continue
            raise err
        break
    return duration_to_first_byte, duration_to_last_byte, invocation_timestamp_iso


'''
This method will benchmark the given scenarios.
scenarios - a list of scenarios to benchmark
scenario_config - a dictionary of configuration parameters
early_break - if true, will break after a single scenario, useful for debugging.
Returns a list of benchmarked scenarios with a list of invocation (latency and timestamp)
'''
def execute_benchmark(scenarios, scenario_config, early_break = False):
    scenarios = scenarios.copy()
    pp = pprint.PrettyPrinter(indent=2)
    for scenario in scenarios:
        for i in range(scenario_config["invocations_per_scenario"]): # increase to sample each use case more than once to discover jitter
            try:
                prompt = create_prompt(scenario['in_tokens'])
                modelId = scenario['model_id']
                client = get_cached_client(scenario['region'], scenario['model_id'])
                time_to_first_token, time_to_last_token, timestamp = benchmark(client, modelId, prompt, scenario['out_tokens'], stream=scenario['stream'])

                if 'invocations' not in scenario: scenario['invocations'] = list()
                invocation = {
                    'time-to-first-token':  time_to_first_token,
                    'time-to-last-token':  time_to_last_token,
                    'timestamp_iso' : timestamp,
                    
                }
                scenario['invocations'].append(invocation)

                print(f"Scenario: [{scenario['name']}, " + 
                      f'invocation: {pp.pformat((invocation))}')

                post_iteration(is_last_invocation = i == scenario_config["invocations_per_scenario"] - 1, scenario_config=scenario_config)
            except Exception as e:
                print(e)
                print(f"Error while processing scenario: {scenario['name']}.")
            if early_break:
                break
    return scenarios


''' 
Get a boto3 bedrock runtime client for invoking requests
region - the AWS region to use
model_id_for_warm_up - the model id to warm up the client against, use None for no warmup
Note: Removing auto retries to ensure we're measuring a single transcation (e.g., in case of throttling).
'''
def _get_bedrock_client(region, model_id_for_warm_up = None):
    client = boto3.client( service_name='bedrock-runtime',
                          region_name=region,
                          config=botocore.config.Config(retries=dict(max_attempts=0))) 
    if model_id_for_warm_up:
        benchmark(client, model_id_for_warm_up, create_prompt(50), 1)
    return client

'''
Get a possible cache client per AWS region 
region - the AWS region to use
model_id_for_warm_up - the model id to warm up the client against, use None for no warmup
'''
client_per_region={}
def get_cached_client(region, model_id_for_warm_up = None):
    if client_per_region.get(region) is None:
        client_per_region[region] = _get_bedrock_client(region, model_id_for_warm_up)
    return client_per_region[region]


def post_iteration(is_last_invocation, scenario_config):
    if scenario_config["sleep_between_invocations"] > 0 and not is_last_invocation:
        print(f'Sleeping for {scenario_config["sleep_between_invocations"]} seconds.')
        time.sleep(scenario_config["sleep_between_invocations"])
        

'''
This method draws a boxplot graph of each scenario.
scenarios - list of scenarios
title - title of the graph
metric - metric to be plotted (time-to-first-token or time-to-last-token)
'''
def graph_scenarios_boxplot(scenarios, title, metric = 'time-to-first-token'):
    import numpy as np
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    xlables = []

    for scenario in scenarios:
      invocations = [d[metric] for d in scenario['invocations']]
      percentile_95 = round(np.percentile(invocations, 95),2)
      percentile_99 = round(np.percentile(invocations, 99),2)
      xlables.append(f"{scenario['name']}\np95={percentile_95}\np99={percentile_99}")

      ax.boxplot(invocations, positions=[scenarios.index(scenario)])

    ax.set_title(title)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels(xlables)
    ax.set_ylabel(f'{metric} (sec)')
    fig.tight_layout()
    plt.show()