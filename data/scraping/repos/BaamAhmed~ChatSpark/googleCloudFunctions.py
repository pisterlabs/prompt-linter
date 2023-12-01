import functions_framework
import json
from cohere import Client

co = Client('') # INSTERT API KEY HERE

def create_response(names, conversation, intent):
    intent_prompt = ''
    if intent == 'mentorship':
        intent_prompt = f'he wants to ask {names["target"]} for mentorship'
    elif intent == 'inquiry':
        intent_prompt = f'he wants to ask {names["target"]} about a job posting'
    elif intent == 'recruiter':
        intent_prompt = f'he wants to invite {names["target"]} to apply for an open job'
    else:
        intent_prompt = 'he just wants to keep the conversation going professionally'
    response = co.generate(
        model='command',
        prompt=f'This is a conversation between {names["user"]} and {names["target"]}\n\n{conversation}\n\n Considering the above dialog, generate a message that {names["user"]} should send if ' + intent_prompt,
        max_tokens=302,
        temperature=0.9,
        k=47,
        stop_sequences=[],
        return_likelihoods='NONE')
  
    return response.generations[0].text

def make_msg(target, myself=''):
    
    target_profile_sum = co.summarize(
        text=json.dumps(target),
        model='command-light',
        temperature=0.5, 
        length='long',
        extractiveness='medium'
    )

    if intent == "mentorship":
        # add "names['user'] intends to ask names['target'] for mentorship and advice" to prompt
        if len(myself) == 0:
            prompt_text = f"Write a short and personal message that expresses my interest in receiving mentorship and advice from the following individual: {target_profile_sum.summary}"
        else:
            prompt_text = f"Given my profile: {json.dumps(myself)}, write a short and personal message that expresses my interest in receiving mentorship and advice from the following individual: {target_profile_sum.summary}"
    elif intent == "connect":
        # add "names['user'] intends to request names['target'] to connect on LinkedIn" to prompt
        if len(myself) == 0:
            prompt_text = f"Write a short and personal message that expresses my interest in connecting with the following individual on LinkedIn: {target_profile_sum.summary}"
        else:
            prompt_text = f"Given my profile: {json.dumps(myself)}, write a short and personal message that expresses my interest in connecting with the following individual on LinkedIn: {target_profile_sum.summary}"
    elif intent == "inquiry":
        # add "names['user'] intends to ask names['target'] about a job posting" to prompt
        if len(myself) == 0:
            prompt_text = f"I want to inquire about a job posting. Write a short message to this individual expressing this fact: {target_profile_sum.summary}"
        else:
            prompt_text = f"Given my profile: {json.dumps(myself)}, write a short message to this individual to inquire about a job posting: {target_profile_sum.summary}"
    elif intent == "recruiter":
        # add "names['user'] intends to scout names['target'] for a new role in their company" to prompt
        if len(myself) == 0:
            prompt_text = f"I am a recruiter. Write a short message to the following individual to express interest in recruiting them: {target_profile_sum.summary}"
        else:
            prompt_text = f"Given my profile as a recruiter: {json.dumps(myself)}, write a short message to the following individual to express interest in recruiting them:  {target_profile_sum.summary}"

    prompt_text = 'All responses must be strictly less than 40 words, with no exceptions. ' + prompt_text
    response = co.generate(
        prompt=prompt_text,
        model='command-light',
        temperature=0.7, # higher value means more creative, or random
        max_tokens=120,
        k=47,
        stop_sequences=[],
        truncate='END',
    )

    return response[0].text

def create_response(conversation):
    response = co.generate(
        model='command',
        prompt=f'This is a conversation between {names["user"]} and {names["target"]}\n\n{conversation}\n\nConsider the above dialog, please generate a message that {names["user"]} should send if he wants to ask {names["target"]} for a referral.',
        max_tokens=302,
        temperature=0.9,
        k=47,
        stop_sequences=[],
        return_likelihoods='NONE')

    return response.generations[0].text

def get_summary(target):
    target_profile_sum = co.summarize(
        text=json.dumps(target),
        temperature=0.5, 
        length='long',
        extractiveness='high'
    )
    return target_profile_sum

@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    # Set CORS headers for the preflight request
    if request.method == "OPTIONS":
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }

        return ("", 204, headers)

    # Set CORS headers for the main request
    headers = {"Access-Control-Allow-Origin": "*"}

    request_json = request.get_json(silent=True)
    request_args = request.args
    print(request_args)
    print(request_json)

    final_msg = make_msg(request_json)
    
    headers = {"Access-Control-Allow-Origin": "*"}
    print(final_msg)
    return (final_msg, 200, headers)
