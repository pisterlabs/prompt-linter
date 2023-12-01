import guidance
import utils.env as env
from utils.gpt import extract_json_from_text_string, gpt_completion

# query can be a str or str[]
def writing(query_string, history):
    print('[TOOL] writing', type(query_string), query_string)
    # ... if we get no input, then indicate back to the LLM this is unacceptable
    try:
        prompt_str = query_string.split('",')[0]
        context_str = query_string.split('",')[1:]
        # ... complete if we can grab prompt/context str
        res = gpt_completion(f"""
        PROMPT: {prompt_str}

        CONTEXT: {context_str}

        RESPONSE: """)
    except Exception as err:
        print('[TOOL] writing err -> ', err)
        # ... otherwise just do a prompt
        res = gpt_completion(f"""
        PROMPT: {query_string}

        CONTEXT: {history[-2000:]}

        RESPONSE: """)
    
    print('[TOOL] writing -> ', res)
    return res


def choose(criteria, choices) -> dict:
    print('[TOOL] choose', criteria, len(choices))
    print('[TOOL] choose: choices', choices)
    # TODO: replace w/ cheaper/smaller model
    llm = guidance.llms.OpenAI("text-davinci-003", token=env.env_get_open_ai_api_key(), caching=False)
    # FYI: "The OpenAI API does not support Guidance pattern controls! Please either switch to an endpoint that does, or don't use the `pattern` argument to `gen`."
    choices_as_str = "\n\n".join(f"CHOICE {idx}: {str}" for idx, str in enumerate(choices))
    program = guidance("""
    Which choice satisfies the stated goal: {{criteria}}

    {{choices}}

    Choice Number: '{{gen 'choice_num' stop="'"}}'
    """, llm=llm)
    # Execute with prompt inputs vars
    executed_program = program(criteria=criteria, choices=choices_as_str)
    print('[TOOL] choose ->', executed_program.variables()['choice_num'])
    # Return
    choice_num = int(executed_program.variables()['choice_num'])
    return choices[choice_num]
