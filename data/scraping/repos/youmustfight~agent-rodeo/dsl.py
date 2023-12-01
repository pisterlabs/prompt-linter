import guidance
import utils.env as env
from utils.gpt import extract_json_from_text_string

# ==========================================================
# SIMPLE JSON
# ==========================================================
def react_json() -> dict:
    llm = guidance.llms.OpenAI("text-davinci-003", token=env.env_get_open_ai_api_key(), caching=False)
    # FYI: "The OpenAI API does not support Guidance pattern controls! Please either switch to an endpoint that does, or don't use the `pattern` argument to `gen`."
    program = guidance("""The following is a order in JSON format.
    ```json
    {
        "name": "{{gen 'name' stop='"'}}",
        "age": {{gen 'age' stop=','}},
        "order": "{{select 'order' options=valid_dish}}",
        "delivery": "{{#select 'delivery'}}Yes{{or}}No{{/select}}",
        "amount": {{gen 'amount' stop='}'}}
    }
    ```""", llm=llm)
    # Execute with prompt inputs vars
    executed_program = program(valid_dish=["Pizza", "Noodlez", "Pho"])
    print('executed_program.text ->', executed_program.text)
    # Return
    return extract_json_from_text_string(executed_program.text)


# ==========================================================
# TEST: SIMPLE JSON
# ==========================================================
print(f'========== Guidance: JSON ==========')
response_react_json = react_json()
print(response_react_json)

exit()
