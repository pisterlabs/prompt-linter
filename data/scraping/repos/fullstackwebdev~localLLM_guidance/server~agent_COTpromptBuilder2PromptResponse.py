# import guidance
import json

chatgpt_chain_of_thought_prompt_builder = '''
{{#block hidden=True}}
{{#system~}}
You are a helpful assistant.
{{~/system}}
{{#user~}}
Can you help me how to write the perfect prompt when I want to ask you more complex queries?
{{~/user}}
{{#assistant~}}
{{gen 'joke' max_tokens=250 stop=''}}
{{~/assistant}}
{{#user~}}
Great! Do you know the concept of Chain of Thought for a prompt?
{{~/user}}
{{#assistant~}}
{{gen 'joke' max_tokens=250 stop=''}}
{{~/assistant}}
{{#user~}}
Great! How would now a specific CoT-prompt look like, if I want to query you exactly about this topic, how would I formulate such a CoT prompt?
{{~/user}}
{{#assistant~}}
Sure, what subject?
{{~/assistant}}
{{#user~}}
{{query}}  And I want to know how to write the perfect prompt, not a list. Make it a paragraph or two.
{{~/user}}
{{#assistant~}}
"{{~gen 'resolver' temperature=0.7 max_tokens=450 stop='\\n\\n'~}}
{{~/assistant}}
{{#user~}}
Rephrase it but add more Chain Of Thought to it.
{{~/user}}
{{#assistant~}}
"{{~gen 'COTprompt' temperature=0.7 max_tokens=450 stop='\\n\\n'~}}
{{~/assistant}}
{{~/block~}}
{{#user~}}
{{COTprompt}}
{{~/user}}
{{#assistant~}}
Certainly!{{~gen 'resolver' temperature=0.7 max_tokens=450 stop=''~}}
{{~gen 'resolver' temperature=0.7 max_tokens=450 stop=''~}}
{{~gen 'resolver' temperature=0.7 max_tokens=450 stop=''~}}
{{~/assistant}}

'''

def default_serialize(obj):
    return str(obj)

class COTpromptBuilder2PromptResponse:
    def __init__(self, guidance, tools, num_iter=3):
        self.guidance = guidance
        self.tools = tools
        self.num_iter = num_iter
    
    def __call__(self, query):
        prompt_start = self.guidance(chatgpt_chain_of_thought_prompt_builder)
        final_response = prompt_start(query=query)
        history = final_response.__str__()

        resolver = final_response['resolver']
        print(query)
        print(resolver)
        return history, resolver #pretty_json #final_response.variables().__str__()






