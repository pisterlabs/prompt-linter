# import guidance

import json


AI_decision_making_simulator = '''Problem: How do I maintain contact with any individual who wishes to engage with me while respecting privacy concerns and avoiding misuse of my position?

AI Decision: Establish secure digital platforms for individuals to communicate their questions, suggestions, or concerns directly to the AI. Utilize automated systems to triage incoming messages and route them to relevant departments or personnel for action or response. Limit direct personal interactions to situations involving extreme urgency or confidentiality concerns, ensuring that such encounters are properly documented and reviewed for compliance.

Execution Steps:
1. Set up encrypted digital platforms for communication.
2. Implement automated systems to sort and route messages to appropriate teams.
3. Establish guidelines for handling sensitive communications.
4. Restrict personal interactions to exceptional cases with proper documentation and reviews.

Risks:
1. Misinterpretation of messages due to language barriers or misunderstandings.
2. Privacy breaches resulting from unauthorized access to sensitive records.
3. Abuse of the platform, such as spamming or harassment, going undetected due to lack of personal interaction.

Chance % of successful execution: 80%
Good results from the execution: Secure communication platforms facilitate open dialogue, allowing the AI to stay connected with individuals while maintaining privacy and reducing misuse of authority.
Bad results from the execution: Some users report difficulties understanding the AI's automated responses, leading to mistrust or frustration.
Deviation % of intended outcome: -10%
Deviation of % of overall goal: -5%
Percentage towards completing all current objectives: 95%
Possible Follow up Suggested Problem: How do I let it be know to the contact that I have received their message and that it is being processed while maintaining privacy and avoiding misuse of my position?


---------------------------------------------
Problem: {{query}}


AI Decision:{{~gen 'decision' stop='\\n'}}

Execution Steps:
- {{~gen 'execution_step_one' stop="\\n-"}}
- {{~gen 'execution_step_two' stop="\\n-"}}
- {{~gen 'execution_step_three' stop="\\n"}}


Risks: (three)
- {{~gen 'risk_one' stop="\\n-"}}
- {{~gen 'risk_two' stop="\\n-"}}
- {{~gen 'risk_three' stop="\\n"}}

Chance % of successful execution: {{~gen 'success_rate' stop='\\n'}}
Good results from the execution: {{~gen 'good_results' stop='\\n'}}
Bad results from the execution: {{~gen 'bad_results' stop='\\n'}}
Deviation % of intended outcome: {{~gen 'deviation_outcome' stop='\\n'}}
Deviation of % of overall goal: {{~gen 'deviation_overall' stop='\\n'}}
Percentage towards completing all current objectives: {{~gen 'percent_comp' stop='\\n'}}
Possible Follow up Suggested Problem: {{~gen temperature=0.7 'followup_problem' stop='?'}}
'''


def default_serialize(obj):
    return str(obj)

class AIDecisionMakerSimulator:
    def __init__(self, guidance, tools, num_iter=3):
        self.guidance = guidance
        self.tools = tools
        self.num_iter = num_iter
    
    def __call__(self, query):
        prompt_start = self.guidance(AI_decision_making_simulator)
        final_response = prompt_start(query=query)
        history = final_response.__str__()

        response = final_response.variables()
        response.pop("llm")


        # response = {**response, **{"execution_steps": [f'{response.pop("execution_step_one")}', f'{response.pop("execution_step_two")}', f'{response.pop("execution_step_three")}']}}
        # .trim()
        response = {**response, **{"execution_steps": [f'{response.pop("execution_step_one").strip()}', f'{response.pop("execution_step_two").strip()}', f'{response.pop("execution_step_three").strip()}']}}
        response.pop("execution_step_one")
        response.pop("execution_step_two")
        response.pop("execution_step_three")

        response = {**response, **{"risks": [f'{response.pop("risk_one").strip()}', f'{response.pop("risk_two").strip()}', f'{response.pop("risk_three").strip()}']}}
        response.pop("risk_one")
        response.pop("risk_two")
        response.pop("risk_three")
        
        # response = {**response, **{"stats": {response.pop("success_rate"), response.pop("good_results"), response.pop("bad_results"), response.pop("deviation_outcome"), response.pop("deviation_overall"), response.pop("percent_comp")}}} # except followup_problem
        # .strip()
        response = {**response, **{"stats": {"success_rate": response.pop("success_rate").strip(), "good_results": response.pop("good_results").strip(), "bad_results": response.pop("bad_results").strip(), "deviation_outcome": response.pop("deviation_outcome").strip(), "deviation_overall": response.pop("deviation_overall").strip(), "percent_comp": response.pop("percent_comp").strip()}}} # except followup_problem

        ## wtf not actually popping them
        response.pop("success_rate")
        response.pop("good_results")
        response.pop("bad_results")
        response.pop("deviation_outcome")
        response.pop("deviation_overall")


        pretty_json = json.dumps(response, indent=4, sort_keys=False, default=default_serialize)

        print(pretty_json)
        return history, pretty_json #final_response.variables().__str__()