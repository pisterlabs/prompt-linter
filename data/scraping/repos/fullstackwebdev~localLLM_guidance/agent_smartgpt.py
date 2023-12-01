# import guidance

prompt_start_template = """
{{question}}

Let's work this out in a step by step way to be sure we have the right answer
{{gen 'steps' max_tokens=250 }}
"""

reflection_researchers_template = """
{{question}}

Answer 1:
    {{response1}}

Answer 2:
    {{response2}}

Answer 3:
    {{response3}}

You are a researcher tasked with investigating the X response options provided. List the flaws and faulty logic of each answer option. Let's work this out in a step by step way to be sure we have all the errors:
{{gen 'reflection'}}

You are a resolver tasked with 1) finding which of the X answer options the researcher thought was best 2) improving that answer, and 3) Printing the improved answer in full. Let's work this out in a step by step way to be sure we have the right answer:

researcher thought was the best answer: 
{{gen 'resolver' temperature=0.7 max_tokens=250 stop='\\n\\n'}}
"""

class AgentGuidanceSmartGPT:
    def __init__(self, guidance, tools, num_iter=3):
        self.guidance = guidance
        self.tools = tools
        self.num_iter = num_iter

    # def do_tool(self, tool_name, actInput):
    #     return self.tools[tool_name](actInput)
    
    def __call__(self, query):
        prompt_start = self.guidance(prompt_start_template)
        firstquery = prompt_start(question=query)
        history = firstquery.__str__()
        secondquery = prompt_start(question=query)
        history += secondquery.__str__()
        thirdquery = prompt_start(question=query)
        history += thirdquery.__str__()

        prompt_reflection = self.guidance(reflection_researchers_template)
        final_response = prompt_reflection(question=query, response1=firstquery['steps'], response2=secondquery['steps'], response3=thirdquery['steps'])
        history += final_response.__str__()
        # prompt_final = self.guidance(prompt_final_template)
        # final_response = prompt_final(history=reflection, valid_answers=valid_answers, valid_tools=valid_tools)
        return history, final_response['resolver']
