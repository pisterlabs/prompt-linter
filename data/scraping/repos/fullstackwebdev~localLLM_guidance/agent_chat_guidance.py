# import guidance

valid_answers = ['Action', 'Final Answer']
valid_tools = ['Google Search']

chatgpt_test = """
{{#system~}}
You are a helpful assistant.
{{~/system}}

{{#user~}}
Question: {{query}}
Who are 3 world-class experts (past or present) who would be great at answering this?
Please don't answer the question or comment on it yet. Don't ask me to clarify or rephrase the question. Just list the experts.
{{~/user}}

{{#assistant~}}
Three world-class experts who would be great at answering this are {{gen 'experts' temperature=0 max_tokens=300}}
{{~/assistant}}

{{#user~}}
Great, now please answer the question as if these experts had collaborated in writing a joint anonymous answer.
In other words, their identity is not revealed, nor is the fact that there is a panel of experts answering the question.
If the experts would disagree, just present their different positions as alternatives in the answer itself (e.g. 'some might argue... others might argue...').
Please start your answer with ANSWER:
{{~/user}}

{{#assistant~}}
ANSWER: {{gen 'resolver' temperature=0.7 max_tokens=500}}
{{~/assistant}}
"""

class ChatGPTAgentGuidance:
    def __init__(self, guidance, tools, num_iter=3):
        self.guidance = guidance
        self.tools = tools
        self.num_iter = num_iter

    # def do_tool(self, tool_name, actInput):
    #     return self.tools[tool_name](actInput)
    
    def __call__(self, query):
        prompt_start = self.guidance(chatgpt_test)
        final_response = prompt_start(query=query)
        history = final_response.__str__()
        return history, final_response['resolver']

# secondquery = prompt_start(question=query)
# history += secondquery.__str__()
# thirdquery = prompt_start(question=query)
# history += thirdquery.__str__()

# prompt_reflection = self.guidance(reflection_researchers_template)
# final_response = prompt_reflection(question=query, response1=firstquery['steps'], response2=secondquery['steps'], response3=thirdquery['steps'])
# history += final_response.__str__()
# # prompt_final = self.guidance(prompt_final_template)
# # final_response = prompt_final(history=reflection, valid_answers=valid_answers, valid_tools=valid_tools)