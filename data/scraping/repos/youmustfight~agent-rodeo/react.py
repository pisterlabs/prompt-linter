import guidance
import tools.calculate
import tools.serps_search
import utils.env as env
from utils.gpt import COMPLETION_MODEL_4, gpt_completion

# ==========================================================
# ReAct (reason, action)
# https://gartist.medium.com/a-simple-agent-with-guidance-and-local-llm-c0865c97eaa9
# https://til.simonwillison.net/llms/python-react-pattern
# https://github.com/QuangBK/localLLM_guidance/blob/main/demo_ReAct.ipynb
# Recreating this concept just to get familiar with it
# ==========================================================

# ==========================================================
# TOOLS
# ==========================================================

dict_tools = {
    'Calculator': tools.calculate,
    'WebSearch': tools.serps_search
}
valid_tools = list(dict_tools.keys())

# ==========================================================
# PROMPT TEMPLATES
# ==========================================================
FINAL_ANSWER = 'Final Answer'
valid_answers = ['Action', FINAL_ANSWER]
# when I removed 'Answer', it seems to strongly sway towards 'Final Answer' and it'd skip tool calls

react_prompt_template_start = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Complete the objective as best you can. You have access to the following tools:

Calculator: Runs a calculation for math computation - uses Python so be sure to use floating point syntax if necessary (example input: 4 * 7 / 3)
WebSearch: Runs a web search and returns JSON data based on search results using Google/Bing Search. Ideal for getting timely and current information (example input: 'What is the age of the president?')

Strictly use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Calculator, WebSearch]
Action Input: the input to the action, should be appropriate for tool input
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

For examples:
Question: How old is CEO of Microsoft wife?
Thought: First, I need to find who is the CEO of Microsoft.
Action: Google Search
Action Input: Who is the CEO of Microsoft?
Observation: Satya Nadella is the CEO of Microsoft.
Thought: Now, I should find out Satya Nadella's wife.
Action: Google Search
Action Input: Who is Satya Nadella's wife?
Observation: Satya Nadella's wife's name is Anupama Nadella.
Thought: Then, I need to check Anupama Nadella's age.
Action: Google Search
Action Input: How old is Anupama Nadella?
Observation: Anupama Nadella's age is 50.
Thought: I now know the final answer.
Final Answer: Anupama Nadella is 50 years old.

### Input:
{{question}}

### Response:
Question: {{question}}
Thought: {{gen 't1' stop='\\n'}}
{{select 'answer' options=valid_answers}}: """
# FYI: this was totally breaking when I had a line break with that """. LINE BREAKS ARE VERY IMPORTANT for flow & regex

react_prompt_template_middle = """{{history}}{{select 'tool_name' options=valid_tools}}
Action Input: {{gen 'actInput' stop='\\n'}}
Observation: {{fn_tool tool_name actInput}}
Thought: {{gen 'thought' stop='\\n'}}
{{select 'answer' options=valid_answers}}: """

react_prompt_template_final = """{{history}}{{select 'tool_name' options=valid_tools}}
Action Input: {{gen 'actInput' stop='\\n'}}
Observation: {{fn_tool tool_name actInput}}
Thought: {{gen 'thought' stop='\\n'}}
{{select 'answer' options=valid_answers}}: {{gen 'fn' stop='\\n'}}"""


# ==========================================================
# AGENT
# ==========================================================

class ReActGuidance():
    def __init__(self, guidance, tools, max_iters=3):
        self.guidance = guidance
        self.llm = guidance.llms.OpenAI("text-davinci-003", token=env.env_get_open_ai_api_key()) # gpt-4 # gpt-3.5-turbo
        self.max_iters = max_iters
        self.tools = tools

    def fn_tool(self, tool_name, actInput):
        return self.tools[tool_name](actInput)

    def query(self, query):
        # upon call, do initial system chat
        prompt_init = self.guidance(react_prompt_template_start, llm=self.llm)
        result_start = prompt_init(question=query, valid_answers=valid_answers)

        # then repeatedly continue calling guidance (each call template has a {{history}} string which prefixes prior convo)
        result_mid = result_start
        for _ in range(self.max_iters - 1):
            # ... if we hit a final answer, 
            if result_mid['answer'] == FINAL_ANSWER:
                break
            history = result_mid.__str__() # grab previous syntax/text so we can preced the next call with this
            prompt_mid = self.guidance(react_prompt_template_middle, llm=self.llm)
            result_mid = prompt_mid(history=history, fn_tool=self.fn_tool, valid_answers=valid_answers, valid_tools=valid_tools)

        # upon being done or hiting a final answer, resolve the prompt
        if result_mid['answer'] != FINAL_ANSWER:
            # ... if we didn't get a final answer, force the conclusion
            history = result_mid.__str__()
            prompt_mid = self.guidance(react_prompt_template_final, llm=self.llm)
            result_final = prompt_mid(history=history, fn_tool=self.fn_tool, valid_answers=[FINAL_ANSWER], valid_tools=valid_tools)
        else:
            # ... otherwise looks like we got final answer! prompt w/ stop token and grab fin
            history = result_mid.__str__()
            prompt_mid = self.guidance(history + "{{gen 'fn' stop='\\n'}}", llm=self.llm)
            result_final = prompt_mid()

        # print('history', history)
        # print('result_final', result_final)
        return result_final['fn']



if __name__ == "__main__":
    # ==========================================================
    # TEST: CALCULATOR
    # ==========================================================
    print(f'========== ReAct Response: Tools - Calculator ==========')
    agent = ReActGuidance(guidance, dict_tools)
    prompt_calculation = 'Whats does 24 + 17 + ((2 + 2) / 2) * 100 - 5 * 65.5 equal? Provide just a number as a final answer.'
    print(prompt_calculation)
    # For this prompt, if I don't include "provide just a number as final answer", the GPT 3.5/4 will do CoT. Probably embedded in their prompt.
    # However, when it's just a final answer response, the IO 3.5/4 gets it wrong and returns a differnet value every time
    response_react = agent.query(prompt_calculation)
    response_io = gpt_completion(prompt=prompt_calculation, model=COMPLETION_MODEL_4)
    print('Response ReAct: ', response_react)
    print('Response IO: ', response_io)

    # # ==========================================================
    # # TEST: WEB SEARCH
    # # ==========================================================
    # print(f'========== ReAct Response: Tools - Web Search ==========')
    # agent = ReActGuidance(guidance, dict_tools)
    # # prompt_calculation = 'Who is the current president of the United Nations General Assembly?'
    # # prompt_calculation = 'How old is the president of the United State\'s wife?'
    # prompt_calculation = ' I need to find out the number 8 of Manchester United'
    # response_react = agent(prompt_calculation)
    # response_io = gpt_completion(prompt=prompt_calculation, model=COMPLETION_MODEL_4)
    # print('Response ReAct: ', response_react)
    # print('Response IO: ', response_io)

    exit()
