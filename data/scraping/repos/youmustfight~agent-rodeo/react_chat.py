from datetime import date
import pydash as _
import re
import guidance
from tools import dict_tools
from tools.llms import choose
import utils.env as env
from utils.gpt import COMPLETION_MODEL_3_5, gpt_completion

# ==========================================================
# ReAct (reason, action)
# https://gartist.medium.com/a-simple-agent-with-guidance-and-local-llm-c0865c97eaa9
# https://til.simonwillison.net/llms/python-react-pattern
# https://github.com/QuangBK/localLLM_guidance/blob/main/demo_ReAct.ipynb
# Recreating this concept just to get familiar with it
# ==========================================================

dict_actions = dict_tools

# ==========================================================
# AGENT
# ==========================================================
class ReActChatGuidance():
    def __init__(self, guidance, actions, caching=False):
        self.guidance = guidance
        self.llm = guidance.llms.OpenAI(COMPLETION_MODEL_3_5, token=env.env_get_open_ai_api_key(), caching=caching) # gpt-4
        self.actions = actions
        # --- prompts
        react_prompt_system_actions_text = "\n".join(map(lambda action_label: f"{action_label}: {dict_actions[action_label]['description']}", list(dict_actions.keys())))
        self.react_prompt_system = "{{#system~}}" + f"""
        Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request and end with a Final Answer. Current date: {date.today()}

        ### Instruction:

        Complete the objective as best you can. You have access to the following actions/tools:

        {react_prompt_system_actions_text}

        Strictly use the following format:

        Thought: you should always think about what to do
        Action: the action to take, should be one of [{", ".join(list(dict_actions.keys()))}]
        Action Input: the input to the action, should be appropriate for tool input an d required
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)

        Thought: do you know the final answer
        Final Answer: the final answer output to the original input question

        """ + "{{~/system}}"
        self.react_prompt_init_user = """
        {{#user~}}
        Question: {{query}}
        {{~/user}}
        """
        self.react_prompt_progress_assistant = """
        {{history}}
        {{#assistant~}}
        {{gen 'fn' temperature=0.7}}
        {{~/assistant}}
        """

    def fn_action(self, action_label, action_input, history=None):
        return self.actions[action_label]['func'](action_input, history)
    
    def query(self, query, return_plan=False, init_history=None):
        chat_prompt_init = self.guidance("\n".join([self.react_prompt_system, self.react_prompt_init_user]), llm=self.llm)
        chat_start = chat_prompt_init(query=query)
        chat_progressing = chat_start # (gets replaced on each cycle, just initial setting)
        history = init_history or chat_progressing.text
        # --- assistant config
        assistant_cycles_num = 0
        assistant_cycles_max = 20
        # --- RUN
        while assistant_cycles_num <= assistant_cycles_max:
            print(f'assistant cycle #{assistant_cycles_num}...')
            chat_prompt_assistant = self.guidance(self.react_prompt_progress_assistant, llm=self.llm)
            chat_progressing = chat_prompt_assistant(history=history)
            chat_progressing_recent_assistant_text = chat_progressing.text[chat_progressing.text.rindex('<|im_start|>assistant'):]
            # ... allow just returning the initial plan (trimming system text)
            if return_plan == True:
                return chat_progressing.text[chat_progressing.text.index('<|im_start|>user'):]
            # ... in the most recent assistant cycle, if we have an "Action", execute it with a tool so we can insert the true value
            if 'Action:' in chat_progressing_recent_assistant_text or assistant_cycles_num == 0:
                # TODO: make this more flexible where we can halt any action evaluation, for when we'd want a ToT approach where it'd vote on paths suggested
                assistant_text_arr = chat_progressing_recent_assistant_text.split('\n')
                val_rexep = re.compile(': (.+)')
                first_action_thought_idx  = _.find_index(assistant_text_arr, lambda txt: 'Thought' in txt)
                print(assistant_text_arr[first_action_thought_idx])
                first_action_idx  = _.find_index(assistant_text_arr, lambda txt: 'Action' in txt) # not doing 'Action:' because we can sometimes get numeric suffixes, like 'Action 1:'
                first_action_name  = val_rexep.findall(assistant_text_arr[first_action_idx])[0]
                first_action_input_idx  = _.find_index(assistant_text_arr, lambda txt: 'Action Input' in txt)
                first_action_input = val_rexep.findall(assistant_text_arr[first_action_input_idx])
                # TODO: wtf am i doing this vvv ???
                if first_action_input:
                    first_action_input = first_action_input[0]
                else: 
                    first_action_input = None
                # ... if we have a valid tool
                # print(assistant_text_arr)
                if dict_actions.get(first_action_name) != None:
                    # ... replace the end of the history with output and CoT can continue w/ updated info
                    first_action_output = self.fn_action(first_action_name, first_action_input, history)
                    updated_agent_block = '\n'.join([
                        f'{assistant_text_arr[first_action_thought_idx]}',
                        f'Action: {first_action_name}',
                        f'Action Input: {first_action_input}',
                        f'Action Output: {first_action_output}',
                    ])
                    # ... and then append an observation on this output TODO: should do another turbo completion to be cheap/faster
                    observation = gpt_completion(prompt=f'''ACTION:\n{updated_agent_block}\n\nQUERY: the result of the action. state nothing if there was no output''')
                    print('Observation: ', observation, '\n', updated_agent_block)
                    updated_agent_block = updated_agent_block + f'\nObservation: {observation}'
                    history = chat_progressing.text[:chat_progressing.text.rindex('<|im_start|>assistant')] + "<|im_start|>assistant\n" + updated_agent_block + "\n<|im_end|>\n"
            # ... if it was just a final answer, and no action, then be done
            elif 'Final Answer' in chat_progressing_recent_assistant_text:
                history = chat_progressing.text
                # ... our final answer can sometimes be a composition over multiple actions/observations, need to find a way to guarentee a "final answer" (in v1, I was just returning the final answer, but that can sometimes just be a meta-response)
                final_answer = history[history.rindex('Final Answer:'):].replace('Final Answer:', '').strip()
                final_action_output = history[history.rindex('Action Output:'):history.rindex('Observation:')].replace('Action Output:', '').strip()
                try:
                    choice = choose(query, [final_action_output, final_answer])
                    clean_choice = choice.replace('<|im_end|>', '')
                    return clean_choice, history
                except Exception as err:
                    print('final_answer', final_answer)
                    print('final_action_output', final_action_output)
                    raise err # return final_answer, history
            else:
                print('\nWARNING: Looping through without Action or Final Answer\n') # history
            # ... increment and run again!
            assistant_cycles_num += 1
            # ... if we hit max cycles, just throw we should have had a response by now
            if assistant_cycles_num == assistant_cycles_max:
                raise 'Max assistant cycles with no final answer.'



if __name__ == "__main__":
# ==========================================================
# TEST: CALCULATOR
# ==========================================================

    print(f'========== ReAct Response: Tools - Search & Math ==========')
    agent = ReActChatGuidance(guidance, actions=dict_actions)
    prompt = "Whats does 24 + 17 + ((2 + 2) / 2) * 100 - 2 * 100 equal? What is the current Bronx Borough President's age? What's the difference between both numbers?"
    response_react_calculations, history = agent.query(prompt)
    print(f'========== ReAct Response: Tools - Search & Math = Result ==========')
    print(response_react_calculations)

    # print(f'========== ReAct Response: Tools - Search & Writing ==========')
    # agent = ReActChatGuidance(guidance, actions=dict_actions)
    # prompt = "You are Karl Marx and you are speaking at the MET Gala. Research the recent theme and write a speech. Return the speech\'s text itself."
    # response_react_writing, conversation = agent.query(prompt)
    # print(f'========== ReAct Response: Tools - Search & Writing = Result ==========')
    # print(response_react_writing)

    print(f'========== ReAct Response: Tools - Search & Chemistry ==========')
    agent = ReActChatGuidance(guidance, actions=dict_actions)
    prompt = "Name a molecule with low solubility? What is it's SMILES code? What is that molecule's SELFIES code?"
    response_react_chemistry, conversation = agent.query(prompt)
    print(f'========== ReAct Response: Tools - Search & Chemistry = Result ==========')
    print(response_react_chemistry)

    exit()
