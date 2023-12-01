from datetime import date, datetime
import enum
import guidance
import re
from react_chat import ReActChatGuidance
from time import sleep
from typing import Any, Dict, List
from transitions import Machine, State
from tools import dict_tools
import utils.env as env
from utils.gpt import COMPLETION_MODEL_3_5, COMPLETION_MODEL_4, extract_json_from_text_string, gpt_completion

# ==========================================================
# TREES OF THOUGHT
# https://huggingface.co/papers/2305.10601
# Trying to implement this idea, but the implementation details are super vague, so reproducing is hard.
# It seems as if they have a more complex agent/task running setup going
# ---
# Concepts to explore
# - breadth vs depth search of several plans (at which point does the system reflect)
# - How can this more closely reflect human exploration/planning
# ==========================================================

# ==========================================================
# ToT V3 (trying to get a bit more clarity/control with a state machine approach)
# ==========================================================

# came up with a way to do multiple generations in 1 go and get discrete vars for it
# self._convert_conversation_block_to_str({
#     'tag': 'geneach',
#     'num_iterations': 3,
#     'name': 'plans',
#     'content': self._convert_conversation_block_to_str({
#         'tag': str(CHAT_ROLES.ASSISTANT),
#         'content': "{{gen 'this.plan' temperature=0.7 max_tokens=600}}"
#     })
# })

class STATES(enum.Enum):
    INIT = 'INIT'
    PLANNING = 'PLANNING'
    REASONING = 'REASONING'
    ACTING = 'ACTING'
    FINAL = 'FINAL'

class CHAT_ROLES(enum.Enum):
    ASSISTANT = 'assistant'
    SYSTEM = 'system'
    USER = 'user'
    def __str__(self): # bc apparently you can't concat by default w/o this...
        return str(self.value)


class TreeChain(object):
    # MACHINE DEFS
    # HELPERS
    def _add_conversation_block(self, tag, content):
        self.conversation.append({ 'tag': str(tag), 'content': content, 'added_at': datetime.now() })

    def _convert_conversation_block_to_str(self, block):
        opening_tag = '{{#' + str(block.get('tag'))
        if block.get('name'):
            opening_tag = opening_tag + f" '{block.get('name')}'"
        if block.get('num_iterations'):
            opening_tag = opening_tag + f" num_iterations={block.get('num_iterations')}"
        opening_tag_closed = opening_tag + '~}}'
        closing_tag = '{{~/' + str(block.get('tag')) + '}}'
        return opening_tag_closed + '\n' + block.get('content') + '\n' + closing_tag
    
    def _convert_conversation_to_str(self):
        return '\n'.join(map(lambda block: self._convert_conversation_block_to_str(block), self.conversation))

    def _form_guidance_template(self, blocks, prefix_history=True):
        blocks_str = "\n".join(map(self._convert_conversation_block_to_str, blocks))
        if prefix_history == True:
            return self._convert_conversation_to_str() + '\n' + blocks_str
        return blocks_str

    def _is_not_done(self):
        return self.done != True

    def _response_to_dict(self, response_str):
        chunks = []
        # --- split on keyword
        for chunk in response_str.split('\n'):
            if re.match("^Thought|Action|Action Input|Observation|Final Answer:", chunk) != None:
                chunks.append(chunk)
            elif chunk != '':
                chunks[-1] = chunks[-1] + ' ' + chunk
        print(chunks) # pretty print does one str per line in the terminal
        print(len(chunks))
        # --- split str on arr
        for idx, string in enumerate(chunks):
            chunks[idx] = string.split(':',1)
        print(chunks) # pretty print does one str per line in the terminal
        print(len(chunks))
        # --- return
        return {
            # 'Thought': response_str[response_str.index('Thought: '):response_str.index('Action: ')].replace('Thought: ', '').strip(),
            # 'Action': response_str[response_str.index('Action: '):response_str.index('Action Input: ')].replace('Action: ', '').strip(),
            # 'Final Answer': response_str[response_str.index('Final Answer: '):].replace('Final Answer: ', ''),
        }


    # ACTIONS
    def _step_planning(self):
        # --- generate X plans given context/history
        plans = []
        while len(plans) < 1:
            print('planning...')
            planning_program = guidance(
                self._form_guidance_template([
                    { 'tag': str(CHAT_ROLES.USER), 'content': 'Observe the task. Then think out loud about it. If it requires creativity, be expressive in your thoughts. If it requires computation, be methodical in your thoughts.' },
                    { 'tag': str(CHAT_ROLES.ASSISTANT), 'content': "{{gen 'plan' temperature=0.9 max_tokens=600}}" },
                ]),
                llm=guidance.llms.OpenAI("gpt-3.5-turbo", token=env.env_get_open_ai_api_key(), caching=False)
            )
            planning_executed = planning_program()
            plans.append(planning_executed.variables()['plan'])
        plans_as_text = "\n---\n".join(f"Option: {idx + 1}\n\n{str}" for idx, str in enumerate(plans)) # 'choices' in language start at 1. I've seen it return 1 even in lists of 0 because it's the 'first' option
        # --- vote on plans
        vote_idx = 0
        tally = dict()
        while vote_idx < 1:
            print('voting...')
            voting_program = guidance(
                self._form_guidance_template([
                    { 'tag': str(CHAT_ROLES.USER), 'content': 'Analyize each choice in detail, and choose which will lead to the best output. Respond in JSON format with both the object keys "choice_integer" and "choice_reason".' },
                    { 'tag': str(CHAT_ROLES.USER), 'content': plans_as_text },
                    { 'tag': str(CHAT_ROLES.ASSISTANT), 'content': "{{gen 'plan' temperature=0.5 max_tokens=600}}" }
                ]),
                llm=guidance.llms.OpenAI("gpt-3.5-turbo", token=env.env_get_open_ai_api_key(), caching=False)
            )
            voting_executed = voting_program()
            # ... attempt to tally
            try:
                plan_str = voting_executed.variables().get('plan')
                plan_json = extract_json_from_text_string(plan_str)
                choice_idx = int(plan_json['choice_integer']) - 1 # subtracting 1 bc we added 1 in the choice text
                tally[choice_idx] = tally.get(choice_idx, 0) + 1
                vote_idx += 1 # TODO: determine if i should do this while
                print(f'Option #{plan_json["choice_integer"]}: ', plan_json['choice_reason'])
            except Exception as err:
                print('Couldnt Tally: ', err, voting_executed.variables())
        winning_plan_idx = max(tally)
        winning_plan = plans[winning_plan_idx]
        # --- add winner as a THOUGHT on an system block (trying it out on system? bc i don't want it to be reacted to like a conversation block)
        self._add_conversation_block(CHAT_ROLES.ASSISTANT, 'Observation: ' + winning_plan)
        # --- move to step
        self.next()

    def _step_reasoning(self):
        print('...')
        print('reasoning...')
        print('...')
        # generate thought (include new user prompted one, but hide it in future executions?)
        thought_program = guidance(
            self._form_guidance_template([
                { 'tag': str(CHAT_ROLES.USER), 'content': 'Think about what to do.' },
                { 'tag': str(CHAT_ROLES.ASSISTANT), 'content': "{{gen 'thought' temperature=0.7 max_tokens=1000}}" },
            ]),
            llm=guidance.llms.OpenAI("gpt-3.5-turbo", token=env.env_get_open_ai_api_key(), caching=False)
        )
        thought_executed = thought_program()
        print(thought_executed.variables().get('thought'))
        # ... if final answer
        # ... else act
        self._add_conversation_block(CHAT_ROLES.ASSISTANT, 'Thought: ' + thought_executed.variables().get('thought'))
        self.next()

    def _step_acting(self):
        print('...')
        print('stepping...')
        print(self._form_guidance_template([
                # { 'tag': str(CHAT_ROLES.USER), 'content': "State an Action and Action Input." }, # this failed so hard. forced an action and did chemistry during creative writing lol
                { 'tag': str(CHAT_ROLES.USER), 'content': 'What is the next Action, Action Input' },
                { 'tag': str(CHAT_ROLES.ASSISTANT), 'content': "{{gen 'next_response' temperature=0.7}}" },
            ]))
        print('...')
        # generate next assistant thought
        next_program = guidance(
            self._form_guidance_template([
                # { 'tag': str(CHAT_ROLES.USER), 'content': "State an Action and Action Input." }, # this failed so hard. forced an action and did chemistry during creative writing lol
                { 'tag': str(CHAT_ROLES.USER), 'content': 'What is the next Action, Action Input for the initial user question/task' },
                { 'tag': str(CHAT_ROLES.ASSISTANT), 'content': "{{gen 'next_response' temperature=0.7}}" },
            ]),
            llm=guidance.llms.OpenAI("gpt-3.5-turbo", token=env.env_get_open_ai_api_key(), caching=False)
        )
        next_executed = next_program()
        next_response = next_executed.variables().get('next_response')
        print(next_response)
        # evaluate returned gen (TODO: maybe we do a user input request if we're stuck? maybe just force final answer if stuck)
        # --- if we see a final answer: eval if its good enough (maybe there should be an input data type expectation?)
        # ------ if so save final data to machine
        # ------ if not, continue
        # --- re-run assistant step
        print(self.conversation)
        self.next()

    # STATES
    STATES = STATES
    # TODO: maybe do a hierarchical, or spawning, machine style so it can replicate and aim for specific outputs
    machine_states = [
        State(name=STATES.INIT, ), # system prompt starting point
        State(name=STATES.PLANNING, on_enter=['_step_planning']), # Tree-of-Thought step where we're generating paths and voting on them
        State(name=STATES.REASONING, on_enter=['_step_reasoning']), # ReAct
        State(name=STATES.ACTING, on_enter=['_step_acting']), # ReAct
        State(name=STATES.FINAL), # is there ever a final? Or is it just possible that we can provide a new prompt to build upon/continue us?
    ]

    # TRANSITIONS
    machine_transitions = [
        { 'trigger': 'next', 'source': STATES.INIT, 'dest': STATES.PLANNING },
        { 'trigger': 'next', 'source': STATES.PLANNING, 'dest': STATES.REASONING },
        { 'trigger': 'next', 'source': STATES.REASONING, 'dest': STATES.ACTING, 'conditions': ['_is_not_done'] },
        { 'trigger': 'next', 'source': STATES.REASONING, 'dest': STATES.FINAL },
        { 'trigger': 'next', 'source': STATES.ACTING, 'dest': STATES.REASONING },
    ]

    # INTERNAL CONTEXT/DATA
    conversation: List[Dict] = []
    tools = dict()
    system_prompt = 'You are a helpful assistant.'
    user_prompt = None
    done = False
    
    # INIT
    def __init__(self, system_prompt=None, tools=dict()) -> None:
        self.machine = Machine(
            model=self,
            states=self.machine_states,
            transitions=self.machine_transitions,
            initial=STATES.INIT
        )
        self.tools = tools
        # --- init convo
        self.system_prompt = system_prompt or self.system_prompt
        self._add_conversation_block(CHAT_ROLES.SYSTEM, system_prompt or self.system_prompt)

    def prompt(self, prompt):
        self.user_prompt = prompt
        self._add_conversation_block(CHAT_ROLES.USER, 'TASK: ' + prompt)
        # ... begin!
        self.next()
        return 'todo'


system_prompt_tools_text = "\n".join(map(lambda action_label: f"{action_label}: {dict_tools[action_label]['description']}", list(dict_tools.keys())))
system_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Current date: {date.today()}

### Instruction:

You have access to the following actions/tools:

{system_prompt_tools_text}

Strictly use the following format:

Thought: you should always think about what to do. do you know the final answer
Action: the action to take, should be one of [{", ".join(list(dict_tools.keys()))}]
Action Input: the input to the action, should be appropriate for tool input an d required
... (this Thought/Action/Action Input can repeat N times)
"""
user_prompt = "Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: 1. It isn't difficult to do a handstand if you just stand on your hands. 2. It caught him off guard that space smelled of seared steak. 3. When she didn't like a guy who was trying to pick her up, she started using sign language. 4. Each person who knows you has a different perception of who you are."
tot = TreeChain(system_prompt=system_prompt, tools=dict_tools)
response_tot = tot.prompt(user_prompt)

print(f'========== ToT Response ==========')
print(response_tot)
# print(f'========== IO Response ==========')
# response_io = gpt_completion(prompt=prompt, model=COMPLETION_MODEL_4)
# print(response_io)



# ==========================================================
# ToT V2
# ==========================================================

# def tot(prompt_task):
# 		print(f'\nPROMPT:\n{prompt_task}\n')
# 		# PLANS
# 		plans = []
# 		# --- generate
# 		while len(plans) < 3:
# 				print(f'GENERATING PLAN #{len(plans) + 1}...')
# 				agent = ReActChatGuidance(guidance, actions=dict_tools) # could be cool to do a quick LLM call to see what tools are relevant dict_tools
# 				query = f'I want the following task done: "{prompt_task}". Please don\'t do the task yet. Who is a lesser known world-class expert (past or present) who can do this task/topic?'
# 				agent_plan = agent.query(query=query, return_plan=True)
# 				plans.append(agent_plan)
# 		# --- format
# 		execution_plans = "\n\n-------\n\n".join(f"## CHOICE {idx} ##\n\n{str}" for idx, str in enumerate(plans))
# 		print('\n\n' + execution_plans)
# 		# VOTE
# 		votes = []
# 		while len(votes) < 5:
# 				print(f'VOTE #{len(votes) + 1}...')
# 				vote_response = gpt_completion(prompt=f'Analyizing each choice in detail, choose the best and respond in JSON format with keys "choice_integer" and "choice_reason". TASK: {prompt_task} \n\n {execution_plans}', model=COMPLETION_MODEL_3_5)
# 				votes.append(vote_response)
# 		# --- tally
# 		tally = dict()
# 		for vote_string in votes:
# 				json = extract_json_from_text_string(vote_string)
# 				choice = int(json['choice_integer'])
# 				tally[choice] = tally.get(choice, 0) + 1
# 				print(f'Voted {choice}: {json.get("choice_reason", "")[0:600]}...')
# 		print(f'\nTALLY:\n{tally}\n')
# 		# EXECUTE PLAN?
# 		highest_voted_plan = plans[max(tally)]
# 		final_execution = gpt_completion(
# 				prompt=f'{prompt_task} (Inspiration: {highest_voted_plan.replace("Final Answer:", "")})',
# 				model=COMPLETION_MODEL_3_5)
# 		# RETURN (select plan via index with highest int)
# 		return final_execution

# ==========================================================
# ToT V1
# ==========================================================

# def tot(prompt_task):
# 		print(f'\nPROMPT:\n{prompt_task}\n')
# 		# THOUGHT STEP #1: GET IMPLICIT NORTH STAR
# 		# Finding that I need to steer the plan statements a bit, so they don't disregard the writing task, but think more wholistcally
# 		# success_criteria = gpt_completion(
# 		# 	prompt=f'In a single sentence, describe a successful outcome for user in the following task. TASK: {prompt_task}',
# 		# 	model=COMPLETION_MODEL_4)
# 		# print(f'\nSUCCESS CRITERIA:\n{success_criteria}\n')
# 		# THOUGHT STEP #2: GENERATE PLANS (5)
# 		# v1 - repeated task with success 
# 		# prompt_plan_v1 = f'Do the following task. SUCCESS CRITERIA: {success_criteria} TASK: {prompt_task}'
# 		prompt_plan_v2 = f'Describe a way to be successful in the following task. TASK: {prompt_task}'
# 		plans = []
# 		while len(plans) < 3:
# 				print('Planning...')
# 				fetched_plan = gpt_completion(prompt=prompt_plan_v2, model=COMPLETION_MODEL_4)
# 				plans.append(fetched_plan)
# 		print(f'PLAN EXECUTIONS:\n')
# 		print("\n---\n".join(plans))
# 		print("\n\n")
# 		# THOUGHT STEP #3: VOTE (5)
# 		# --- generate
# 		votes = []
# 		while len(votes) < 5:
# 				print('Voting...')
# 				execution_plans = "\n\n".join(f"CHOICE {idx}) {str}" for idx, str in enumerate(plans))
# 				vote_response = gpt_completion(
# 					prompt=f'Analyizing each choice in detail in relation to the task, choose the best and respond in JSON format with keys "choice_integer" and "choice_reason". TASK: {prompt_task} \n\n {execution_plans}',
# 					model=COMPLETION_MODEL_3_5)
# 				votes.append(vote_response)
# 		# --- tally
# 		tally = dict()
# 		for vote_string in votes:
# 				try:
# 						json = extract_json_from_text_string(vote_string)
# 						choice = int(json['choice_integer'])
# 						print(f'Voted {choice}: {json.get("choice_reason", "")[0:300]}...')
# 						tally[choice] = tally.get(choice, 0) + 1
# 				except Exception as err: 
# 						print('No vote json/choice.', err)
# 		print(f'\nTALLY:\n{tally}\n')
# 		# THOUGHT STEP #4: Use highest vote as inspiration, write again
# 		highest_voted_plan = plans[max(tally)]
# 		final_execution = gpt_completion(
# 				prompt=f'Do the following task.\n\nINSPIRATION: {highest_voted_plan}\n\nTASK: {prompt_task}',
# 				model=COMPLETION_MODEL_4)
# 		# RETURN (select plan via index with highest int)
# 		return final_execution
		


# ==========================================================
# TEST: Creative Writing
# ==========================================================
# prompt = "Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: 1. It isn't difficult to do a handstand if you just stand on your hands. 2. It caught him off guard that space smelled of seared steak. 3. When she didn't like a guy who was trying to pick her up, she started using sign language. 4. Each person who knows you has a different perception of who you are."
# response_tot = tot(prompt)
# print(f'========== ToT Response ==========')
# print(response_tot)
# print(f'========== IO Response ==========')
# response_io = gpt_completion(prompt=prompt, model=COMPLETION_MODEL_4)
# print(response_io)

# Comparison of ToT and IO responses
# Result: ToT voted on a plan that presented the thread/idea of personal growth

# ToT  --> 
# The journey of personal growth is often paved with new experiences that teach us valuable lessons about ourselves and the world around us. As we face challenges and learn new skills, we become more versatile and resilient. One such example can be seen in the process of learning to do a handstand. With dedication and practice, one can go from barely being able to balance on their hands to confidently performing this impressive feat. It isn't difficult to do a handstand if you just stand on your hands.
# Sometimes, these lessons come from unexpected sources and can catch us completely off guard. An astronaut, venturing into the vastness of space for the first time, might anticipate many strange and wonderful sights. However, they might not expect the peculiar sensory experiences that await them. Floating weightlessly in the International Space Station, the astronaut noticed an odd aroma filling the airlock after a spacewalk. It caught him off guard that space smelled of seared steak.
# Creative problem-solving is another skill we often develop through personal experiences. Life has a way of presenting us with difficult situations that require quick thinking and adaptability. For instance, a woman at a bar might find herself dealing with the unwelcome advances of a man. Rather than confronting him outright or enduring the uncomfortable situation, she devises a clever solution to deter his attention. When she didn't like a guy who was trying to pick her up, she started using sign language.
# Ultimately, our experiences shape not only our skills and knowledge but also how others perceive us. As we navigate through life, each person we encounter will form their own unique understanding of who we are based on their interactions with us. Just as the handstand practitioner becomes an acrobat in the eyes of their friends, or the astronaut becomes a connoisseur of cosmic cuisine, each experience we have contributes to the multifaceted identity we present to the world. Each person who knows you has a different perception of who you are.

# IO --> 
# Learning to do a handstand can be an exciting and rewarding experience. It requires dedication, practice, and most importantly, confidence. To get started, find a suitable spot with plenty of space and a soft surface. Begin by positioning your hands shoulder-width apart and slowly lift your legs up into the air. With enough practice, it isn't difficult to do a handstand if you just stand on your hands.
# During a recent interview, an astronaut shared his intriguing experiences from his time in space. One peculiar observation he made was that the International Space Station had a distinct smell. He described it as a mixture of metal and something else distinctly familiar. It caught him off guard that space smelled of seared steak.
# In a crowded bar, a woman noticed a man attempting to approach her. She wasn't interested in his advances, so she decided to come up with a creative way to avoid engaging in conversation with him. Instead of simply turning him down, she acted as if she couldn't hear him. When she didn't like a guy who was trying to pick her up, she started using sign language.
# The way we perceive ourselves may not necessarily align with how others perceive us. Our friends, family, and acquaintances all have unique experiences and interactions with us that shape their individual understanding of our character. It is important to remember that, just like a kaleidoscope, each person who knows you has a different perception of who you are.

# ==========================================================

exit()
