import os
import time
from itertools import islice
import dotenv
import guidance
from anytree import RenderTree


dotenv.load_dotenv()

# guidance.llm = guidance.llms.OpenAI("gpt-3.5-turbo-0613", api_key=os.getenv("OPENAI_API_KEY"), caching=False)
guidance.llm = guidance.llms.OpenAI("gpt-4-0613", api_key=os.getenv("OPENAI_API_KEY"), caching=False)

allowed_yes = ('y', 'yes')
allowed_no = ('n', 'no')
allowed_yes_no = allowed_yes + allowed_no


class ThoughtNode:
    def __init__(self, thought, parent=None, depth=0, horn_clauses=""):
        self.thought = thought
        self.parent = parent
        self.children = []
        self.depth = depth
        self.mistake_str = None
        self.mistake_evaluated = False
        self.is_mistake = None
        self.solution_str = None
        self.solution_evaluated = False
        self.is_solution = None
        self.horn_clauses = horn_clauses

    def __eq__(self, other):
        return self.thought == other.thought

    def get_previous_thoughts(self):
        previous_thoughts = []
        node = self
        while node.parent:
            previous_thoughts.append(node.parent.thought)
            node = node.parent
        return list(reversed(previous_thoughts))

    def get_previous_mistakes(self):
        previous_mistakes = []
        for child in self.children:
            if child.mistake_evaluated and child.is_mistake:
                previous_mistakes.append(child.mistake_str)
            previous_mistakes.extend(child.get_previous_mistakes())
        return previous_mistakes

    def generate_thought_extension(self):
        previous_thoughts = self.get_previous_thoughts()
        previous_mistakes = self.get_previous_mistakes()

        # if previous_thoughts:
        #     initial_question = previous_thoughts.pop(0)
        #     thought = self.thought
        # else:
        #     initial_question = self.thought
        #     thought = ''
        mistake_prompt = ""
        if previous_mistakes:
            print("adding previous mistakes to prompt...")
            double_newline = '\n\n' # f str helper
            mistake_prompt = "{{#user}}Thanks. Previous attempt(s) at extending these thoughts produced the following mistake(s).{{/user}}\n"
            mistake_prompt += "{{#user}}Mistakes:\n\n" + double_newline.join(previous_mistakes) + "\n\n{{/user}}"
            mistake_prompt += "{{#user}}Take care not to repeat these mistakes.{{/user}}"

        solution_prompt = ""
        if self.depth > 1:
            assert self.parent.solution_evaluated
            solution_prompt = "\n\n{{#user}}Thanks. Has a solution been reached?{{/user}}{{#assistant}}" + self.parent.solution_str + "\n\n{{/assistant}}"

        sysprompt = "You are skilled in problem solving and deduction. You reason step-by-step. You communicate efficiently with terse brevity."
        #         {{#user}}Represent the argument so far as a set of Horn clauses.{{/user}}
        #         {{#assistant}}{{gen 'horn_clauses' temperature=0.0 max_tokens=500}}\n{{/assistant}}
        # {{  # user}}{{initial_question}}\n\n{{/user}}
        thought_extension_prompt = guidance('''
        {{#system}}''' + sysprompt + '''{{/system}}
        {{#user}}Please state your thoughts so far.{{/user}}
        {{#assistant}}{{previous_thoughts}}\n\n{{thought}}\n\n{{/assistant}}
        ''' + solution_prompt + mistake_prompt + '''
        {{#user}}Represent the argument as a set of Horn clauses.{{/user}}
        {{#assistant}}{{gen 'horn_clauses' temperature=0.0 max_tokens=500}}\n{{/assistant}}
        {{#user}}Give the next reasoning step. You can give subsequent reasoning steps later if needed; focus on just the next step. Be strategic.{{/user}}
        {{#assistant}}Next step:{{/assistant}}
        {{#assistant}}{{gen 'next_step' temperature=0.0 max_tokens=300}}{{/assistant}}
        ''')
        thought_extension_eval = thought_extension_prompt(thought=self.thought, previous_thoughts='\n\n'.join(previous_thoughts), mistake_prompt=mistake_prompt, solution_prompt=solution_prompt)
        thought_extension = thought_extension_eval['next_step']
        horn_clauses = thought_extension_eval['horn_clauses']
        print(horn_clauses)
        new_thought_node = ThoughtNode(thought_extension, parent=self, depth=self.depth + 1, horn_clauses=horn_clauses)
        self.children.append(new_thought_node)
        return new_thought_node

    def evaluate_mistake(self):
        if self.mistake_evaluated:
            return self.is_mistake
        previous_thoughts = self.get_previous_thoughts()
        sysprompt = "You a highly logical and skeptical genius. You check carefully for mistakes."
        mistake_eval_prompt = guidance('''
        {{#system}}''' + sysprompt + '''{{/system}}
        {{#user}}This is a reasoned argument answering a question. It may be incomplete:{{/user}}
        {{#user}}{{previous_thoughts}}\n\n{{horn_clauses}}\n\n{{thought}}{{/user}}
        {{#user}}Explain whether or not there is a mistake in the (possibly unfinished) reasoning so far.{{/user}}
        {{#assistant}}{{gen 'mistake_evaluation' max_tokens=300 temperature=0.0}}{{/assistant}}
        {{#user}}To summarize, did the reasoning have a mistake? Answer y or n. If you are unsure, answer n.{{/user}}
        {{#assistant}}{{gen 'is_mistake' max_tokens=1 temperature=0.0}}{{/assistant}}''')
        mistake_eval = mistake_eval_prompt(thought=self.thought, previous_thoughts='\n\n'.join(previous_thoughts), horn_clauses=self.horn_clauses)
        is_mistake = mistake_eval['is_mistake'].lower()
        mistake_evaluation = mistake_eval['mistake_evaluation']
        # horn_clauses = mistake_eval['horn_clauses']
        # print(f"Mistake eval Horn clauses: {horn_clauses}")
        print(f'Mistake: {is_mistake}')
        print(f'Mistake evaluation: {mistake_evaluation}')
        assert is_mistake in allowed_yes_no
        self.is_mistake = is_mistake in allowed_yes
        if self.is_mistake:
            self.mistake_str = f"Mistake: {self.thought}\n\nMistake Explanation: {mistake_evaluation}"
        self.mistake_evaluated = True
        return self.is_mistake

    def evaluate_solution(self):
        if self.solution_evaluated:
            return self.is_solution
        previous_thoughts = self.get_previous_thoughts()
        # You check solutions for completeness and correctness. You communicate tersely and value brevity and efficiency.
        sysprompt = "You are an expert logician. You check solutions for completeness and correctness."
        solution_eval_prompt = guidance('''
        {{#system}}''' + sysprompt + '''{{/system}}
        {{#user}}Given these thoughts: "{{previous_thoughts}}\n\n{{thought}}"\n\nWas a complete solution given? Why or why not?{{/user}}
        {{#assistant}}{{gen 'solution_evaluation' max_tokens=500 temperature=0.0}}{{/assistant}}
        {{#user}}In summary, was the original question clearly and completely answered? Answer y or n.{{/user}}
        {{#assistant}}{{gen 'is_solution' max_tokens=1 temperature=0.0}}{{/assistant}}
        ''')
        solution_eval = solution_eval_prompt(thought=self.thought, previous_thoughts='\n\n'.join(previous_thoughts))
        is_solution = solution_eval['is_solution'].lower()
        solution_evaluation = solution_eval['solution_evaluation']
        print(f'Found solution: {is_solution}')
        print(f'Solution evaluation: {solution_evaluation}')
        assert is_solution in allowed_yes_no
        self.solution_str = solution_evaluation
        self.is_solution = is_solution in allowed_yes
        self.solution_evaluated = True
        return self.is_solution




MAX_DEPTH = 8
MAX_WIDTH = 4


class SolutionFound(Exception):
    pass

def explore(node, depth=0):
    if depth > MAX_DEPTH:
        return
    print(f'--- Current Depth: {depth}/{MAX_DEPTH} ---')
    print('Current thought: ')
    print(node.thought)
    print()
    if depth > 0:
        if node.evaluate_mistake():
            print('--- Mistake detected, skipping current thought. ---')
            return

        if node.evaluate_solution():
            print("*** Solution found: ***")
            print('\n'.join(node.get_previous_thoughts() + [node.thought]))
            print()
            raise SolutionFound("Solution has been found, stopping further exploration.")

    if depth < MAX_DEPTH:
        for i in range(MAX_WIDTH):
            print(f'--- Generating extension {i + 1}/{MAX_WIDTH} ---')
            extension = node.generate_thought_extension()
            explore(extension, depth + 1)
            if not extension.mistake_str:
                # if no mistake detected and got here, then max depth was exceeded
                extension.mistake_str = f"Mistake: {node.thought}" + f"Mistake Explanation: subsequent thoughts did not reach a solution."
                print("updated extension mistake str")


# Using the search
question = """
Use the input numbers and basic arithmetic operations + - * / and parentheses to write an expression that evaluates to 24. The expression must use each number exactly as many times as it appears in the input.
Input: 4 4 6 8
"""

question = """
We meet three people, A, B, and C, one of whom is a knight, one a knave, and one a spy. The knight always tells the truth, the knave always lies, and the spy can either lie or tell the truth.

A says: "C is a knave."
B says: "A is a knight."
C says: "I am the spy."

Who is the knight, who the knave, and who the spy?
"""
root = ThoughtNode(question)


try:
    explore(root)
except SolutionFound as e:
    print('Solution found, stopping exploration.')
