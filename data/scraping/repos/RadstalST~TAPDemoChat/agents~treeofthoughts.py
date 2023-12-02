
import concurrent.futures
import json
import os
import time
from abc import ABC, abstractmethod

import guidance
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

DATA_PATH = './data'
import argparse
import logging

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

api_key = os.getenv('OPENAI_API_KEY')

class AbstractLanguageModel(ABC):
    @abstractmethod
    def generate_thoughts(self, state, k):
        pass

    @abstractmethod
    def evaluate_states(self, states):
        pass


class CustomLanguageModel(AbstractLanguageModel):
    def __init__(self, model):
        self.model = model

    def generate_thoughts(self, state, k):
        #implement the thought generation logic using self.model
        pass

    def evaluate_states(self, states):
        #implement state evaluation logic using self.model
        pass
    
class CustomLanguageModel(AbstractLanguageModel):
    def generate_thoughts(self, state, k):
        # Example logic: generate k thoughts based on the provided state using self.model
        thoughts = self.model.generate(state, k)
        return thoughts

    def evaluate_states(self, states):
        # Example logic: evaluate provided states using self.model
        evaluations = [self.model.evaluate(state) for state in states]
        return evaluations
    
class OpenAILanguageModel(AbstractLanguageModel):
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", api_base="", api_model="", enable_ReAct_prompting=True):
        os.getenv("OPENAI_API_KEY")
        if api_key == "" or api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key != "":
            openai.api_key = api_key
        else:
            raise Exception("Please provide OpenAI API key")

        if api_base == ""or api_base is None:
            api_base = os.environ.get("OPENAI_API_BASE", "")  # if not set, use the default base path of "https://api.openai.com/v1"
        if api_base != "":
            # e.g. https://api.openai.com/v1/ or your custom url
            openai.api_base = api_base
            print(f'Using custom api_base {api_base}')

        if api_model == "" or api_model is None:
            api_model = os.environ.get("OPENAI_API_MODEL", "")
        if api_model != "":
            self.api_model = api_model
        else:
            self.api_model = "text-davinci-003"
        print(f'Using api_model {self.api_model}')

        self.use_chat_api = 'gpt' in self.api_model

        # reference : https://www.promptingguide.ai/techniques/react
        self.ReAct_prompt = ''
        if enable_ReAct_prompting:
            self.ReAct_prompt = "Write down your observations in format 'Observation:xxxx', then write down your thoughts in format 'Thoughts:xxxx'."

        self.strategy = strategy
        self.evaluation_strategy = evaluation_strategy

    def openai_api_call_handler(self, prompt, max_tokens, temperature, k=1, stop=None):
        while True:
            try:
                if self.use_chat_api:
                    messages = [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                    response = openai.ChatCompletion.create(
                        model=self.api_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                else:
                    response = openai.Completion.create(
                        engine=self.api_model,
                        prompt=prompt,
                        n=k,
                        max_tokens=max_tokens,
                        stop=stop,
                        temperature=temperature,
                    )
                with open("openai.logs", 'a') as log_file:
                    log_file.write("\n" + "-----------" + '\n' +"Prompt : "+ prompt+"\n")
                return response
            except openai.error.RateLimitError as e: #If there's a rate limit error, it will sleep for a specified time and then retry.
                sleep_duratoin = os.environ.get("OPENAI_RATE_TIMEOUT", 30)
                print(f'{str(e)}, sleep for {sleep_duratoin}s, set it by env OPENAI_RATE_TIMEOUT')
                time.sleep(sleep_duratoin)

    def openai_choice2text_handler(self, choice): #Processes the response choice (message or text) based on whether the chat API is being used.
        if self.use_chat_api:
            text = choice['message']['content']
        else:
            text = choice.text.strip()
        return text

    def generate_text(self, prompt, k):
        if self.use_chat_api:
            thoughts = []
            for _ in range(k):
                response = self.openai_api_call_handler(prompt, 1200, 0.5, k)
                text = self.openai_choice2text_handler(response.choices[0])
                thoughts += [text]
                print(f'thoughts: {thoughts}')
            return thoughts

        else:
            response = self.openai_api_call_handler(prompt, 1200, 0.5, k)
            thoughts = [self.openai_choice2text_handler(choice) for choice in response.choices]
            return thoughts

    def generate_thoughts(self, state, k, initial_prompt):
        if (type(state) == str):
            state_text = state
        else:
            state_text = '\n'.join(state)
        print("THIS IS WHERE IT GENERATE THE THOUGHTS BASING ON THE STATES:")
        print("We receive STATE of type", type(state), "For state: ", state, "\n\n")

        # prompt = f"Given the current state of reasoning: \n\n\n'{state_text}'\n\n\nGenerate the next best coherent thought to achieve the reasoning process and get the solution: "
        # prompt = f"Based on the current state of reasoning: \n\n\n'{state_text} Provide the next coherent thought that will help progress the reasoning process and reach an soluton "
        # prompt = f"These are the thoughts you've had: \n\n\n{state_text}, provide the next coherent thought that will help advance the reasoning process and reach an solution for this problem {initial_prompt}. Think sharply, think out of the box, predict failure. Do not leave any open questions. Unleash your mind."
        prompt = f"Considering the thoughts you've had until now: THE STATES ARE: \n\n{state_text}\n\nDevise the next coherent thought that will aid in advancing the reasoning process and achieving a solution to {initial_prompt}. Assess various scenarios, think unconventionally, anticipate potential challenges, and resolve any outstanding queries. Tap into your mind's full potential and make certain no open questions remain."

        prompt += self.ReAct_prompt
        print(prompt)
        thoughts = self.generate_text(prompt, k)

        # try comments for each thought generated.
        for idx, thought in enumerate(thoughts):
            # #Comment generation prompt.
            # comment_prompt = (f"Given the generated thought:\n\n{thought}\n\n"
            #               "Provide a brief comment or analysis regarding its relevance, quality, "
            #               "or any potential improvements that could be made.")
            # comment = self.generate_text(comment_prompt, 1)[0]
            print(f"Thought {idx + 1}: {thought}")

            # print(f"Thought {idx + 1}: {thought}\nComment: {comment}\n---")

        return thoughts

        # print(thoughts)
        print(f"Generated thoughts: {thoughts}")
        return thoughts


    def generate_solution(self, initial_prompt, state):
        if (type(state) == str):
            state_text = state
        else:
            state_text = '\n'.join(state)

        prompt = f"Considering the reasoning provided:\n\n'{state_text}'\n\nDevise the best possible solution for the task: {initial_prompt}"
        answer = self.generate_text(prompt, 1)
        # print(thoughts)
        print(f"General solution : {answer}")
        return answer

    def evaluate_states(self, states, initial_prompt):
        if self.evaluation_strategy == 'value':
            state_values = {}
            for state in states:
                state_text = ' '.join(state)
                print("We receive a state of type", type(state), "For state: ", state, "\n\n")
                prompt = f"Given the current state of reasoning: '{state_text}', evaluate its value as a float between 0 and 1, become very pessimistic think of potential adverse risks on the probability of this state of reasoning achieveing {initial_prompt} and DO NOT RESPOND WITH ANYTHING ELSE: OTHER THAN AN FLOAT"

                response = self.openai_api_call_handler(prompt, 10, 1)
                try:
                    value_text = self.openai_choice2text_handler(response.choices[0])
                    print(f'state: {value_text}')
                    value = float(value_text)
                    print(f"value: {value}")
                except ValueError:
                    value = 0  # Assign a default value if the conversion fails
                state_values[state] = value
            return state_values

        elif self.evaluation_strategy == 'vote':
            states_text = '\n'.join([' '.join(state) for state in states])

            prompt = f"Given the following states of reasoning, vote for the best state utilizing an scalar value 1-10:\n{states_text}\n\nVote, on the probability of this state of reasoning achieveing {initial_prompt} and become very pessimistic very NOTHING ELSE"

            response = self.openai_api_call_handler(prompt, 1200, 1)

            print(f'state response: {response}')

            best_state_text = self.openai_choice2text_handler(response.choices[0])

            print(f"Best state text: {best_state_text}")

            best_state = tuple(best_state_text.split())

            print(f'best_state: {best_state}')

            return {state: 1 if state == best_state else 0 for state in states}

        else:
            raise ValueError("Invalid evaluation strategy. Choose 'value' or 'vote'.")
        
class OptimizedOpenAILanguageModel(OpenAILanguageModel):
    #Constructor Method
    def __init__(self, api_key, strategy="cot", evaluation_strategy="value", cache_enabled=True, api_base="", api_model="", enable_ReAct_prompting=False):
        super().__init__(api_key, strategy, evaluation_strategy, api_base, api_model, enable_ReAct_prompting) #Calls the constructor of the parent class
        self.cache_enabled = cache_enabled #A boolean that toggles whether caching is enabled.
        self.thought_cache = {}
        self.state_evaluation_cache = {}
          #thought_cache and state_evaluarion_cache are dictionaries to cache results of thought generation and state evaluation, respectively, to prevent redundant calculations.
    def parallel_generate_thoughts(self, states, k): #generate thoughts for multiple states simultaneously.
        print(f"=== DEBUG ===\nStates: {states}, k: {k}")
        with concurrent.futures.ThreadPoolExecutor() as executor:
            thoughts = list(executor.map(lambda state: self.generate_thoughts(state, k), states))
            print(f"=== DEBUG ===\nGenerated thoughts: {thoughts}")
            # print(f"Parallel generated thoughts: {thoughts}")
        return thoughts

    def parallel_evaluate_states(self, states, initial_prompt):#this method also utilizes parallel processing, but for evaluating states.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            state_values = list(executor.map(self.evaluate_states, states, initial_prompt))
            print(f"Parallel evaluated state values: {state_values}")
        return state_values
    
class TreeofThoughts:
    """
    1. Thought Decomposition --> based on problem properties

    2. Thought Generator -> create a thought generator function G(p0, s, k) with 2 strategies a sample iid thoughts from a cot prompt b. propose thoughts
    sequentially using a propose prompt

    3. create a state evaluator function V(p0, S) with 2 strategies a value each state independently b. vote across states

    4. Choose a search algo based on tree structure [BFS or DFS]

    Implement chosen search algorithm for bfs (algo1):
        init S0 with the input x
        for t = 1 to T (step limit):
            generate candidate thoughts for each state in St-1
            eveluate the candiate states using the state evaluator V
            select the b most promising states for St

        return the final output by genertaing the thought for the best state in St for DFS(algo2)

        defien a recurseive DFS function with the current state s, step t, and other required params

        if t > T record the output by generating the thought for current state S

        for each candidate state s in the sorted list of generated thoughts for s:

            if the evaluated value of s is greater the the threshold of vth call the dfs function recursively
            with s and t + 1

    execute the chosen search algo with the input problem, thought generator, and state evaluator, and other required params
    """

    def __init__(self, model, search_algorithm):
        self.model = model
        self.search_algorithm = search_algorithm
        self.tree = {
            "nodes": [],
            "metrics": {
                "thoughts": [],
                "evaluations": []
            }
        }

    def solve(self, x, k=None, T=None, b=None, vth=None, timeout=None, confidence_threshold=None, max_iterations=None, convergence_threshold=None, convergence_count=None):
        #intended to find a solution to a problem instance x using the configured search algorithm (BFS or DFS) with other parameters.
        start_time = time.time()
        file_name = f"logs/tree_of_thoughts_output_{self.search_algorithm}.json"
        try:
            if self.search_algorithm == 'BFS':
                while timeout is None or time.time() - start_time < timeout:
                    result = self.tot_bfs(x, k, T, b) #b is number of promising states
                    if result:
                        self.save_tree_to_json(file_name)
                        return result
            elif self.search_algorithm == 'DFS':
                while timeout is None or time.time() - start_time < timeout:
                    result = self.tot_dfs(x, k, T, vth) #Value threshold for DFS
                    if result:
                        self.save_tree_to_json(file_name)
                        return result
            else:
                raise ValueError("Invalid search algorithm. Choose 'BFS' or 'DFS'.")
        except KeyboardInterrupt:
            logger.error("Keyboard interrupt detected.")
        except ValueError as e:
            logger.error(f"Error: {e}")
        finally:
            logger.info("Saving the current tree and metrics.")
            self.save_tree_to_json(file_name)



    def tot_bfs(self, x, k, T, b):
        S0 = {x}
        for t in range(1, T + 1):
            S0_t = set()
            for s in S0:
                for z in self.model.generate_thoughts(s, k, x):
                    if (type(s) == str):
                        S0_t.add((s, z))
                    else:
                        S0_t.add((*s, z))
            Vt = self.model.evaluate_states(S0_t, x)
            St = sorted(S0_t, key=lambda s: Vt[s], reverse=True)[:b]
            S0 = set(St)

            logger.info(f'Step: {t}, S0_t: {S0_t}, Vt: {Vt}, St: {St}, S0: {S0}')



        best_state = max(St, key=lambda s: Vt[s])

        return best_state


    def tot_dfs(self, x, k, T, vth, pruning_threshold=0.5, confidence_threshold=None, max_iterations=None, convergence_threshold=None, convergence_count=None):
        output = [] #List to store potential solutions (thoughts) and their evaluations.
        iteration_count = 0
        consecutive_convergence_count = 0
        prev_best_value = None
        file_name = f"logs/tree_of_thoughts_output_{self.search_algorithm}.json"


        def dfs(s, t): #A nested function to perform the recursive DFS. It takes s (the current state) and t (the current depth of search) as parameters.
            nonlocal consecutive_convergence_count, prev_best_value, iteration_count, output
            if t > T: #the search is too deep and must be curtailed. It generates a thought from the model for the current state s, evaluates it, and appends it along with its evaluation to output.
                thought = self.model.generate_thoughts(s, 1, x)
                print(f'thoughts inside dfs {thought}')

                value = self.model.evaluate_states({s}, x)[s]
                print(f'values inside dfs {value}')

                output.append((thought, value))
                print(f'output {output}')

                if confidence_threshold is not None and value >= confidence_threshold:
                    return True

                if prev_best_value is not None and convergence_threshold is not None:
                    if abs(value - prev_best_value) < convergence_threshold:
                        consecutive_convergence_count += 1
                    else:
                        consecutive_convergence_count = 0

                prev_best_value = value
                iteration_count += 1

                if (max_iterations is not None and iteration_count >= max_iterations) or (convergence_count is not None and consecutive_convergence_count >= convergence_count):
                    return True

                return False

            for s_prime in sorted(self.model.generate_thoughts(s, k, x)):
                state_value = self.model.evaluate_states({s_prime}, x)[s_prime]
                logger.info(f"State: {s_prime}, Value: {state_value}")

                if state_value > vth and (pruning_threshold is None or state_value >= pruning_threshold):
                    if (type(s) == str):
                        child = (s, s_prime)
                    else:
                        child = (*s, s_prime)
                    # self.tree['nodes'][child] = s
                    # self.tree["metrics"]["thoughts"][child] = s_prime
                    # self.tree["metrics"]["evaluations"][child] = state_value

                    if dfs(child, t + 1):
                        return True

            self.save_tree_to_json(file_name)
            return False


        dfs(x, 4)
        print(f'output  {output}')
        best_state = max(output, key=lambda x: x[1])
        return best_state[0]

    def save_tree_to_json(self, file_name): #Intended to save the current state of the tree to a JSON file.
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        with open(file_name, 'w') as json_file:
            json.dump(self.tree, json_file, indent=4)

    def print_tree(self, x, node=None, depth=0):
        if node is None:
            node = self.tree["nodes"][x]

        thought = self.tree["metrics"]["thoughts"][node]
        evaluation = self.tree["metrics"]["evaluations"][node]

        tree_info = {
            "node": node,
            "thought": thought,
            "evaluation": evaluation,
            "children": []
        }

        for child, parent in self.tree["nodes"].items():
            if parent == node:
                child_info = self.print_tree(child, depth + 1)
                tree_info["children"].append(child_info)

        return tree_info
    
class OptimizedTreeofThoughts(TreeofThoughts):
    def solve(self, x, k=None, T=None, b=None, vth=None, timeout=None, confidence_threshold=None, max_iterations=None, convergence_threshold=None, convergence_count=None):
        #k: number of thoughts, T: step limit, b = Number of most promising states, vth:Value threshold for DFS
        start_time = time.time()
        print(f'Start time {start_time}')
        if self.search_algorithm == 'BFS':
            while timeout is None or time.time() - start_time < timeout:
                result = self.tot_bfs(x, k, T, b)
                print(f'resultttt in optimized tree of thoughts: {result}')
                if result:
                    return result
        elif self.search_algorithm == 'DFS':
            while timeout is None or time.time() - start_time < timeout:
                result = self.tot_dfs(x, k, T, vth, confidence_threshold=confidence_threshold, max_iterations=max_iterations, convergence_threshold=convergence_threshold, convergence_count=convergence_count)
                if result:
                    return result
        else:
            raise ValueError("Invalid search algorithm. Choose 'BFS' or 'DFS'.")
        
        
def ask(question):
    search_algorithm = "DFS"
    strategy = "cot"
    evaluation_strategy="vote"

    #create instance
    model = OpenAILanguageModel(os.getenv("OPENAI_API_KEY"), api_model="gpt-3.5-turbo")
    tree_of_thoughts = OptimizedTreeofThoughts(model, search_algorithm)

    # input_problem = "using question from Dataset in HuggingFace"
    class args:
        problem = question
        search_algorithm = "DFS"
        k = 3
        T = 4
        b = 5
        vth = 0.4
        timeout = 10
        confidence = 0.8
        max_iterations = 40
        convergence_threshold = 0.01
        convergence_count = 5

    #solve the problem using the tree of thoughts class
    optimized_tree_of_thoughts = OptimizedTreeofThoughts(model, search_algorithm=args.search_algorithm)

    #solve the porblem using tree of thoughts problem helper
    best_state = optimized_tree_of_thoughts.solve(args.problem, k=args.k, T=args.T, b=args.b, vth=args.vth)


    #generate the final silution
    final_solution = optimized_tree_of_thoughts.model.generate_solution(best_state, args.problem)



    #print the final solutions
    print(f"THE FINAL SOLUTION IS: {final_solution}")
    return final_solution

    # trees = optimized_tree_of_thoughts.print_tree(final_solution)
