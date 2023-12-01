"""
    Giving an AI the ability to improve its own underlying architecture through an evolutionary algorithm is, besides being poetically 
    beautiful, also a very promising paradigm. This paper is heavily based on the EvoPrompt paper by Angelica Chen David M. Dohan and David R. So.
    The original soft promted tuned a PALM 62B model. Since I dont have access to this model i instead finetune gpt3. Which is an expensive endeavour, but 
    very cool nevertheless. 
"""

import concurrent.futures
import json
import os
import random

import numpy as np
import openai

openai.api_key = "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"

class EvoPrompting:
    def __init__(self, lm, task, seed_folder, environment, T, m, k, n, p, alpha, 
                 n_evaluations, target_model_size, target_episodes, seed_evaluation=False, evaluation_path=None):
        self.seed_folder = seed_folder # Folder where the seed codes are located
        self.seed_evaluation = seed_evaluation # Do we have to evaluate the seed codes?
        self.pre_evaluated_seed_metrics = self.load_pre_evaluated_seed_metrics(evaluation_path) # Pre evaluated seed metrics
        self.lm = lm # the crossover LM
        self.temperatures = [0.2, 0.6, 0.8, 1.0] # uniformly sample from these temperaturs
        self.environment = environment # In our case CartPole-v1
        self.T = T # Number of rounds
        self.m = m # number of few-shot prompts per round
        self.n = n # number of samples to generate per prompt,
        self.k = k # number of in-context examples per prompt
        self.p = p # number of survivors to select per generation
        self.n_evaluations = n_evaluations # Number of times to run each model
        self.alpha = alpha # the upper threshold for the test error
        self.global_population = [] # Global historical Population

        self.target_model_size = target_model_size # Target model size of the few shot prompt
        self.target_episodes = target_episodes # Target number of episodes of the few shot prompt
        
        # Set initial well designed architectures as parent models.
        # (Evaluate them useing the same eval function as used in the aalgo)
        self.current_population = []
        self.initialize_population()
    

    def read_seed_files(self, file_path):
        with open(file_path, "r") as file:
            return file.read()


    def load_pre_evaluated_seed_metrics(self, file_path):
        with open(file_path, "r") as file:
            return json.load(file)


    def initialize_population(self):
        # Initialize the population with seed architectures
        # List all the Python files in the seed folder
        seed_files = [f for f in os.listdir(self.seed_folder) if f.endswith('.py')]

        for seed_file in seed_files:
            print("EVALUATING SEED: ", seed_file)
            seed_file_path = os.path.join(self.seed_folder, seed_file)
            seed_code = self.read_seed_files(seed_file_path)

            if self.seed_evaluation:
                avg_episodes, model_size = self.eval_t(seed_code)
            else:
                json= self.pre_evaluated_seed_metrics[seed_file]
                # convert string to float           
                avg_episodes = float(json["avg_episodes"])
                model_size = float(json["model_size"])

            print("EVALUATED SEED: ", seed_file, "avg_episodes: ", avg_episodes, "model_size: ", model_size)
            metrics = {
                "avg_episodes": avg_episodes,
                "model_size": model_size,
            }
            
            fitness_score = avg_episodes * model_size
            self.global_population.append((seed_code, metrics, fitness_score))
            self.current_population.append((seed_code, metrics, fitness_score))
        

    def make_few_shot_prompt(self, in_context_examples):
        # Create a few-shot prompt using the in context examples E
        min_avg_episodes = float('inf')
        min_model_size = float('inf')
        prompt = "" # Initialize empty prompt string

        for example in in_context_examples:
            metrics = example[1]
            min_avg_episodes = min(min_avg_episodes, metrics['avg_episodes']) # Retrieve the minium avg episodes of the parent architectures
            min_model_size = min(min_model_size, metrics['model_size']) # Retrieve the minium model size of the parent architectures
            prompt += f'\nMetrics: {example[1]}\n\n'
            prompt += f'Code: {example[0]}\n\n'

        target_avg = min_avg_episodes * self.target_episodes
        target_model_size = min_model_size * self.target_model_size

        prompt += f'\nmetrics: {{ "avg_episodes": {target_avg}, "model_size": {target_model_size} }}\n\n'
        prompt += f'Code:\n'

        return prompt


    def generate_child (self, prompt):
        child_code = openai.Completion.create(
            model="gpt-4",
            prompt=prompt,
            temperature=np.random.choice(self.temperatures, size=1, replace=True).item(),
            n=1,
            max_tokens = 1000,
        )
        #print("child code= ", child_code.choices[0].text)
        return child_code.choices[0].text
        

    def eval_t(self, code_segment):
        def single_evaluation():
            print("Executing code segment")
            exec(code_segment, globals())  # Add globals() here
            episodes, model_size = globals()['main'](self.environment)
            print(f"Finished executing code segment: episodes={episodes}, model_size={model_size}")
            return episodes, model_size

        sum_episodes = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            print("Submitting tasks to the thread pool")
            futures = [executor.submit(single_evaluation) for _ in range(self.n_evaluations)]
            for future in concurrent.futures.as_completed(futures):
                episodes, model_size = future.result()
                sum_episodes += episodes

        avg_episodes = sum_episodes / self.n_evaluations
        print(f"Average episodes: {avg_episodes}, Model size: {model_size}")
        return avg_episodes, model_size


    def get_top(self, global_population):
        """
        Returns the top entries from the global_population based on their fitness scores.

        This function takes a list of global_population entries, where each entry is a tuple containing:
        (code, metadata, fitness_score). It sorts the entries based on their fitness scores in descending
        order and returns the top num_top entries.

        Parameters:
        global_population (list): A list of tuples, where each tuple represents an entry in the global
                                population, containing (code, metadata, fitness_score).
        num_top (int, optional): The number of top entries to return. Defaults to 5.

        Returns:
        list: A list containing the top num_top entries from the global_population based on their fitness
            scores.
        """
        sorted_population = sorted(global_population, key=lambda x: x[2], reverse=True)
        top_entries = sorted_population[:self.p]
        return top_entries


    def cross_mutation(self):
        child_architectures = [] # C is the set of architectures of length k
        for _ in range(self.m): # create m number of few shot prompts
            in_context_examples = random.sample(self.current_population, self.k) # Pick k amount of parants from P
            prompt = self.make_few_shot_prompt(in_context_examples)
            Ci = [self.generate_child(prompt) for _ in range(self.n)]
            child_architectures.extend(Ci)
        return child_architectures


    def fitness_function(self, model_size, n_episodes):
        return model_size * n_episodes


    def filter_and_eval(self, child_architectures, environment, alpha):
        CEVALED = []
        for code_segment in child_architectures:
            avg_episodes, model_size = self.eval_t(code_segment)
            if avg_episodes < alpha: # filter out the bad models
                metrics = {
                    "avg_episodes": avg_episodes,
                    "model_size": model_size,
                }
                fitness_score = self.fitness_function(model_size, avg_episodes)
                CEVALED.append((code_segment, metrics, fitness_score))
        return CEVALED
    

    def train(self, CEVALED):
        # The original author of the paper proposes a soft prompt tune method here
        # I need a model here that can be soft promt tuned, probably gpt2 on huggingface.
        pass

    def evolve(self):
        t = 0
        while t < self.T: # number of evoluationary rounds
            child_architectures = self.cross_mutation() # Generate the set of code samples
            evaluated_children = self.filter_and_eval(child_architectures, self.environment, self.alpha)
            self.global_population.extend(evaluated_children)

            if t < self.T - 1:
                self.current_population = self.get_top(global_population=self.global_population)
                #run without training
                #self.lm = self.train(self.lm, [c for c, _ in evaluated_children if c not in self.current_population])
            
            t += 1 

        return self.get_top(global_population=self.global_population)

if __name__ == "__main__":
    # Initialize the EvoPrompting class
    T = 10 # Number of rounds
    m = 10 # number of few-shot prompts per round
    n = 16 # number of samples to generate per prompt,
    k = 2 # number of in-context examples per prompt
    p = 1 # number of survivors to select per generation
    n_evaluations = 5 # Number of times to run each model
    alpha = 600000 # TBD (cutoff fitness for evaluated children)
    task = "create a solution that genreates the best model with the smallest paramter size"
    environment = "CartPole-v1" # environment of the task
    seed_folder = "seeds" # Folder which contains al the initial seed architectures
    lm = "text-davinci-003" # Language model to use for prompt generation

    target_model_factor = 0.90 
    target_episodes = 0.95

    evo_prompt = EvoPrompting(lm, task, seed_folder, environment, T, m, k, n, p, alpha,
                              n_evaluations, target_model_factor, target_episodes, seed_evaluation=True,
                              evaluation_path="seeds/pre_evaluated_seed_metrics.json")
    # Run the main evolutionary loop
    evo_prompt.evolve()

    # evo_prompt.initialize_population()
    # print("evorpompt Global Population: ", evo_prompt.global_population)

    # top = evo_prompt.get_top(global_population = evo_prompt.global_population)
    # print('top', top)