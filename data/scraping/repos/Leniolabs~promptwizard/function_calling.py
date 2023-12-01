from ..openai_calls import openai_call
import json
from ..cost import input, output
import concurrent.futures
from typing import List, Dict

class functionCalling:
    def __init__(self, test_cases: List[Dict], prompts: List[str], functions: List[Dict], model_test: str="gpt-3.5-turbo", model_test_temperature: float=0.6, model_test_max_tokens: int=1000, best_prompts: int=2, function_call: str="auto", timeout: int=10, n_retries: int=5):

        """
        Initializes an instance of the functionCalling class.

        Args:
            test_cases (list): A list of test cases, each containing 'input', 'output1', and 'output2' fields.
            number_of_prompts (int): The number of prompts to evaluate.
            model_test (str): The name of the GPT model used for testing.
            model_test_temperature (float): The temperature setting for testing.
            model_test_max_tokens (int): The maximum number of tokens for testing.
            model_generation (str): The name of the GPT model used for prompt generation.
            model_generation_temperature (float): The temperature setting for prompt generation.
            prompts (list): A list of prompts to evaluate.
            functions (list): A list of functions used in function calls.
            function_call (str): The function call format.
            best_prompts (int): The number of best prompts to select.

        Note:
            The 'system_gen_system_prompt' attribute is predefined within the class constructor.
        Initializes the functionCalling class with the provided parameters.
        """

        self.test_cases = test_cases
        self.model_test = model_test
        self.model_test_temperature = model_test_temperature
        self.model_test_max_tokens = model_test_max_tokens
        self.prompts = prompts
        self.best_prompts = best_prompts
        self.functions = functions
        self.function_call = function_call
        self.system_gen_system_prompt = """Your job is to generate system prompts for GPT, given a description of the use-case and some test cases.

In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output. Be creative with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.

You should respond a function_call and nothing else. Don't text us back.

You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified.

Most importantly, output NOTHING but the prompt. Do not include anything else in your message."""
        self.timeout = timeout
        self.n_retries = n_retries

    def process_prompt(self, prompt, test_case, model, model_max_tokens, model_temperature, functions, function_call):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"{test_case['input']}"}
        ]
        response = openai_call.create_chat_completion(model, messages, model_max_tokens, model_temperature, 1, functions=functions, function_call=function_call, timeout=self.timeout, n_retries=self.n_retries)
        partial_tokens_input = response.usage.prompt_tokens
        partial_tokens_output = response.usage.completion_tokens
        print(response)
        response_content = response
        
        return partial_tokens_input, partial_tokens_output, response_content, test_case['output1'], test_case['output2']

    def test_candidate_prompts(self):

        """
        Test candidate prompts against provided test cases.

        Returns:
            tuple: A tuple containing data list, best prompts, cost, input tokens used, and output tokens used.

        This method evaluates a set of prompts by generating responses using a GPT model and comparing the generated
        responses to expected outputs from test cases. It returns information about the performance of each prompt.
        """

        cost = 0
        tokens_input = 0
        tokens_output = 0
        prompt_results = {prompt: {'correct': 0, 'total': 0} for prompt in self.prompts}
        results = [{"method": "Function Calling"}]

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []

            for prompt in self.prompts:
                prompt_and_results = [{"prompt": prompt}]
                for test_case in self.test_cases:
                    future = executor.submit(
                        self.process_prompt,
                        prompt,
                        test_case,
                        self.model_test,
                        self.model_test_max_tokens,
                        self.model_test_temperature,
                        self.functions,
                        self.function_call
                    )
                    futures.append(future)
                    partial_tokens_input, partial_tokens_output, response, ideal_output1, ideal_output2 = future.result()
                    tokens_input += partial_tokens_input
                    tokens_output += partial_tokens_output

                    if "function_call" in response.choices[0].message:
                    # Update model results
                        json_object = json.loads(str(response.choices[0].message.function_call.arguments))

                        def extract_keys_and_values(json_list):
                            all_keys = list(json_object.keys())
                            key_value_pairs = [json_object[key]for key in all_keys]
                            return all_keys, key_value_pairs[0]

                        if ideal_output1 == response.choices[0].message.function_call.name and extract_keys_and_values(json_object)[1] == ideal_output2:
                            prompt_results[prompt]['correct'] += 1
                        prompt_results[prompt]['total'] += 1
                        if ideal_output1 == response.choices[0].message.function_call.name and ideal_output2 == extract_keys_and_values(json_object)[1]:
                            result = True
                        if ideal_output1 == response.choices[0].message.function_call.name and not (ideal_output2 == extract_keys_and_values(json_object)[1]):
                            result = 'variable error'
                        if not (ideal_output1 == response.choices[0].message.function_call.name) and ideal_output2 == extract_keys_and_values(json_object)[1]:
                            result = 'function error'
                        if not (ideal_output1 == response.choices[0].message.function_call.name) and (not ideal_output2 == extract_keys_and_values(json_object)[1]):
                            result = 'function and variable error'
                        prompt_and_results.append({"test": test_case['input'], "answer": {"function": f"{response.choices[0].message.function_call.name}", "variable": f"{extract_keys_and_values(json_object)[1]}"}, "ideal": {"function": f"{ideal_output1}", "variable": f"{ideal_output2}"}, "result": result})
                    if "function_call" not in response.choices[0].message:
                        prompt_and_results.append({"test": test_case['input'], "answer": 'not a function call', "ideal": {"function": f"{ideal_output1}", "variable": f"{ideal_output2}"}, "result": 'Received text data instead of JSON.'})
                        prompt_results[prompt]['total'] += 1

                results.append(prompt_and_results)
                prompt_and_results = []

        cost_input = input.cost(tokens_input, self.model_test)
        cost_output = output.cost(tokens_output, self.model_test)
        cost = cost + cost_input + cost_output

        # Calculate and print the percentage of correct answers and average time for each model
        best_prompt = self.prompts[0]
        best_percentage = 0
        data_list = []
        for i, prompt in enumerate(self.prompts):
            correct = prompt_results[prompt]['correct']
            total = prompt_results[prompt]['total']
            percentage = (correct / total) * 100
            data_list.append({"prompt": prompt, "rating": percentage})
            print(f"Prompt {i+1} got {percentage:.2f}% correct.")
            if percentage >= best_percentage:
                best_percentage = percentage
                best_prompt = prompt
        sorted_data = sorted(data_list, key=lambda x: x['rating'], reverse=True)
        best_prompts = sorted_data[:self.best_prompts]
        print(f"The best prompt was '{best_prompt}' with a correctness of {best_percentage:.2f}%.")
        sorted_data.append(results)
        return sorted_data, best_prompts, cost, tokens_input, tokens_output
    
    def evaluate_optimal_prompt(self):

        """
        Evaluate the optimal prompt for function calling.

        Returns:
            tuple: A tuple containing data list, best prompts, cost, input tokens used, and output tokens used.

        This method evaluates the optimal prompt for function calling by calling the `test_candidate_prompts` method.
        """
        
        return self.test_candidate_prompts()