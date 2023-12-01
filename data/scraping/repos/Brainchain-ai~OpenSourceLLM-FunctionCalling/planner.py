import aiohttp
import asyncio
import uuid
import os
from dependency_graph import DependencyGraph
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
import json
from collections import defaultdict
import tiktoken as tt
from pydantic import BaseModel

import promptlayer
from promptlayer import openai

from executor import Executor
from schema import Plan, Step
# Patching instructor as previously done
import instructor
instructor.patch()
from tools import web_search, web_content, web_cache, functions
FUNCTIONS = functions()

def oai_conversation(input_list):
    output_list = []
    i = 0
    while i < len(input_list):
        role = input_list[i]
        if role not in ["system", "user", "assistant", "function"]:
            raise ValueError("Invalid role specified: {}".format(role))
        if role == "function":
            i += 1
            content = input_list[i]
            i += 1
            name = input_list[i]
            output_list.append({"role": role, "content": content, "name": name})
        else:
            i += 1
            content = input_list[i]
            output_list.append({"role": role, "content": content})
        i += 1
    return output_list

# Define your Planner class
class Planner:
    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature
        self.messages = []
        self.setup_api()
        self.prompts = []
        self.plans = []

    def setup_api(self):
        promptlayer.api_key = os.environ["PROMPTLAYER_API_KEY"]
        openai.api_type = "azure"
        openai.api_key = os.environ["AZURE_OPENAI_API_KEY"]
        openai.api_base = "https://brainchainuseast2.openai.azure.com"
        openai.api_version = "2023-08-01-preview"

    async def generate_plan(self, prompt: str):
        self.prompts.append(prompt)
        enc = tt.encoding_for_model("gpt-4")
        user_content = f"Prompt: {prompt}"
        messages = self.construct_messages(user_content, functions=FUNCTIONS)
        self.messages = messages
        plan_response = self.get_plan(messages)
        print(plan_response)
        self.plans.append(plan_response)
        return self.plans[-1]

    def improve_plan(self, n_iters: int = 2, new_plans: list = []):
        logging.debug(f'Messages at start of improve_plan: {self.messages}')
        logging.debug(f'Plans at start of improve_plan: {self.plans}')

        if n_iters == 0:
            return new_plans
        else:
            if self.plans and len(self.plans) > 0:
                latest_plan = self.plans[-1].dict()
                messages = oai_conversation(
                    [
                        "system", 
                        f"""
                        You are a component of a Plan and Execute machine learning system. Your task is to evaluate and improve the current plan, which is in the form of a Dependency Directed Acyclic Graph (DAG). Each node in this DAG represents a plan step, and edges represent dependencies between steps.

                            1. Ensure the DAG has a unidirectional flow. All dependencies should propagate in a single direction, ensuring a coherent sequence of execution.
                            
                            2. Identify and eliminate or connect independent subgraphs. These are sets of nodes that are connected amongst themselves but are not linked to other nodes in the main graph.
                            
                            3. Check for nodes with identical sets of dependencies. Such nodes should either be merged or should share a subsequent child dependency to justify their separate existence.
                            
                            4. If you can replace a node with a more detailed node, feel free to modify the language to make it more descriptive.
                            
                            5. If a node can be decomposed into more specific nodes that are independent and can be potentially parallelized, feel free to create new nodes that are dependencies of the target node and make sure the graph ordering and logic is preserved or improved.
                            
                            6. Examine the graph for pairs or sets of nodes that have overlapping dependencies but do not contribute to the same subsequent node(s). Either merge these nodes, reroute their dependencies to a common subsequent node, or justify their independent existence.
                            
                            7. Not all nodes have dependencies but if a node has no dependencies, it must be a dependency for another node. Ensure that this is the case. If not, adjust the node order or node dependencies, or merge or explode nodes as needed.
                                                            
                        Taking a deep breath and thinking step by step, please execute the following instructions based on the rules outlined above:
                        Given this plan: {latest_plan} you must validate, improve and provide an optimized, improved plan adhering to the DAG state guidelines.
                        """.replace("\n","").strip()
                    ])
                # print("Messages after latest plan: ", json.dumps(messages, indent=2))
                
                self.temperature = 0.9*self.temperature

                for message in messages:
                    logging.debug(message)
                
                plan_response = openai.ChatCompletion.create(
                    engine=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    response_model=Plan,
                    n=1
                )

                new_plans.append(plan_response)
                return self.improve_plan(n_iters=n_iters-1, new_plans=new_plans)
        
    def construct_messages(self, user_content, functions=FUNCTIONS):
        logging.debug(f'Messages in construct_messages: {self.messages}')
        tools_available = [f"{function['name']}: {function['description']}" for function in functions]
        print(tools_available)
        planner_prompt = """
            Instruction: 

            Firstly, understand the problem thoroughly and devise a plan to solve it. The plan should be outputted in the following JSON format: two keys 1) 'prompt' with the original user prompt and 1) 'plan' with a list of JSON objects. Each JSON object should have three keys: 'plan_step_index', which is the index of the current plan step (starts at 0); 'step', which is a string description of this step of the plan; and 'dependencies', which is a list of integer indices corresponding to any other plan steps that this plan step is dependent on. 

            A dependency is created when a step requires data or outcomes from a previous step to execute. If a step can utilize the results of another step, it should be listed as a dependency, ensuring a logical flow of data and processing. 

            The plan should be a directed acyclic graph, i.e. If a node B exists that depends on node A, then node A cannot be dependent on node B. if B-->A, then A-/->B. If B-->A and C-->B, then A-/->B and B-/->C.

            Whenever possible, design the plan such that steps that do not have dependencies between them can be executed in parallel. This will enable multiple LLM executors to operate concurrently, working through the plan steps in parallel and then amalgamating the results as needed. If parallel execution isn't feasible due to the nature of the task, proceed with a sequential plan but always aim to minimize the number of steps and dependencies to promote efficiency.

            The final step in the plan, especially if the task is a question, should be 'Given the above steps taken, please respond to the user's original question'. This step should be dependent on every other step of the plan that contributes to forming the answer. Make sure that the plan has the minimum number of steps required to accurately complete the task, and that each step clearly contributes to the task’s resolution.

            If any URLs or any other specific identifiers are used, please make sure that the plan steps includes all relevant info.

            At the end of your plan, denote '<END_OF_PLAN>'.
        """.strip().replace("\n"," ")

        for pair in (oai_conversation([
            "system",
            f"Take a deep breath and begin thinking in a methodical, calculated manner. {planner_prompt}. After you have understood and contemplated the problem, try to incorporate the tools from the assistant in the plan steps where appropriate.",
            "assistant",
            f"Here are the tools available: {tools_available}",
            "user",
            user_content,
        ])): self.messages.append(pair)

        # print(json.dumps(self.messages, indent=2))
        return self.messages

    def get_messages(self):
        return self.messages

    def get_plan(self, messages):
       return openai.ChatCompletion.create(
            engine=self.model,
            messages=messages,
            temperature=self.temperature,
            response_model=Plan,
            n=1
        )

    async def execute_plan(self, plan):
        # Step results will be stored here
        step_results = {}

        # Create an empty list for each step to hold its dependencies
        dependencies = defaultdict(list)
        print(plan)

        # Fill in the dependencies list based on the plan
        if type(plan) == list:
            plan = plan[0]
        if type(plan) == dict:
            plan = Plan(**plan)
        
        for step in plan.steps:
            for dep in step.dependencies:
                dependencies[step.plan_step_index].append(dep)

        # Asynchronous function to execute a single step
        async def execute_step(step):
            # Wait for dependencies to be resolved
            for dep in dependencies[step.plan_step_index]:
                await futures[dep]  # Wait on the Future object, not the index
            # Execute the step and store the result
            executor = Executor(step=step, dependency_results={k: step_results[k] for k in step.dependencies})
            result = await executor.begin(prev_steps=[step_results[d] for d in step.dependencies])
            step_results[step.plan_step_index] = result
            return result

        # Create a future for each step
        futures = {}
        for step in plan.steps:
            futures[step.plan_step_index] = asyncio.ensure_future(execute_step(step))

        # Wait for all steps to complete
        await asyncio.gather(*futures.values())

        # Aggregate results and answer the original query
        aggregated_results = await self.aggregate_results(step_results)
        return aggregated_results

    async def aggregate_results(self, step_results):
        # First, aggregate the step results into a format that can be easily understood.
        aggregated_info = "\n".join([f"Step {step}: {result}" for step, result in step_results.items()])
        print(aggregated_info)
        messages = oai_conversation([
            "system"
            ,"You are tasked with giving a final response given all the output responses of the previous steps executed in response to a user's query. Please include footnotes, hyperlinks and references inline where appropriate in Markdown format, and ensure that the final response is grammatically correct and coherent. Given that the steps were executed in parallel, you may need to rephrase the responses to ensure that they are coherent with each other."
            ,"user"
            ,f"Original Query: {self.prompts[-1]}"  # Assuming the last prompt is the user's original query
            ,"assistant"
            ,f"Aggregated Step Results: {aggregated_info}"
        ])

        # Finally, make a Chat API call to generate the summary.
        summary_response = await openai.ChatCompletion.acreate(
            engine=self.model,
            messages=messages,  
            temperature=0.5,
        )

        return summary_response['choices'][0]['message']['content']

# Usage
api_keys = {
    "PROMPTLAYER_API_KEY": os.environ["PROMPTLAYER_API_KEY"],
    "AZURE_OPENAI_API_KEY": os.environ["AZURE_OPENAI_API_KEY"]
}

async def main():
    planner = Planner(model="gpt-4-32k", temperature=0.5, api_keys=api_keys)
    query = input("Enter your query: ")
    plan = await planner.generate_plan(query)
    dependency_graph = DependencyGraph(plan)

    uuid_obj = str(uuid.uuid4())
    directory = str(os.getcwd())
    filepath = os.path.join(directory, uuid_obj)
    print(filepath)

    dependency_graph.visualize(filepath+"_0")  # This will create a PNG image and open it
    improved_plan = planner.improve_plan(n_iters=1)

    dependency_graph = DependencyGraph(improved_plan)
    dependency_graph.visualize(filepath+"_1")  # This will create a PNG image and open it
    run_info = await planner.execute_plan(improved_plan)
    print(run_info)

    
    # dependency_graph = DependencyGraph(improved_plan)
    # dependency_graph.visualize(filepath+"_1")  # This will create a PNG image and open it    
    print(run_info)
    return run_info
    
if __name__ == "__main__":
    asyncio.run(main())