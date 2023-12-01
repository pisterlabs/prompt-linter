import time
import gradio as  gr
import openai
from chainofaction.agents.inferenceAgent import InferenceAgent
import vector_database.vector_database as skills
import threading
import queue

class InteractiveEnv:
    def __init__(self):
        dataset = "leetcode"#?
        emb_func = skills.MiniLML6V2EmbeddingFunction()
        data_dir = "chainofaction/data"
        self.api_key_1 = "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"
        self.api_key_2 = "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"
        self.db= skills.ChromaWithUpsert(
        name=f"{dataset}_minilm6v2",
        embedding_function=emb_func,  # you can have something here using /embed endpoint
        persist_directory= data_dir
        )
        self.skillagent = InferenceAgent(self.db,self, self.api_key_1)
        self.zeroagent = InferenceAgent(self.db, self, self.api_key_2)



    def reset(self):
        self.agent.reset()
        self.init_db()


    def predict_concurrently_stream(self, problem_description):
        zero_shot_queue = queue.Queue()
        skill_db_queue = queue.Queue()

        def call_zero_shot():
            response = self.zeroagent.get_response_zeroshot(problem_description,self.api_key_1)
            for res in response:
                zero_shot_queue.put(res)
            zero_shot_queue.put(None)

        def call_skill_db():
            response = self.skillagent.get_response(problem_description,self.api_key_2)
            for res in response:
               skill_db_queue.put(res)
            skill_db_queue.put(None)

        # Start threads for concurrent processing
        threading.Thread(target=call_zero_shot).start()
        threading.Thread(target=call_skill_db).start()
        zero_shot_done, skill_db_done, skills_done = False, False,False
        skills_response = ""
        # Yield responses as they become available
        while True:
            try:
                if not zero_shot_done:
                    response = zero_shot_queue.get(timeout=30)
                    if response is None:
                        zero_shot_done = True
                    else:
                        zero_shot_response = response

                if not skill_db_done:
                    response = skill_db_queue.get(timeout=30)
                    if response is None and not skills_done:
                        skills_done = True
                    elif response is None and skills_done:
                        skill_db_done = True
                    elif response is not None and skills_done:
                        skill_db_response = response
                    elif response is not None and not skills_done:
                        print(response)
                        code_fetched, skill, distance = response
                        skills_response += f"Step: {skill}\nCandidate Skill: {code_fetched}\nDistance: {distance}\n\n"
                        skill_db_response = ""
                yield zero_shot_response, skill_db_response, skills_response

                # Break if both threads are done
                if zero_shot_done and skill_db_done:
                    break

            except queue.Empty:
                print("QUEUE EMPTY")
                break  # Break if timeout occurs

env = InteractiveEnv()


# Define the interface components
problem_description = gr.Textbox(label="💻 Problem Description", placeholder="Enter the LeetCode problem description here...", lines=5)
zero_shot_solution = gr.Textbox(label="🚀 ZeroShot Code Solution", placeholder="ZeroShot solution will be displayed here...", lines=10, interactive=True)
skilldb_solution = gr.Textbox(label="🛠️ SkillDB Code Solution", placeholder="SkillDB solution will be displayed here...", lines=10, interactive=True)
skills_found = gr.Textbox(label="🔎 Skills Found", placeholder="Skills found will be displayed here...", lines=10, interactive=True)
# Define the inputs and outputs for the Gradio Interface
inputs = [problem_description]
outputs = [zero_shot_solution, skilldb_solution, skills_found]


# Custom CSS to improve mobile responsiveness
custom_css = """
@media (max-width: 700px) {
    .gradio-container {
        width: 95% !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }
    .gradio-input, .gradio-output {
        width: 100% !important;
    }
}
"""


# Create the Gradio Interface
iface = gr.Interface(
    fn= env.predict_concurrently_stream, 
    inputs=inputs, 
    outputs=outputs, 
    title="LeetCode Problem Solver 🎉",
    description="Enter the LeetCode problem, and get solutions from both ZeroShot and SkillDB agents streamed in real-time!",
    examples=[["Example problem description"]],
    css= custom_css
)

# Launch the interface with queueing to manage load
iface.queue().launch(share = True)
