import chromadb
from langchain import OpenAI
from agentmemory import (
    create_memory,
    get_memories,
    update_memory
)


class SurveyAnalysisAgent:
    def __init__(self):
        self.langchain = OpenAI()
        self.db = chromadb.HttpClient(
            host="localhost", port=8000)

        # TODO Figure out how to create system prompt
        # Read the Sensei prompt from the file
        # with open('templates/sensei_prompt.txt', 'r') as file:
        # sensei_prompt = file.read()

        with open('templates/survey_analysis.txt', 'r') as file:
            self.prompt_template = file.read()

        # Set the system prompt in LangChain (refer to LangChain docs for the exact method)
        # self.langchain.set_system_prompt(sensei_prompt)

    def analyze_and_store_goals(self, survey_data):
        # Analyze survey data using LangChain
        prompt = self.prompt_template.format(survey_data=survey_data)

        goals = self.langchain.analyze_survey_data(prompt)

        # Store goals using AgentMemory
        for goal in goals:
            create_memory(category="goal", content=goal)

    # Rest of the class implementation

    def analyze_survey_data(self, survey_data):
        # Analyze survey data using LangChain
        goals = self.langchain.analyze_survey_data(survey_data)
        return goals

    def generate_daily_tasks(self, goals):
        # Generate daily tasks using LangChain
        tasks = self.langchain.generate_daily_tasks(goals)
        return tasks

    def calculate_task_completion_percentage(self, user_id):
        # Retrieve user's tasks from AgentMemory
        tasks = get_memories(category="task", user_id=user_id)

        # Calculate task completion percentage
        completed_tasks = [
            task for task in tasks if task['content']['status'] == 'completed']
        completion_percentage = len(completed_tasks) / len(tasks) * 100

        return completion_percentage
