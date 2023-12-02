from dotenv import load_dotenv

# Griptape 
from griptape.structures import Workflow
from griptape.tasks import PromptTask, ToolkitTask
from griptape.tools import WebScraper
from griptape.drivers import OpenAiChatPromptDriver


# Load environment variables
load_dotenv()

# Define the OpenAiChatPromptDriver with Max Tokens
driver = OpenAiChatPromptDriver(
    model="gpt-4",
    max_tokens=500
)

# Create a Workflow
workflow = Workflow()

# Create a list of movie descriptions
movie_descriptions = [
    "A boy discovers an alien in his back yard",
    "a shark attacks a beach.",
    "A princess and a man named Wesley"
]

compare_task = PromptTask("""
    How are these movies the same: 
    {% for key, value in parent_outputs.items() %}
    {{ value }}
    {% endfor %}
    """,
    prompt_driver=driver,
    id="compare")

# Iterate through the movie descriptions
for description in movie_descriptions:
    movie_task = PromptTask(
        "What movie title is this? Return only the movie name: {{ description }} ",
        context={"description": description},
        prompt_driver=driver
        )
    
    summary_task = ToolkitTask(
        "Give me a summary of the movie: {{ (parent_outputs.items()|list|last)[1] }}",
        tools=[WebScraper()],
        prompt_driver=driver
        )
    
    workflow.add_task(movie_task)
    movie_task.add_child(summary_task)
    summary_task.add_child(compare_task)

# Run the workflow
workflow.run()

# View the output
for task in workflow.output_tasks():
    print(task.output.value)    
