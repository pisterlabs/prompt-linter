import openai
from termcolor import colored

# Initialize the OpenAI API
openai.api_key = 'KEY'  # Please replace with your actual key

def get_next_story_segment(prompt):
    """Get the next segment of the story based on the prompt."""
    print(colored("Prompt:", "magenta", attrs=["bold"]))
    print(colored(prompt, "magenta"))
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=500
    )
    output = response.choices[0].message['content']
    print(colored("Output:", "cyan", attrs=["bold"]))
    print(colored(output, "cyan"))
    print("\n" + "-"*50 + "\n")  # Print a separator
    return output

def refine_story(existing_story, goal):
    """Refine the existing story towards a goal."""
    prompt = f"Given the story: '{existing_story}', how can it be refined and improved to align more with the goal where the {goal}?"
    return get_next_story_segment(prompt)

def compare_stories(initial_story, final_story):
    """Ask the model to compare the two versions and highlight the improvements."""
    prompt = f"Compare the initial draft: '{initial_story}' with the final story: '{final_story}'. Why is the final version considered better?"
    return get_next_story_segment(prompt)

def generate_story(seed, max_iterations, goal):
    """Iteratively generate and refine a story towards a certain goal."""
    
    # Generate the initial story draft
    initial_prompt = f"{seed} The ultimate end goal is where the {goal}. What happens next?"
    initial_draft = get_next_story_segment(initial_prompt)
    story = initial_draft

    # Refinement iterations
    for i in range(max_iterations):
        story = refine_story(story, goal)
        
        # Check for the goal in the story
        if goal.lower() in story.lower():
            print(
                colored(f"Goal reached at iteration {i+1}!", "green", attrs=["bold"]))
            break

    return initial_draft, story

# Example usage
seed = "In a galaxy far, far away, after the fall of the Empire, a new protagonist emerges."
max_iterations = 3
goal = "main character becomes ruler of the universe"

initial_draft, final_story = generate_story(seed, max_iterations, goal)

print(colored("\nInitial Draft:\n", "yellow", attrs=["bold"]))
print(colored(initial_draft, "blue"))

print(colored("\nFinal Story:\n", "yellow", attrs=["bold"]))
print(colored(final_story, "blue"))

comparison = compare_stories(initial_draft, final_story)
print(colored("\nComparison and Improvements:\n", "yellow", attrs=["bold"]))
print(colored(comparison, "green"))
