import openai
from jetbot import Robot
import time

# Initialize the robot
robot = Robot()

# Define basic motion functions
def stop():
    robot.stop()

def step_forward():
    robot.forward(0.4)
    time.sleep(0.5)
    robot.stop()

def step_backward():
    robot.backward(0.4)
    time.sleep(0.5)
    robot.stop()

def step_left():
    robot.left(0.3)
    time.sleep(0.5)
    robot.stop()

def step_right():
    robot.right(0.3)
    time.sleep(0.5)
    robot.stop()

# Set up OpenAI API
openai.api_key = 'YOUR_OPENAI_API_KEY'  # Replace with your API key

def get_command_from_gpt(prompt_text):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=prompt_text,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def execute_complex_command(user_wish):
    # Provide GPT with the user's wish and the basic commands
    prompt_text = f"The user wants the robot to '{user_wish}'. Translate this wish into one of the following basic actions: forward, backward, left, right, stop."
    command_sequence = get_command_from_gpt(prompt_text).split(', ')  # Assuming GPT returns a comma-separated list of commands
    
    for command in command_sequence:
        if command == "forward":
            step_forward()
        elif command == "backward":
            step_backward()
        elif command == "left":
            step_left()
        elif command == "right":
            step_right()
        elif command == "stop":
            stop()
        else:
            print(f"Invalid command: {command}")

# Example usage
user_wish = "go in circles"
execute_complex_command(user_wish)
