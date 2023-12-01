import os
import openai
from dotenv import load_dotenv
import re
from functions import detect_wall, turn_left, go_forward, turn_right, shortest_path, judge_path

def evaluate(code, maze):
    # Here is where you will evaluate the code and return the results
    # start bottom right, end top left
    end =  maze[0]['end']
    start =  maze[0]['start']
    map_part = maze[0]['map']
    path_taken =  []
    namespace = {}

    # Check input size
    print("Code length in evaluate: ", len(str(code)))
    if len(str(code)) < 3:
        print("Code is too short")
        return False, 0, "The given code is either too long or too short. Please try again!", path_taken
    elif len(str(code)) > 2048:
        print("Code is too long")
        return False, 0, "The given code is either too long or too short. Please try again!", path_taken

    # Predefined functions
    with open("functions.txt", "r", encoding="utf-8") as file:
        functions_predefined = file.read()

    parameters = f"""start_point = {start}
end_point = {end}
maze = {map_part}
current_position = start_point
direction = "down"
result = []
result.append(current_position)
"""

    full_code = parameters + "\n" + functions_predefined + "\n" + code #  + "\nresult = execute_labyrinth()"
    print("Full code:")
    print(full_code)

    result_path = path_taken  #TODO change this later
    result = False
    score = 0
    try:
        exec(full_code, {'detect_wall': detect_wall, 'turn_left': turn_left, 'turn_right': turn_right, 'go_forward': go_forward}, namespace)
        result_path = namespace["result"] #TODO inverse the path?
        print("Result Path:")
        print(result_path)
            # Check if the result is correct
        if (result_path[-1] == end):
            result = True
        else:
            result = False
    except Exception as e:
        error_message = str(e)
        print("Error occurred during execution: ", error_message)
        # result_path = path_taken
        # print("Result path replaced with the correct path.")

    # print("Result:")
    # print(result)

    # Call to OpenAI to get the feedback and score
    load_dotenv()
    openai.api_key = os.environ["OPENAI_API_KEY"]

    if result:
        answer = "Feedback: The code is correct. Super! The player can get from start point to end point using that python code. Score: 100. The percentage of success is 100."
    else:
        try:
            # OpenAI API call
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant providing feedback and a score for the given python code."},
                    {"role": "user", "content": f"""I am going to give you a python code which was previously translated from the pseudo code, so it is also normal that there is no comment in the code. 
                    Assume that the defined functions without the code inside are defined correctly and working correctly. Focus on the usage & results of the functions. 
                    Main goal of this code is the player tries to travel from start point {start} to end point {end} in this specific maze: {map_part}. This code is not a general algorithm, but a specific one to solve this specific maze. When direction is 'down', one direction forward means going from (5, 5) to (4, 5).
                    Please provide a short feedback and don't forget that the player could'nt from start point to end point using this python code.
                    The soultion is this array: {shortest_path(maze, start, end)}. The path that user has taken is this array: {result_path}.
                    The functions that the player can use to reach {end} are the following: {functions_predefined}.
                    Most importantly please give the feedback, so that I will extract it as 'Feedback:\s*(.*)' using re.compile().'.
                    You should provide feedback to the following lines of code: {code}"""},
                ]
            )
            # Extract the answer from the API response
            answer = response['choices'][0]['message']['content']
        except Exception as e:
            print(f"Error while making API call: {e}")
            answer = "Feedback: Unfortunataly can't give feedback right now due to problem with the API call. Please contact the developers."
            return False, 0, answer, result_path
    
    print("Answer from OpenAI - Feedback:")
    print(answer)

    feedback_pattern = re.compile(r'Feedback:\s*(.*)', re.IGNORECASE)
    feedback_match = feedback_pattern.search(answer)

    if feedback_match:
        feedback = feedback_match.group(1).strip()
    else:
        feedback = ""  # TODO: Handle this error
  

    try:
        feedback += f" {judge_path(maze, start, end, result_path)}"
    except Exception as e:
        print(f"Error while judging the path: {e}")

    print("path taken:", len(result_path))
    print("shortest path:", len(shortest_path(maze, start, end)))

    if len(result_path) == len(shortest_path(maze, start, end)):
        score = 100
    elif len(result_path) > len(shortest_path(maze, start, end)): # TODO feedback
        score = round((len(shortest_path(maze, start, end)) / len(result_path) * 100), 2)
    else:
        score = 0

    return result, score, feedback, result_path