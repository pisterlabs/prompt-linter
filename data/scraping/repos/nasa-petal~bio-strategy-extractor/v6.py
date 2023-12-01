# ! pip install google-api-python-client --quiet
# ! pip install openai --quiet
# ! pip install langchain --quiet

import openai
import langchain
import os

from langchain.utilities import GoogleSearchAPIWrapper
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool

os.environ["GOOGLE_CSE_ID"] = "GOOGLE CSE ID"
os.environ["GOOGLE_API_KEY"] = "GOOGLE API KEY"

##################################################################################################
# Path 1: the engineer has a problem with a previously implemented solution (ex. electric plane) #
##################################################################################################
# Path 2: the engineer has a novel product to implement (ex. flying car)                         #
##################################################################################################
# Path 3: the engineer has a novel problem to solve (ex. time travel)                            #
##################################################################################################

# after the path is chosen and product/problem is entered, the engineer will be returned 10 reframed 
# versions of the input, which will either be labelled as made by looking at the big picture or by
# looking at the smaller details

TREE_DIRECTORY = {}

def main():
  path = 2
  problem_statement = "flying car"
  current_index = 0
  if path == 2:
    vars = vars_path_2()
    vars["product"] = problem_statement
    questions = questions_path_2(vars)
    question_flow_path_2(questions, vars, 0, 0, 0)
  if path == 3:
    vars = vars_path_3()
    vars["problem"] = problem_statement
    questions = questions_path_3(vars)
    question_flow_path_3(questions, vars, 0, 0, 0)

def question_flow_path_2(questions, vars, q_index, a_index, curr_index):
  global current_index
  print(vars)
  questions = questions_path_2(vars)
  question = questions[q_index]
  answers_unparsed = generate_response(question)
  answers = parsed(answers_unparsed)
  reframed.append(answers)
  start = a_index + 1
  end = a_index + len(answers)
  list_of_ans = []
  for answer in answers:
    if "" in vars[question[1]]:
      vars[question[1]] = []
    vars[question[1]].append(answer)
    current_index += 1
    list_of_ans.append(current_index)
    if q_index < len(questions) - 1:
      question_flow_path_2(questions, vars, q_index + 1, end, current_index)
  TREE_DIRECTORY[curr_index] = list_of_ans

def questions_path_2(vars):
  return [["Describe the potential users of the " + vars["product"] + ".", "users"],
          ["What are the intended outcomes of the " + vars["users"][-1] + " when using the " + vars["product"] + "?", "outcomes"],
          ["List " + vars["users"][-1] + " scenarios when using a " + vars["product"] + " for " + vars["outcomes"][-1] + ".", "scenarios"],
          [vars["scenarios"][-1] + "\n\nName the actions that " + vars["users"][-1] + " take that lead up to using the " + vars["product"] + ".", "lead_up"],
          [vars["scenarios"][-1] + "\n\nName the actions that " + vars["users"][-1] + " take that directly follow after using the " + vars["product"] + ".", "after"]]

def vars_path_2():
  return {"product" : "",
          "users" : [""],
          "outcomes" : [""],
          "scenarios" : [""],
          "lead_up" : [""],
          "after" : [""],
          "all_records" : ""}

def question_flow_path_3(questions, vars, q_index, a_index, curr_index): # q_index initially set to 0 by default
  global current_index, reframed
  questions = questions_path_3(vars)
  question = questions[q_index]
  answers_unparsed = generate_response(question)
  if "What are the expected outcomes" in question[0]:
    answers_unparsed = openai.Completion.create(model = 'text-davinci-003',
                                                prompt = question[0],
                                                temperature = 0.7,
                                                max_tokens = 256,
                                                top_p = 1,
                                                frequency_penalty = 0,
                                                presence_penalty = 0)['choices'][0]['text']
  answers = parsed(answers_unparsed)
  for answer in answers:
    reframed.append(answer)
  start = a_index + 1
  end = a_index + len(answers)
  list_of_ans = []
  for answer in answers:
    vars[question[1]].append(answer)
    current_index += 1
    list_of_ans.append(current_index)
    if q_index < len(questions) - 1:
      question_flow_path_2(questions, vars, q_index + 1, end, current_index)
  TREE_DIRECTORY[curr_index] = list_of_ans

def questions_path_3(vars):
  return [["Describe the physical space in which " + vars["problem"] + " exists.", "envir"],
          ["What are the expected outcomes in " + vars["envir"] + " when you " + vars["problem"], "outcomes"],
          ["List obstacles associated with " + vars["outcomes"] + ", as well as obstacles that are similar to " + vars["problem"] + ".", "obstacles"],
          ["Name what can trigger " + vars["problem"] + " to occur.", "causes"],
          ["Name what can result from " + vars["problem"] + ".", "effects"]]

def vars_path_3():
  return {"problem" : "",
          "envir" : [""],
          "outcomes" : [""],
          "obstacles" : [""],
          "causes" : [""],
          "effects" : [""],
          "all_records" : ""}

def generate_response(query):
  llm = OpenAI(temperature = 0.1, openai_api_key = 'OPENAI API KEY')
  search = GoogleSearchAPIWrapper()
  tools = [Tool(name = "Intermediate Answer",
                description = "search", # sometimes you need to rotate through alternative descriptions, like "use this search tool" to overcome a langchain error
                func = search.run)]
  agent = initialize_agent(tools, llm, agent = "chat-zero-shot-react-description", verbose = True, return_intermediate_steps = True)
  response = agent({"Question: " + query[0] + "\n\nAnswer: "})
  vars["all_records"] += str(response['intermediate_steps'])
  return response['output']

def parsed(answer):
  response = openai.Completion.create(model = 'text-davinci-003',
                                      prompt = "Question: The users of a flying car would be consumers, military users, and business users.\nAnswer: consumers, military users, business users\n\nQuestion: The potential benefits of using a flying car include minimizing traffic pollution, lower emissions, shorter travel distances, freeing up roads, reduced roadway congestion, the option value of having a flying car, maneuverability, lower and more reliable travel time, likelihood of fuel, maintenance and operational cost savings, and greater flexibility to leverage the benefits of flying cars.\nAnswer: minimizing traffic pollution, lower emissions, shorter travel distances, freeing up roads, reduced roadway congestion, the option value of having a flying car, maneuverability, lower and more reliable travel time, likelihood of fuel, maintenance and operational cost savings, greater flexibility to leverage the benefits of flying cars\n\nQuestion: Consumer scenarios when using a flying car for shorter travel distances include commuting to and from work, errands, business travel, short-distance leisure travel, such as trips to the beach or mountains, and taking fewer long round-trip flights to reduce personal carbon footprints.\nAnswer: commuting to and from work, errands, business travel, short-distance leisure travel to the beach or mountains, taking fewer long round-trip flights to reduce personal carbon footprints\n\nQuestion: " + query + "\nAnswer: ",
                                      temperature = 0.7,
                                      max_tokens = 256,
                                      top_p = 1,
                                      frequency_penalty = 0,
                                      presence_penalty = 0)['choices'][0]['text']
  return response.split(", ")

if __name__ == "__main__":
    main()
