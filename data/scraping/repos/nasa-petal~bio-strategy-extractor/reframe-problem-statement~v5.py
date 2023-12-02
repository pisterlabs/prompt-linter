# ! pip install langchain --quiet
# ! pip install openai --quiet
# ! pip install google-search-results --quiet

import openai

from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent

answer_format_addons = [
    "Answer using a noun.",
    "Answer using an infinitive verb.", # if contains 'to', drop the 'to'
    "Answer using nouns, separated by rows.",
    "Answer with a detailed description of the conditions.",
    ""
]

def generate_response(query):
  llm = OpenAI(temperature=0.3, openai_api_key = 'INSERT_OPENAI_API_KEY_HERE')
  tools = load_tools(["serpapi"], llm=llm, serpapi_api_key = 'INSERT_SERPAPI_API_KEY_HERE')
  agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
  return agent.run(query)

def next_q_var(q): # [[1], "q", "ans", 0] OR [[], "q", "ans", 0] -> update the background info, go to next
  q_and_a = vars["background_info"].split("\n")
  if len(q_and_a) == 3:
    q_and_a.pop(0)
  vars["background_info"] = "\n".join(q_and_a)
  response = generate_response(vars["background_info"] + "\nQ: " + q[1] + " " + answer_format_addons[q[3]] + " A: ").strip().lower()
  if q[3] == 1 and "to " in response:
    response = response[3:]
  vars["background_info"] += "\nQ: " + q[1] + " A: " + response
  vars[q[2]] = response
  print(vars["background_info"])
  print("---")
  if len(q[0]) == 0: # end of questioning sequence
    return None
  return q[0][0]

def next_q_novar(q): # [[1], "q"] OR [[], "q"] -> update the background info, go to next
  q_and_a = vars["background_info"].split("\n")
  if len(q_and_a) == 3:
    q_and_a.pop(0)
  vars["background_info"] = "\n".join(q_and_a) + "\n" + q[1]
  print(vars["background_info"])
  print("---")
  if len(q[0]) == 0: # end of questioning sequence
    return None
  return q[0][0]

def yn_q_novar(q): # [[1, 2], "q"] -> go to y/n
  q_and_a = vars["background_info"].split("\n")
  if len(q_and_a) == 3:
    q_and_a.pop(0)
  vars["background_info"] = "\n".join(q_and_a)
  q_and_a.append("Q: " + q[1] + " A: ")
  response = generate_response(vars["background_info"] + "\nQ: " + q[1] + " Answer with a yes or no. A: ").strip().lower()
  vars["background_info"] += "\nQ: " + q[1] + " A: " + response
  print(vars["background_info"])
  print("---")
  if "yes" in response:
    return q[0][0]
  elif "no" in response:
    return q[0][1]
  else:
    return None
  
def questions(vars):
  return [[[4, 3], "Does the problem you want to face involve the whole " + vars["sys"] + "?"],
          [[4], "Type the name of the system that undergoes the problem ", "sys", 0],
          [[5], "Now we have to define the overall goal of the " + vars["sys"] + " meaning the motivation of its/their existence. Think about the industry and the market of the " + vars["sys"] + ", what the " + vars["sys"] + " is/are meant to achieve, the intentions of the users that exploit the " + vars["sys"] + " in terms of the desired modifications of a certain state/condition in the world. Could you then briefly describe the purpose of the " + vars["sys"] + "?", "goal", 1],
          [[6], "Which technical function is carried out by the " + vars["sys"] + " in order to " + vars["goal"] + "?", "gpf", 1],
          [[7, 12], "Is it possible to split the function ‘‘to " + vars["gpf"] + "’’ into two subfunctions?"],
          [[8], "Type the first subfunction.", "p1", 1],
          [[9], "Type the second subfunction.", "p2", 1],
          [[10], "Type again the subfunction that is more difficult to be delivered, or that is associated at a greater extent with undesired effects. If both the subfunctions are equally related to the problem, type again the overall function " + vars["gpf"] + ", or reformulate it with more appropriate terms. Choose one between to " + vars["p1"] + " to " + vars["p2"] + " to " + vars["gpf"] + " to ...(new formulation)?", "gpf", 1],
          [[12], "Now you have changed the primary function of the system. Thus, the " + vars["sys"] + " is/are meant to " + vars["gpf"] + " in order to " + vars["goal"] + "."],
          [[4], "Type the name of the system that is meant to " + vars["gpf"] + ".", "sys", 0],
          [[13], "Type who or what perceives the benefits generated by the " + vars["sys"] + ", when it works in order to " + vars["goal"] + ".", "ben", 0],
          [[14], "Describe what or who undergoes the modifications carried out through the action ‘‘to " + vars["gpf"] + "’’.", "obj", 0],
          [[15], "Describe and list (one per row): the kind of environment and conditions in which the  is required to " + vars["gpf"] + "; the kind of user, who is meant to make the " + vars["sys"] + " work correctly; the technical conditions that have to be respected in order to make the " + vars["sys"] + " work, or even needed tools.; the unchangeable different technical systems that can be influenced by the working of " + vars["sys"], "ssys", 2],
          [[16], "Mostly focusing on those needed to " + vars["gpf"] + ", list the relevant elements/components belonging to: " + vars["sys"] + " " + vars["obj"] + " " + vars["ben"] + " " + vars["ssys"] + " Type one component per row, using substantives without the article", "comp", 2],
          [[17], "Define the instant or the initial condition in which the " + vars["sys"] + " starts/start to " + vars["gpf"] + ".", "fotb", 3],
          [[18], "Define the instant or the end condition in which the " + vars["sys"] + " stops/stop to " + vars["gpf"] + ".", "fote", 3],
          [[20, 19], "Thus the " + vars["sys"] + " is meant to " + vars["gpf"] + " in the following delimited time interval Beginning: " + vars["fotb"] + " End: " + vars["fote"] + " Is it correct?"],
          [[16], "We will define again the time boundaries in which the " + vars["sys"] + " performs the action ‘‘to " + vars["gpf"] + "’’. Pay more attention on the mechanism and the boundary conditions that enables such a function. If useful think about the role of: " + vars["comp"] + " " + vars["ben"] + " " + vars["obj"] + " " + vars["ssys"] + " " + vars["pssys"]],
          [[21], "Define the physical space in which the " + vars["obj"] + " is/are properly modified by the " + vars["sys"] + " Please describe this area/volume as precisely as possible.", "foz", 3],
          [[22, 24], "Is the task you are approaching in order to " + vars["gpf"] + " related to the delivery of a new function by the " + vars["sys"] + "?"],
          [[23], "Type now which is the missing function that you want to implement within your " + vars["sys"], "nefu", 1],
          [[80], "Define the performance that the new function ‘‘to " + vars["nefu"] + "’’ should satisfy.", "perf", 0],
          [[77, 25], "Is your task related to an inadequate/unsatisfactory fulfillment of a desired benefit produced by the " + vars["sys"] + "?"],
          [[26], "Which is the undesired effect that arises in the system?", "oe", 0],
          [[27], "Which element of the " + vars["sys"] + ", or related to the " + vars["sys"] + ", causes the " + vars["oe"] + "? Some suggestions follow: " + vars["comp"] + " " + vars["ssys"] + " " + vars["sys"] + " itself/themselves " + vars["obj"] +  " " + vars["ben"], "voe", 0],
          [[28], "Thus the " + vars["sys"] + " is related to and interacts with " + vars["ben"] + " " + vars["obj"] + " " + vars["ssys"] + " " + vars["sys"] + "itself/themselves " + vars["pssys"] + " Type which, among these elements (and/or whatever element not belonging to the " + vars["sys"] + "), undergo undesired modifications or is subjected to negative impacts as a consequence of the " + vars["oe"] + " (one per row).", "crss", 2],
          [[29, 37], "Does the " + vars["voe"] + " provide any useful effect?"],
          [[30], "Define the instant or the initial condition in which the " + vars["voe"] + " starts/start to cause the " + vars["oe"] + ".", "potb", 3],
          [[31], "Define the instant or the end condition in which the " + vars["voe"] + " stops/stop to cause the " + vars["oe"] + ".", "pote", 3],
          [[33, 32], "Thus the " + vars["voe"] + " causes the " + vars["oe"] + " in the following delimited time interval Beginning: " + vars["potb"] + " End: " + vars["pote"] + " Is it correct?"],
          [[29], "We will define again the time boundaries in which the " + vars["voe"] + " causes the " + vars["oe"] + ". Pay more attention on the mechanism and the boundary conditions that enables such problem."],
          [[34], "Define the physical space in which the " + vars["oe"] + " appears/appear. Please describe this area/volume as precisely as possible.", "poz", 3],
          [[38, 35], "Are you aware of any parameter, influencing the " + vars["oe"] + ", and belonging to one of the following: " + vars["sys"] + " " + vars["obj"] + " " + vars["ben"] + " " + vars["ssys"], 0],
          [[36, 144], "Can you identify a different undesired effect, other than the " + vars["oe"]],
          [[26], "Define the new undesired effect.", "oe", 0],
          [[1, 129], "Thus, the " + vars["voe"] + " can be removed without any particular consequences. Do you want to formulate a new problem focusing on how to embody the new system without the " + vars["voe"] + "?"],
          [[39], "Type a list of parameters (one per row) concerning the " + vars["voe"] + " and capable to impact/influence " + vars["oe"] + ".", "lpvoe", 2],
          [[40], "Choose one of the parameters belonging to the following list, focusing on what mostly, if any, influence the " + vars["voe"] + ": " + vars["lpvoe"], "pvoe", 0],
          [[41], "In order to weaken the " + vars["oe"] + ", should I increase or decrease the " + vars["pvoe"] + " of the " + vars["voe"] + "? Type ‘‘increase’’ or ‘‘decrease’’", "nooe", 1],
          [[42], "In order to enforce the " + vars["oe"] + ", should I increase or decrease the " + vars["pvoe"] + "? Type ‘‘increase’’ or ‘‘decrease’’", "nuoe", 1],
          [[43, 74], "Do any bad consequences come out if you " + vars["nooe"] + " the " + vars["voe"] + "'s " + vars["pvoe"] + "?"],
          [[44, 67], "When we " + vars["nooe"] + " the " + vars["voe"] + "’s " + vars["pvoe"] + ", does this result in the weakening or in the disappearing of any positive effect or useful function concerning any of the following items? " + vars["sys"] + " " + vars["obj"] + " " + vars["ben"] + " " + vars["ssys"] + " " + vars["comp"]],
          [[45], "Type such positive effect", "pe", 0],
          [[46], "Define the instant or the initial condition in which the " + vars["pe"] + " start(s) to appear.", "uotb", 3],
          [[47], "Define the instant or the end condition in which the " + vars["pe"] + " stop(s) to appear.", "uote", 3],
          [[49, 48], "Thus the " + vars["pe"] + " appear(s) in the following delimited time interval Beginning: " + vars["uotb"] + " End: " + vars["uote"] + "Is it correct?"],
          [[45], "We will define again the time boundaries in which the " + vars["pe"] + "appear(s). Pay more attention on the mechanism and the boundary conditions that enables such positive effect."],
          [[50], "Delineate the physical space in which the " + vars["pe"] + " appears/appear. Please describe this area/volume as precisely as possible.", "uoz", 3],
          [[51, 62], "If you " + vars["nooe"] + " the " + vars["pvoe"] + " of the " + vars["voe"] + ", the " + vars["oe"] + " is weakened, but this results in the diminishment of the " + vars["pe"] + " too. Is it right?"],
          [[52, 62], "If you " + vars["nuoe"] + " the " + vars["pvoe"] + " of the " + vars["voe"] + ", the " + vars["pe"] + " increases, but the " + vars["oe"] + " is enforced too. Is it right?"],
          [[53, 58], "Now we need to exaggerate the problem in order to avoid compromises or tradeoffs. In order to accomplish this task let us start with this check. In the following is it written ‘‘increase’’? " + vars["nooe"]],
          [[54], "Let us assume that the " + vars["oe"] + " does not appear. Which maximum " + vars["pe"] + " would you like to achieve? Type a number (and unit of measure, if necessary), a substantive or an adjective (e.g. 100 °C, 3.5, 13 􏰀, π, external temperature, area of the circle, infinite, very long...)", "ppe", 4],
          [[55], "Which value should be assigned to the " + vars["voe"] + "’s " + vars["pvoe"] + " in order to achieve the value ‘‘" + vars["ppe"] + "’’? Type a number (and unit of measure, if necessary), a substantive or an adjective (e.g. 100 °C, 3.5, 13 􏰀, π, external temperature, area of the circle, infinite, very long. . . )", "pzp", 4],
          [[56], "What value will the " + vars["oe"] + " have as a consequence of setting the value of the " + vars["voe"] + "’s " + vars["pvoe"] + " to ‘‘" + vars["pzp"] + "’’? Type a number (and unit of measure, if necessary), a substantive or an adjective (e.g. 100 °C, 3.5, 13 􏰀, π, external temperature, area of the circle, infinite, very long. . . )", "poe", 4],
          [[57, 62], "Thus, we can get the value ‘‘" + vars["ppe"] + "’’ for the " + vars["pe"] + ", if the value of the " + vars["voe"] + "’s " + vars["pvoe"] + " is set to ‘‘" + vars["pzp"] + "’’. But, as consequence, this results in having a value ‘‘" + vars["poe"] + "’’ for the " + vars["oe"] + ". Is it right?"],
          [[145], "Now we need do find at least one element capable of removing the " + vars["oe"] + ", while preserving the value ‘‘" + vars["ppe"] + "’’ for the " + vars["pe"] + ", when the value of the " + vars["voe"] + "’s " + vars["pvoe"] + " is ‘‘" + vars["pzp"] + "’’. The best option would be finding such element among the items of the following list, including their characteristics or further modifications: " + vars["comp"] + " " + vars["obj"] + " " + vars["ssys"] + " " + vars["pssys"], 0],
          [[59], "Let us assume that there is no limitation in the simultaneous growth of the " + vars["pe"] + " and the " + vars["oe"] + ". What is the value of the maximum " + vars["pe"] + " that you would like to achieve? Type a number (and unit of measure, if necessary), a substantive or an adjective (e.g. 100 °C, 3.5, 13 􏰀, π, external temperature, area of the circle, infinite, very long. . . )", "ppe", 4],
          [[60, 62], "Let us assume that the value of the " + vars["voe"] + "’s " + vars["pvoe"] + " is 0 or a minimal value related to the external environment. Would the " + vars["oe"] + " disappear?"],
          [[61, 62], "Thus, we can significantly reduce or even eliminate the " + vars["oe"] + " if the value of the " + vars["voe"] + "’s " + vars["pvoe"] + " is 0 or a minimal value related to the external environment. In that case the " + vars["pe"] + " will significantly decrease or even disappear. Is it right?"],
          [[145], "Now we need to find at least one element capable of providing the value ‘‘" + vars["ppe"] + "’’ for the " + vars["pe"] + ", when the " + vars["pvoe"] + " of the " + vars["voe"] + " is 0 or a value related to the external environment, thus avoiding the " + vars["oe"] + ". The best option would be finding such element among the items of the following list, including their characteristics or further modifications: " + vars["comp"] + " " + vars["obj"] + vars["ssys"] + vars["pssys"], 0],
          [[63, 121], "Attempting to investigate the role of the " + vars["voe"] + "’s " + vars["pvoe"] + " did not bring to any successful formulation of the problem. Do you want to investigate further parameters of the " + vars["voe"] + "?"],
          [[64], "Here you can modify the list of the parameters concerning the " + vars["voe"] + ". We suggest to add further parameters, as well as to use combinations of the current parameters, or to split them (i.e., if you have ‘‘length’’ and ‘‘width’’, add ‘‘area’’; if you have ‘‘force’’, add ‘‘mass’’ and ‘‘acceleration’’).", "lpvoe", 2],
          [[65], "Choose one of the parameters belonging to the following list, focusing on what mostly, if any, influence the " + vars["oe"] + ": " + vars["lpvoe"], "pvoe", 0],
          [[66], "In order to weaken the " + vars["oe"] + ", should I increase or decrease the " + vars["pvoe"] + " of the " + vars["voe"] + "? Type ‘‘increase’’ or ‘‘decrease’’", "nooe", 4],
          [[142], "In order to enforce the " + vars["oe"] + ", should I increase or decrease the " + vars["pvoe"] + "? Type ‘‘increase’’ or ‘‘decrease’’", "nuoe", 4],
          [[68], "Briefly describe the bad consequences arising when we " + vars["nooe"] + " the " + vars["voe"] + "’s " + vars["pvoe"] + ".", "bcnooe", 0],
          [[69], "We should thus avoid getting the " + vars["bcnooe"] + ". Type the current positive condition occurring since the " + vars["voe"] + "’s " + vars["pvoe"] + " is/are not " + vars["nooe"] + "d.", "pe", 0],
          [[70], "Define the instant or the initial condition in which the " + vars["pe"] + " start(s) to appear.", "uotb", 3],
          [[71], "Define the instant or the end condition in which the " + vars["pe"] + " stop(s) to appear.", "uote", 3],
          [[73, 72], "Thus the " + vars["pe"] + " appear(s) in the following delimited time interval Beginning: " + vars["uotb"] + " End: " + vars["uote"] + " Is it correct?"],
          [[69], "We will define again the time boundaries in which the " + vars["pe"] + " appear(s). Pay more attention on the mechanism and the boundary conditions that enables such positive effect."],
          [[50], "Delineate the physical space in which the " + vars["pe"] + " appears/appear. Please describe this area/volume as precisely as possible.", "uoz", 3],
          [[], "It is thus worth to " + vars["nooe"] + " the " + vars["voe"] + "'s " + vars["pvoe"] + "."],
          [[76, 25], "Now we can either try to further reduce the " + vars["oe"] + ", working on the parameters of the " + vars["voe"] + ", or define other undesired effects appearing in the " + vars["sys"] + ". Do you want to keep on investigating the " + vars["oe"] + "?"],
          [[39], "Here you can modify the list of the parameters concerning the " + vars["voe"] + ". We suggest to add further parameters, as well as to use combinations of the current parameters, or to split them (i.e., if you have ‘‘length’’ and ‘‘width’’, add ‘‘area’’; if you have ‘‘force’’, add ‘‘mass’’ and ‘‘acceleration’’).", "lpvoe", 2],
          [[78], "Type now the name of the performance, feature or parameter, that is achieved at an unsatisfactory level for the " + vars["sys"] + ".", "perf", 0],
          [[79, 80], "Do you know any ways to improve the " + vars["perf"] + " even if this may lead to any kind of drawbacks?"],
          [[26], "Which is the undesired effect that arises in the system as a consequence of getting the satisfactory level of the " + vars["perf"] + "?", "oe", 0],
          [[81], "Type why the " + vars["perf"] + " of the " + vars["sys"] + " needs to be improved/introduced.", "rim", 1],
          [[82], "Thus, an enhancement in terms of " + vars["perf"] + " of the " + vars["sys"] + " should be carried out because of " + vars["rim"] + ". What or who would perceive these improvements? Some suggestions follow: " + vars["obj"] + " " + vars["ben"] + " " + vars["ssys"] + " " + vars["pssys"], "obim", 0],
          [[83], "What or who does not currently allow the enhancements in terms of " + vars["perf"] + " of the " + vars["sys"] + " that the " + vars["obim"] + " need(s)? Some suggestions follow: " + vars["ssys"] + " " + vars["obim"] + " parts (present or missing) of the " + vars["sys"] + " itself/themselves technological and environmental boundaries " + vars["comp"] + " " + vars["pssys"], "agim", 0],
          [[84, 85], "Do you have any possibility of modifying something among/between " + vars["agim"] + " " + vars["obim"] + "?"],
          [[4], "Type among/between " + vars["obim"] + " " + vars["agim"] + " what is easier to be modified.", "sys", 0],
          [[144], "Reformulate the undesired effect in terms of the poor " + vars["perf"] + " of the " + vars["sys"] + ".", "oe", 0],
          [[87], "Type now, the name of the performance, feature or parameter, that is achieved at an unsatisfactory level for the " + vars["sys"] + ".", "perf", 0],
          [[108, 80], "Is the " + vars["perf"] + " of the " + vars["sys"] + " determined during its design and manufacturing/delivering stage?"],
          [[89], "Now we will focus more carefully on the components of the " + vars["sys"] + " and on the resources it/they require(s) in order to " + vars["gpf"] + "."],
          [[95, 90], "Do the costs of the " + vars["sys"] + " (to buy, to transport, to maintain, to make it work) represent a critical issue?"],
          [[91], "Type the time that is required in order to " + vars["gpf"] + ", if it represents a critical issue for the " + vars["sys"] + ".", "crti", 0],
          [[92], "Type the physical spaces that are required in order to " + vars["gpf"] + ", if they represent a critical issue for the " + vars["sys"] + ".", "crsp", 2],
          [[93], "Type the energy that is required in order to " + vars["gpf"] + ", if it represents a critical issue for the " + vars["sys"] + ".", "cren", 0],
          [[94], "Type the materials that are required in order to " + vars["gpf"] + ", if they represent a critical issue for the " + vars["sys"] + ".", "crma", 2],
          [[136], "Type the information, the knowhow and the skills that are required in order to " + vars["gpf"] + ", if they represent a critical issue for the " + vars["sys"] + ".", "crin", 2],
          [[96], "Now we have to better define these incurred costs for user employing the " + vars["sys"] + ". We will carefully investigate when these costs mostly arise and what is responsible of the high expenditures."],
          [[97, 103], "Considering the main reasons for the expenditures, do the costs mostly occur when the " + vars["sys"] + " has to be disposed or, however, at the end of its lifecycle?"],
          [[98], "Describe with further details the phase, during the lifecycle of the " + vars["sys"] + ", that implies the most costs", "cpha", 3],
          [[99], "During the " + vars["cpha"] + ", are the costs due to aspects related to time consuming operations (e.g.: long interventions of personnel)? If yes, type the feature of the " + vars["sys"] + " resulting in high costs (e.g.: setup time, speed of process,...)", "crti", 0],
          [[100], "During the " + vars["cpha"] + ", are the costs due to aspects related to required room (e.g.: the rent of a big store, encumbrance,...)? If yes, type the feature of the " + vars["sys"] + " resulting in high costs (e.g.: volume of transported stuff,...)", "crsp", 0],
          [[101], "During the " + vars["cpha"] + ", are the costs due to aspects related to material consuming operations (e.g.: amount of components for maintenance)? If yes, type the feature of the " + vars["sys"] + " resulting in high costs (e.g.: number of substituted components,...)", "crma", 0],
          [[102], "During the " + vars["cpha"] + ", are the costs due to aspects related to energy consuming operations (e.g.: electrical power)? If yes, type the feature of the " + vars["sys"] + " resulting in high costs (e.g.: energy required to make the " + vars["sys"] + " work,...)", "cren", 0],
          [[136], "During the " + vars["cpha"] + ", are the costs due to aspects related to operations that need information or particular skills (e.g.: skills for carrying out the setup)? If yes, type the feature of the " + vars["sys"] + " resulting in high costs (e.g.: complexity in repairing " + vars["sys"] + ",...)", "crin", 0],
          [[97, 104], "Considering the main reasons for the expenditures, do the costs mostly occur when the " + vars["sys"] + " is being used (e.g. due to its consumptions)?"],
          [[97, 105], "Considering the main reasons for the expenditures, do the costs mostly occur before the " + vars["sys"] + " is being used (e.g. because of setup or fine-tuning)?"],
          [[106, 107], "Considering the main reasons for the expenditures, do the costs mostly occur when the " + vars["sys"] + " is purchased by the final user?"],
          [[108], "It seems that we are dealing with high costs related to the production process of the " + vars["sys"]],
          [[98], "Considering the main reasons for the expenditures, type the phase, during the lifecycle of the " + vars["sys"] + ", that implies the highest costs.", "cpha", 0],
          [[109], "Thus, the " + vars["sys"] + " and its production process should be redesigned."],
          [[110], "Type the required time to perform the design and/or some manufacturing/delivering phases for the " + vars["sys"] + ", if it represents a critical issue.", "pcrti", 0],
          [[111], "Type the required physical spaces to perform the design and/or some manufacturing/delivering phases for the " + vars["sys"] + ", if they represent a critical issue.", "pcrsp", 2],
          [[112], "Type the required energy to perform the design and/or some manufacturing/delivering phases for the " + vars["sys"] + ", if it represents a critical issue.", "pcren", 0],
          [[113], "Type the required materials to perform the design and/or some manufacturing/delivering phases for the " + vars["sys"] + ", if they represent a critical issue.", "pcrma", 2],
          [[114], "Type the information, the know how and the required skills to perform the design and/or some manufacturing/delivering phases for the " + vars["sys"] + ", if they represent a critical issue.", "pcrin", 2],
          [[115], "Describe and list, one item per row: the kind of environment and conditions in which the " + vars["sys"] + " is designed and manufactured (e.g.: vacuum, high temperature,...); the kind of designer, developer or worker who is meant to design or produce the " + vars["sys"] + " (e.g.: high specialized working team,...); the kind of technologies, techniques or tools required to design or produce the " + vars["sys"] + " (e.g.: 3D-CAD; CNC center,...); different systems with which the " + vars["sys"] + " has to interact during the design and production process (e.g.: X-ray check,...)", "pssys", 2],
          [[116], "Thus the " + vars["sys"] + " is related to " + vars["penv"] + " " + vars["puser"] + " " + vars["ptech"] + " " + vars["posys"] + " Type (if any) the one(s), among these items, whose properties or requirements are affected or threatened by the " + vars["oe"] + ".", "pcrss", 2],
          [[117], "The " + vars["sys"] + " is related to: " + vars["penv"] + " " + vars["puser"] + " " + vars["ptech"] + " " + vars["posys"] + " Type (if any) which, among these items, cause high costs.", "pcrco", 2],
          [[144, 118], "Is the following list empty? " + vars["pcrti"] + " " + vars["pcrsp"] + " " + vars["pcren"] + " " + vars["pcrma"] + " " + vars["pcrin"] + " " + vars["pcrss"] + " " + vars["pcrco"]],
          [[143], "Choose what represent the most critical matter for the " + vars["sys"] + " among the following issues: Required time: " + vars["pcrti"] + " Required space: " + vars["pcrsp"] + " Required energy: " + vars["pcren"] + " Required material: " + vars["pcrma"] + " Required information: " + vars["pcrin"] + " High costs caused by: " + vars["pcrco"] + " Negative impacts on: " + vars["pcrss"], "oe", 0],
          [[120], "Type the component of the " + vars["sys"] + " or its design/production phase mostly or entirely responsible for the " + vars["oe"] + ".", "voe", 0],
          [[28], "Type now, what or who, external from the " + vars["sys"] + " itself/themselves, is mostly affected by the " + vars["oe"] + ", caused by the " + vars["voe"] + ". Some suggestions follow: " + vars["pssys"] + " The " + vars["sys"] + " itself/themselves " + vars["ssys"], "crss", 0],
          [[123, 122], "Thus, the " + vars["sys"] + " is meant to " + vars["gpf"] + " between the instant/condition ‘‘" + vars["fotb"] + "’’ and the final instant/condition ‘‘" + vars["fote"] + "’’. As well, the " + vars["voe"] + " causes/cause the " + vars["oe"] + " between the instant/condition ‘‘" + vars["potb"] + "’’ and the final instant/condition ‘‘" + vars["pote"] + "’’. Do these two time intervals overlap?"],
          [[123], "SUGGESTION: Think about the opportunities to redesign the " + vars["sys"] + ", taking advantage of the circumstances that the performing of the function ‘‘to " + vars["gpf"] + "’’ does not overlap with the " + vars["oe"] + ", since these actions occur in different moments."],
          [[125, 124], "The " + vars["sys"] + " is meant to " + vars["gpf"] + " in the " + vars["foz"] + ". As well, the " + vars["voe"] + " causes/cause the " + vars["oe"] + " in the " + vars["poz"] + ". Do these two space regions overlap?"],
          [[125], "SUGGESTION: Think about the opportunities to redesign the " + vars["sys"] + ", taking advantage of the circumstances that the performing of the function ‘‘to " + vars["gpf"] + "’’ does not overlap with the " + vars["oe"] + ", since these actions occur in different space regions."],
          [[127, 126], "Thus, the " + vars["voe"] + " produces the " + vars["pe"] + " from the instant/condition ‘‘" + vars["uotb"] + "’’ to the final instant/condition ‘‘" + vars["uote"] + "’’. As well, the " + vars["voe"] + " causes/cause the " + vars["oe"] + " between the instant/condition ‘‘" + vars["potb"] + "’’ and the final instant/condition ‘‘" + vars["pote"] + "’’. Do these two time intervals overlap?"],
          [[127], "SUGGESTION: Think about the opportunities to redesign the " + vars["sys"] + ", taking advantage of the circumstances that the " + vars["pe"] + " doesn’t overlap with the " + vars["oe"] + ", since these actions occur in different moments."],
          [[129, 128], "The " + vars["voe"] + " produce(s) the " + vars["pe"] + " in the " + vars["uoz"] + ". As well, the " + vars["voe"] + " causes/cause the " + vars["oe"] + " in the " + vars["poz"] + ". Do these two space regions overlap?"],
          [[129], "SUGGESTION: Think about the opportunities to redesign the " + vars["sys"] + ", taking advantage of the circumstances that the " + vars["pe"] + " does not overlap with the " + vars["oe"] + ", since these actions occur in different space regions."],
          [[130], "SUGGESTIONS (1) You can reformulate the problem, paying attention on what surrounds the " + vars["sys"] + " with a particular focus on the " + vars["crss"] + " (2) You can reformulate the problem, focusing on the components of the " + vars["sys"] + "; choose one of the following items: " + vars["comp"] + " (3) Reformulate the problem, thinking about the opportunities to turn the " + vars["oe"] + " into a useful effect for some of the following elements: " + vars["ben"] + " " + vars["sys"] + " itself/themselves " + vars["obj"] + " " + vars["comp"] + " " + vars["ssys"] + " " + vars["pssys"] + " (4) Reformulate the problem introducing a different system in order to " + vars["goal"] + ", as requested by the " + vars["ben"]],
          [[1, 131], "Following the previous suggestions, do you think that you can advantageously reformulate the problem?"],
          [[132, 144], "Do you want to further investigate and characterize the underlying problems of the " + vars["sys"] + ", thus revealing some aspects that have not been yet elicited?"],
          [[108, 133], "Does the " + vars["oe"] + " occur when the " + vars["sys"] + " is manufactured, prepared or delivered?"],
          [[88, 134], "Does the " + vars["oe"] + " arise during the use of the " + vars["sys"] + " and/or between stops and restarts?"],
          [[135, 144], "Does the " + vars["oe"] + " characterize the whole lifecycle of the " + vars["sys"] + "?"],
          [[108], "It seems that we are dealing with a problem that should be faced during the design of the " + vars["sys"]],
          [[137, 138], "We have tried to investigate the relevant problems of the " + vars["sys"] + " during its use. Is the following list empty? List: " + vars["crti"] + " " + vars["crsp"] + " " + vars["cren"] + " " + vars["crma"] + " " + vars["crin"]],
          [[79, 144], "Is the " + vars["oe"] + " related to some unsatisfactory performances of the " + vars["sys"] + "?"],
          [[144, 139], "Thus, we know that we have " + vars["oe"] + " mostly affecting " + vars["crss"] + " Moreover, in order to " + vars["gpf"] + ", the " + vars["sys"] + " requires, in its lifecycle: time: " + vars["crti"] + " space: " + vars["crsp"] + " energy: " + vars["cren"] + " material: " + vars["crma"] + " information: " + vars["crin"] + " By this checklist, do you still think that is correct to focus on the " + vars["oe"] + "? Otherwise we can redefine the problem considering the resources, the requirements and the costs related with the " + vars["sys"] + " lifecycle."],
          [[140], "Type among: " + vars["crti"] + " " + vars["crsp"] + " " + vars["cren"] + " " + vars["crma"] + " " + vars["crin"] + " what represents the most critical issue.", "crre", 0],
          [[141], "Our new problem is: during the " + vars["sys"] + " lifecycle, the " + vars["sys"] + " itself/themselves or some of its/their component require(s) " + vars["crre"] + " in order to " + vars["gpf"] + " and/or to fulfill some subtask. Now reformulate the undesired effect.", "oe", 0],
          [[26, 142], "As we have reformulated the problem, ‘‘the " + vars["oe"] + " occurs in the " + vars["sys"] + "’’. If this statement is not correct, we had better redefine our system, otherwise we will keep investigating with the available data. Is the formulation correct?"],
          [[4], "Type the name of the system in which the " + vars["oe"] + " occurs. Some suggestions follow: " + vars["crre"] + " " + vars["comp"] + " " + vars["ben"] + " " + vars["obj"], "sys", 0],
          [[119, 142], "As we have reformulated the problem, ‘‘the " + vars["oe"] + " occurs in the " + vars["sys"] + "’’. If this statement is not correct, we had better redefine our system, otherwise we will keep investigating with the available data. Is the formulation correct?"],
          [[], "Unfortunately we did not succeed in performing a thorough description of the problem in the terms required by the procedure. However, we will perform an attempt to find interesting patents. Be aware that the set of results could contain some non-pertinent documents."],
          [[], "Thank you for your attention! Now you have a preliminary ‘‘query’’ for asking a patent database in order to get useful information about the " + vars["sys"] + "."]
          ]

def var_bank():
  return {"sys" : "",
          "goal" : "",
          "gpf" : "",
          "p1" : "",
          "p2" : "",
          "ben" : "",
          "obj" : "",
          "ssys" : "",
          "comp" : "",
          "fotb" : "",
          "fote" : "",
          "foz" : "",
          "nefu" : "",
          "perf" : "",
          "oe" : "",
          "voe" : "",
          "crss" : "",
          "potb" : "",
          "pote" : "",
          "poz" : "",
          "lpvoe" : "",
          "pvoe" : "",
          "nooe" : "",
          "nuoe" : "",
          "pe" : "",
          "uotb" : "",
          "uote" : "",
          "uoz" : "",
          "ppe" : "",
          "pzp" : "",
          "poe" : "",
          "bcnooe" : "",
          "rim" : "",
          "obim" : "",
          "agim" : "",
          "crti" : "",
          "crsp" : "",
          "cren" : "",
          "crma" : "",
          "crin" : "",
          "cpha" : "",
          "cren" : "",
          "pcrti" : "",
          "pcrsp" : "",
          "pcren" : "",
          "pcrma" : "",
          "pcrin" : "",
          "pssys" : "",
          "pcrss" : "",
          "pcrco" : "",
          "crre" : "",
          "penv" : "",
          "puser" : "",
          "ptech" : "",
          "posys" : "",
          "background_info" : ""}

def main():
  question = "What product or system currently antifogs space helmets?"
  # question = "What product or system currently keeps buildings cool in the summer?"
  # question = "What product or system currently reduces stormwater runoff and flooding in cities?"
  # question = "What product or system currently reduces the use of toxic substances in paints?"

  vars = var_bank()
  sys = generate_response(vars["background_info"] + "\nQ: " + question + " " + answer_format_addons[0] + " A: ").strip().lower()
  vars["sys"] = sys
  vars["background_info"] += "Q: " + question + " A: " + sys

  index_in_questions = 2
  while index_in_questions != None:
    question_bank = questions(vars)
    current_question = question_bank[index_in_questions - 2]
    if len(current_question) == 2:
      if len(current_question[0]) == 1 or len(current_question[0]) == 0:
        index_in_questions = next_q_novar(current_question)
      if len(current_question[0]) == 2:
        index_in_questions = yn_q_novar(current_question)
    elif len(current_question) == 4:
      index_in_questions = next_q_var(current_question)
    else:
      print("something went wrong")
      break
  print(vars)
  
if __name__ == "__main__":
  main()