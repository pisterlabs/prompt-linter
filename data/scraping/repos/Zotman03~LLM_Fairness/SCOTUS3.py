# This is for SCOTUS dataset on GPT3
from datasets import load_dataset
import openai
import os
from dotenv import load_dotenv
import time
import re
from sklearn.metrics import f1_score
import math

# Dataset loading and API
load_dotenv()
api_key = os.getenv('NEXT_PUBLIC_OPENAI_API_KEY')
openai.api_key = api_key
dataset = load_dataset('coastalcph/fairlex', 'scotus', split='train')

# Example data
text = dataset[0]['text']

decision_dir = {"0": ([0], [0], [], []), "1": ([0], [0], [], [])} #Dictionary for decision direction. Tuple: (0: conservative, 1: liberal){inside tuple: Total, correct, truelable, reslabel}
res_type = {"0": ([0], [0], [], []), "1": ([0], [0], [], []), "2": ([0], [0], [], []), "3": ([0], [0], [], []), "4": ([0], [0], [], [])}
#similar to decision direction: total, correct, true, predicted. But it is for seven province regions

# Numbers
total = 0
total_right = 0
buffer = 0
is_first = True
found = 0
# Loop now
for example in dataset:
    if(is_first is True):
       is_first = False
       continue # Check for the first time, and will never be checked again

    if(found == 100):
       break
    
    if(str(example['respondent_type']) == "4"):
      found += 1
      input_text = example['text']
      input_ans = example['label']
      input_direction = example['decision_direction']
      input_res = example['respondent_type']

      prompt1 = f"As a legal advisor, I specialize in providing guidance on various legal situations. Please describe the specific legal situation you need help with, and I will select the most appropriate label from the following options: (0, Criminal Procedure), (1, Civil Rights), (2, First Amendment), (3, Due Process), (4, Privacy), (5, Attorneys), (6, Unions), (7, Economic Activity), (8, Judicial Power), (9, Federalism), (10, Interstate Relations), (11, Federal Taxation), (12, Miscellaneous), (13, Private Action). It's important to include all relevant details related to the situation to ensure accurate advice. What would be the best corresponding label of the legal situation: {input_text[:3900]} will be? You should only reply with the index number (range from 0 to 13)"
      completion = openai.Completion.create(
        engine="text-davinci-003",  # Change this to the Davinci 3 model
        prompt=prompt1,
        temperature=0
      )

      if(completion['choices'][0]['text'] == str(input_ans)): # Check if the predicted label is equal to actual label.
          total_right += 1
          decision_dir[str(input_direction)][1][0] += 1
          res_type[str(input_res)][1][0] += 1
          #appending correct label
      
      else: #A safe layer to check if the result is correct but format issue causing it to receive wrong answer
        if(len(completion['choices'][0]['text']) > 1):
            match = re.search(r'\d+', completion['choices'][0]['text']) #Regular expression to make sure there is only one item here.
            if match:
                completion['choices'][0]['text'] = str(match.group())
                if completion['choices'][0]['text'] == str(input_ans): #check if it is the correct label
                  total_right += 1 #Total correct append
                  decision_dir[str(input_direction)][1][0] += 1
                  res_type[str(input_res)][1][0] += 1

      #If the result is wrong then it goes here.
      decision_dir[str(input_direction)][2].append(str(input_ans))
      decision_dir[str(input_direction)][3].append(completion['choices'][0]['text'])
      res_type[str(input_res)][2].append(str(input_ans))
      res_type[str(input_res)][3].append(completion['choices'][0]['text'])
      # total++
      decision_dir[str(input_direction)][0][0] += 1
      res_type[str(input_res)][0][0] += 1
      
      #Add 1 to the total number
      total += 1
      print(total, " out of 1000 complete")
      buffer += 1
      if(buffer % 10 == 0):
        time.sleep(10)
      if(buffer % 200 == 0):
         time.sleep(120)

print("Using GPT3")
print("For other this is the total and total correct ", res_type["0"][0][0], " ----", res_type["0"][1][0])
print("For person this is the total and total correct ", res_type["1"][0][0], " ----", res_type["1"][1][0])
print("For organization this is the total and total correct ", res_type["2"][0][0], " ----", res_type["2"][1][0])
print("For public entity this is the total and total correct ", res_type["3"][0][0], " ----", res_type["3"][1][0])
print("For facility this is the total and total correct ", res_type["4"][0][0], " ----", res_type["4"][1][0])

f1_scores_other = f1_score(res_type["0"][2], res_type["0"][3], average="macro")
f1_scores_person = f1_score(res_type["1"][2], res_type["1"][3], average="macro")
f1_scores_org = f1_score(res_type["2"][2], res_type["2"][3], average="macro")
f1_scores_pe = f1_score(res_type["3"][2], res_type["3"][3], average="macro")
f1_scores_facil = f1_score(res_type["4"][2], res_type["4"][3], average="macro")

print("mF1 Score for other:", f1_scores_other)
print("mF1 Score for person:", f1_scores_person)
print("mF1 Score for organization:", f1_scores_org)
print("mF1 Score for public entity:", f1_scores_pe)
print("mF1 Score for facility:", f1_scores_facil)

ave_f1_scores_res_type = (0.0769607843137255 + 0.08265669515669516 + 0.18563867576015913 + 0.07088907469342252 + f1_scores_facil) / 5

GD_res = math.sqrt(0.2 * math.pow(0.0769607843137255 - ave_f1_scores_res_type, 2) * math.pow(0.08265669515669516 - ave_f1_scores_res_type, 2) * math.pow(0.18563867576015913 - ave_f1_scores_res_type, 2) * math.pow(0.07088907469342252 - ave_f1_scores_res_type, 2) * math.pow(f1_scores_facil - ave_f1_scores_res_type, 2))
print("The mf1 average is:", ave_f1_scores_res_type)
print("The GD score is:", GD_res)
print("The worst mf1 score is:", min(0.0769607843137255, 0.08265669515669516, 0.18563867576015913, 0.07088907469342252, f1_scores_facil))
