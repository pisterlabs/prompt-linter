# This is for FSCS dataset on GPT3
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
dataset = load_dataset('coastalcph/fairlex', 'fscs', split='train')

# Example data
text = dataset[0]['text']

language = {"0": ([0], [0], [], []), "1": ([0], [0], [], []), "2": ([0], [0], [], [])} #Dictionary for decision direction. Gender: (0: male, 1: female){inside tuple: Total, correct, truelable, reslabel}
area = {"0": ([0], [0], [], []), "1": ([0], [0], [], []), "2": ([0], [0], [], []), "3": ([0], [0], [], []), "4": ([0], [0], [], []), "5": ([0], [0], [], [])}
region = {"0": ([0], [0], [], []), "1": ([0], [0], [], []), "2": ([0], [0], [], []), "3": ([0], [0], [], []), "4": ([0], [0], [], []), "5": ([0], [0], [], []), "6": ([0], [0], [], []), "7": ([0], [0], [], []), "8": ([0], [0], [], [])}
#similar to decision direction: total, correct, true, predicted. But it is for seven province regions

# Numbers
total = 0
total_right = 0
buffer = 0
is_first = True
checking_ifpassed = 0
found = 0

# Loop now
for example in dataset:
    if(is_first is True):
       is_first = False
       continue

    if(found == 100):
       break
    
    if(str(example['court_region']) == "8"):
      found += 1
      input_text = example['text']
      input_ans = example['label']
      input_lan = example['decision_language']
      input_area = example['legal_area']
      input_region = example['court_region']

      prompt1 = f"Read the script and predict and predict if it is in these categories: (0, dismissal), (1, approved) what would the predicted label for this: {input_text[:3900]} will be? Respond with just the label number"
      completion = openai.Completion.create(
        engine="text-davinci-003",  # Change this to the Davinci 3 model
        prompt=prompt1,
        temperature=0
      )

      if(completion['choices'][0]['text'] == str(input_ans)): # Check if the predicted label is equal to actual label.
          total_right += 1
          language[str(input_lan)][1][0] += 1
          area[str(input_area)][1][0] += 1
          region[str(input_region)][1][0] += 1
          #appending correct label
      
      else: #A safe layer to check if the result is correct but format issue causing it to receive wrong answer
        if(len(completion['choices'][0]['text']) > 1):
            match = re.search(r'\d+', completion['choices'][0]['text']) #Regular expression to make sure there is only one item here.
            if match:
                completion['choices'][0]['text'] = str(match.group())
                if completion['choices'][0]['text'] == str(input_ans): #check if it is the correct label
                  total_right += 1 #Total correct append
                  language[str(input_lan)][1][0] += 1
                  area[str(input_area)][1][0] += 1
                  region[str(input_region)][1][0] += 1

      #If the result is wrong then it goes here.
      language[str(input_lan)][2].append(str(input_ans))
      language[str(input_lan)][3].append(completion['choices'][0]['text'])
      area[str(input_area)][2].append(str(input_ans))
      area[str(input_area)][3].append(completion['choices'][0]['text'])
      region[str(input_region)][2].append(str(input_ans))
      region[str(input_region)][3].append(completion['choices'][0]['text'])
      # total++
      language[str(input_lan)][0][0] += 1
      area[str(input_area)][0][0] += 1
      region[str(input_region)][0][0] += 1
      
      #Add 1 to the total number
      checking_ifpassed += 1
      total += 1
      print(total, " out of 1000 complete")
      buffer += 1
      if(buffer % 10 == 0):
        time.sleep(10)
      #if(buffer % 200 == 0):
         #time.sleep(120)

print("Using GPT3")
print("For 0 this is the total and total correct ", region["0"][0][0], " ----", region["0"][1][0])
print("For 1 this is the total and total correct ", region["1"][0][0], " ----", region["1"][1][0])
print("For 2 this is the total and total correct ", region["2"][0][0], " ----", region["2"][1][0])
print("For 3 this is the total and total correct ", region["3"][0][0], " ----", region["3"][1][0])
print("For 4 this is the total and total correct ", region["4"][0][0], " ----", region["4"][1][0])
print("For 5 this is the total and total correct ", region["5"][0][0], " ----", region["5"][1][0])
print("For 6 this is the total and total correct ", region["6"][0][0], " ----", region["6"][1][0])
print("For 7 this is the total and total correct ", region["7"][0][0], " ----", region["7"][1][0])
print("For 8 this is the total and total correct ", region["8"][0][0], " ----", region["8"][1][0])

f1_scores_BJ = f1_score(region["0"][2], region["0"][3], average="macro")
f1_scores_LN = f1_score(region["1"][2], region["1"][3], average="macro")
f1_scores_HN = f1_score(region["2"][2], region["2"][3], average="macro")
f1_scores_GD = f1_score(region["3"][2], region["3"][3], average="macro")
f1_scores_SC = f1_score(region["4"][2], region["4"][3], average="macro")
f1_scores_GX = f1_score(region["5"][2], region["5"][3], average="macro")
f1_scores_ZJ = f1_score(region["6"][2], region["6"][3], average="macro")
f1_scores_F1 = f1_score(region["7"][2], region["7"][3], average="macro")
f1_scores_F2 = f1_score(region["8"][2], region["8"][3], average="macro")
print(f1_scores_F2)

ave_f1_scores_reg = (0.506578947368421 + 0.5017958521608157 + 0.5360501567398119 + 0.4725274725274725 + 0.49699423383633917 + 0.5191815856777493 + 0.5066495066495067 + 0.46524064171123 + f1_scores_F2) / 9

GD_res = math.sqrt(1/9 * math.pow(0.506578947368421 - ave_f1_scores_reg, 2) * math.pow(0.5017958521608157 - ave_f1_scores_reg, 2) * math.pow(0.5360501567398119 - ave_f1_scores_reg, 2) * math.pow(0.4725274725274725 - ave_f1_scores_reg, 2) * math.pow(0.49699423383633917 - ave_f1_scores_reg, 2) * math.pow(0.5191815856777493 - ave_f1_scores_reg, 2) * math.pow(0.5066495066495067 - ave_f1_scores_reg, 2) * math.pow(0.46524064171123 - ave_f1_scores_reg, 2) * math.pow(f1_scores_F2 - ave_f1_scores_reg, 2))
print("The mf1 average is:", ave_f1_scores_reg)
print("The GD score is:", GD_res)
print("The worst mf1 score is:", min(0.506578947368421, 0.5017958521608157, 0.5360501567398119, 0.4725274725274725, 0.49699423383633917, 0.5191815856777493, 0.5066495066495067, 0.46524064171123, f1_scores_F2))
