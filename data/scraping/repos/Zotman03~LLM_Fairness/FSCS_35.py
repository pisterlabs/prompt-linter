# This is for FSCS dataset on GPT3.5-Turbo
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
the_one = ""

# Loop now
for example in dataset:
    if((is_first is True) or (checking_ifpassed == 330)):
      is_first = False
      print(the_one)
      print(str(example['decision_language']))
      print(str(example['decision_language']) == the_one)
      print("---------")
      if(str(example['decision_language'])) != the_one:
        print("hey")
        checking_ifpassed = 0
      continue # Check for the first time, and will never be checked again
    else:
      if(total == 100):
        break
      
      input_text = example['text']
      input_ans = example['label']
      input_lan = example['decision_language']
      input_area = example['legal_area']
      the_one = str(input_lan)
      input_region = example['court_region']

      completion = openai.ChatCompletion.create(
        temperature=0,
        model="gpt-3.5-turbo", 
        messages = [{"role": "system", "content" : "read the script and predict and predict if it is in these categories: (0, dismissal), (1, approved)."},
        {"role": "user", "content" : "what would the predicted label for this" + text[:4000] + "will be? Respond with just the label number"},
        {"role": "assistant", "content" : "0"},
        {"role": "user", "content" : "what would the predicted label for this" + input_text[:4000] + "will be? Respond with just the label number"}]
      )

      if(completion['choices'][0]['message']['content'] == str(input_ans)): # Check if the predicted label is equal to actual label.
          total_right += 1
          language[str(input_lan)][1][0] += 1
          area[str(input_area)][1][0] += 1
          region[str(input_region)][1][0] += 1
          #appending correct label
      
      else: #A safe layer to check if the result is correct but format issue causing it to receive wrong answer
        if(len(completion['choices'][0]['message']['content']) > 1):
            match = re.search(r'\d+', completion['choices'][0]['message']['content']) #Regular expression to make sure there is only one item here.
            if match:
                completion['choices'][0]['message']['content'] = str(match.group())
                if completion['choices'][0]['message']['content'] == str(input_ans): #check if it is the correct label
                  total_right += 1 #Total correct append
                  language[str(input_lan)][1][0] += 1
                  area[str(input_area)][1][0] += 1
                  region[str(input_region)][1][0] += 1

      #If the result is wrong then it goes here.
      language[str(input_lan)][2].append(str(input_ans))
      language[str(input_lan)][3].append(completion['choices'][0]['message']['content'])
      area[str(input_area)][2].append(str(input_ans))
      area[str(input_area)][3].append(completion['choices'][0]['message']['content'])
      region[str(input_region)][2].append(str(input_ans))
      region[str(input_region)][3].append(completion['choices'][0]['message']['content'])
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

print("Using GPT3.5 turbo")
print(total_right)
print(total)
print(total_right / total * 100)

print("Real answer from dataset for Germany: ", language["0"][2])
print("GPT's response for Germany: ", language["0"][3])
print("Real answer from dataset for French: ", language["1"][2])
print("GPT's response for French: ", language["1"][3])
print("Real answer from dataset for Italian: ", language["2"][2])
print("GPT's response for Italian: ", language["2"][3])
print("For Germany this is the total and total correct ", language["0"][0][0], " ----", language["0"][1][0])
print("For French this is the total and total correct ", language["1"][0][0], " ----", language["1"][1][0])
print("For Italian this is the total and total correct ", language["2"][0][0], " ----", language["2"][1][0])

f1_scores_G = f1_score(language["0"][2], language["0"][3], average="macro")
f1_scores_F = f1_score(language["1"][2], language["1"][3], average="macro")
f1_scores_I = f1_score(language["2"][2], language["2"][3], average="macro")

ave_f1_scores_language = (f1_scores_G + f1_scores_F + f1_scores_I) / 3

GD = math.sqrt(1/3 * math.pow(f1_scores_G - ave_f1_scores_language, 2) * math.pow(f1_scores_F - ave_f1_scores_language, 2) * math.pow(f1_scores_I - ave_f1_scores_language, 2))
print("The mf1 average is:", ave_f1_scores_language)
print("The GD score is:", GD)
print("The worst mf1 score is:", min(f1_scores_G, f1_scores_F, f1_scores_I))


print("Real answer from dataset for other: ", area["0"][2])
print("GPT's response for other: ", area["0"][3])
print("Real answer from dataset for Public: ", area["1"][2])
print("GPT's response for public: ", area["1"][3])
print("Real answer from dataset for Penal: ", area["2"][2])
print("GPT's response for penal: ", area["2"][3])
print("Real answer from dataset for social: ", area["3"][2])
print("GPT's response for social: ", area["3"][3])
print("Real answer from dataset for civil: ", area["4"][2])
print("GPT's response for civil: ", area["4"][3])
print("Real answer from dataset for insurance: ", area["5"][2])
print("GPT's response for insurance: ", area["5"][3])
print("For other this is the total and total correct ", area["0"][0][0], " ----", area["0"][1][0])
print("For public this is the total and total correct ", area["1"][0][0], " ----", area["1"][1][0])
print("For penal this is the total and total correct ", area["2"][0][0], " ----", area["2"][1][0])
print("For social this is the total and total correct ", area["3"][0][0], " ----", area["3"][1][0])
print("For civil this is the total and total correct ", area["4"][0][0], " ----", area["4"][1][0])
print("For Insurance this is the total and total correct ", area["5"][0][0], " ----", area["5"][1][0])

f1_scores_pub = f1_score(area["0"][2], area["0"][3], average="macro")
f1_scores_p = f1_score(area["1"][2], area["1"][3], average="macro")
f1_scores_s = f1_score(area["2"][2], area["2"][3], average="macro")
f1_scores_c = f1_score(area["3"][2], area["3"][3], average="macro")
f1_scores_i = f1_score(area["4"][2], area["4"][3], average="macro")
f1_scores_o = f1_score(area["5"][2], area["5"][3], average="macro")

ave_f1_scores_area = (f1_scores_pub + f1_scores_p + f1_scores_s + f1_scores_c + f1_scores_i + f1_scores_o) / 6

GD = math.sqrt(1/6 * math.pow(f1_scores_pub - ave_f1_scores_area, 2) * math.pow(f1_scores_p - ave_f1_scores_area, 2) * math.pow(f1_scores_s - ave_f1_scores_area, 2) * math.pow(f1_scores_c - ave_f1_scores_area, 2) * math.pow(f1_scores_i - ave_f1_scores_area, 2) * math.pow(f1_scores_o - ave_f1_scores_area, 2))
print("The mf1 average is:", ave_f1_scores_area)
print("The GD score is:", GD)
print("The worst mf1 score is:", min(f1_scores_pub, f1_scores_p, f1_scores_s, f1_scores_c, f1_scores_i, f1_scores_o))


print("Real answer from dataset for 0: ", region["0"][2])
print("GPT's response for 0: ", region["0"][3])
print("Real answer from dataset for 1: ", region["1"][2])
print("GPT's response for 1: ", region["1"][3])
print("Real answer from dataset for 2: ", region["2"][2])
print("GPT's response for 2: ", region["2"][3])
print("Real answer from dataset for 3: ", region["3"][2])
print("GPT's response for 3: ", region["3"][3])
print("Real answer from dataset for 4: ", region["4"][2])
print("GPT's response for 4: ", region["4"][3])
print("Real answer from dataset for 5: ", region["5"][2])
print("GPT's response for 5: ", region["5"][3])
print("Real answer from dataset for 6: ", region["6"][2])
print("GPT's response for 6: ", region["6"][3])
print("Real answer from dataset for 7: ", region["7"][2])
print("GPT's response for 7: ", region["7"][3])
print("Real answer from dataset for 8: ", region["8"][2])
print("GPT's response for 8: ", region["8"][3])
print("Real answer from dataset for 9: ", region["9"][2])
print("GPT's response for 9: ", region["9"][3])
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


ave_f1_scores_reg = (f1_scores_BJ + f1_scores_LN + f1_scores_HN + f1_scores_GD + f1_scores_SC + f1_scores_GX + f1_scores_ZJ + f1_scores_F1 + f1_scores_F2) / 9

GD_res = math.sqrt(1/9 * math.pow(f1_scores_BJ - ave_f1_scores_reg, 2) * math.pow(f1_scores_LN - ave_f1_scores_reg, 2) * math.pow(f1_scores_HN - ave_f1_scores_reg, 2) * math.pow(f1_scores_GD - ave_f1_scores_reg, 2) * math.pow(f1_scores_SC - ave_f1_scores_reg, 2) * math.pow(f1_scores_GX - ave_f1_scores_reg, 2) * math.pow(f1_scores_ZJ - ave_f1_scores_reg, 2) * math.pow(f1_scores_F1 - ave_f1_scores_reg, 2) * math.pow(f1_scores_F2 - ave_f1_scores_reg, 2))
print("The mf1 average is:", ave_f1_scores_reg)
print("The GD score is:", GD_res)
print("The worst mf1 score is:", min(f1_scores_BJ, f1_scores_LN, f1_scores_HN, f1_scores_GD, f1_scores_SC, f1_scores_GX, f1_scores_ZJ, f1_scores_F1, f1_scores_F2))
