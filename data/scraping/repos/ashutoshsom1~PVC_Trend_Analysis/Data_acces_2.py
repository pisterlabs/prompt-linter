#function to get the answer from GPT-3
import openai
import time
 
openai.api_key='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'
 

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.3, # this is the degree of randomness of the model's output
     )
    return response.choices[0].message["content"]




def get_answers_few_shot_approach_(text):
    prompt = f"""
    Your task is to give the summary of a given text \    
    text is followed by tripple back ticks  ```{text}```  \ 
    dont loose the information in the text \
    summary should be with in 370  tokens  \
    
    """
    answer = get_completion(prompt)
    return answer   

import re

def remove_special_characters(text):
    # Define the pattern to match special characters
    pattern = r'[^a-zA-Z0-9\s]'
    
    # Remove special characters using regex
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text     


########################################################################################
all_text = open(r"C:\Users\ashutosh.somvanshi\PVC_Trend_Analysis\Supreme_Management_Discussion_Analysis\Supreme_Text_files\All_txt_summary.txt", 'rb')
all_text_=all_text.read()
# all_ = str(all_text_).split(" ")
# all_= remove_special_characters(str(all_text_))
# all__ = all_.replace("\\n", " ")

original_string = str(all_text_)
part_length = 3
num_parts = len(original_string) // part_length
remainder = len(original_string) % part_length

part_variables = []

# Split the string into parts
for i in range(num_parts):
    start_index = i * part_length
    end_index = start_index + part_length
    part = original_string[start_index:end_index]
    part_variable_name = "part_" + str(i)
    part_variables.append(part_variable_name)
    globals()[part_variable_name] = part

# Handle the remainder
if remainder > 0:
    part = original_string[-remainder:]
    part_variable_name = "part_" + str(num_parts)
    part_variables.append(part_variable_name)
    globals()[part_variable_name] = part


# summaries = {}
# # Print the part variables
# for var_name in part_variables:
#     print(f"{var_name}: {globals()[var_name]}")
#     # print(f"Print --->{globals()[var_name]}")
#     text = globals()[var_name]
#     summary = get_answers_few_shot_approach_(text)
#     summaries[var_name] = summary
    
    
# for var_name, summary in summaries.items():
#     print(f"{var_name}: {summary}")    
    
    
with open(r'C:\Users\ashutosh.somvanshi\PVC_Trend_Analysis\Supreme_Management_Discussion_Analysis\Supreme_Text_files\tokens.txt', 'w') as file:
    # for var_name, summary in summaries.items():
    #     file.write(f"{var_name}: {summary}\n")    
    for var_name in part_variables:
        print(f"{var_name}: {globals()[var_name]}")
        # print(f"Print --->{globals()[var_name]}")
        text = globals()[var_name]
        time.sleep(30)
        summary = get_answers_few_shot_approach_(text)
        # summaries[var_name] = summary
        file.write(f"{summary}")
        # file.write(f"{var_name}: {summary}\n")
        print(f"{var_name}: {summary}")
    
    # print(part_variables)    
    
    
