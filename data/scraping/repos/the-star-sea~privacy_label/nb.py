from summarizer.sbert import SBertSummarizer
import os
import re

def remove_non_english(text):
    # Regular expression pattern to match non-English characters
    pattern = r'[^\x00-\x7F]+'
    
    # Remove non-English characters from the string
    clean_text = re.sub(pattern, '', text)
    
    return clean_text
#traverse all the txt files in the directory
dir_path="tool"
#create if not exist
output_path=dir_path+"/process"
if not os.path.exists(output_path):
    os.makedirs(output_path)

for file in os.listdir(dir_path):
    if file.endswith(".txt"):
        body = open(os.path.join(dir_path,file), 'r').read()
        body=remove_non_english(body)
        #remove lines with less than 10 characters and blank above and below
        body = re.sub(r'\n\s*\n', '\n', body)
        model = SBertSummarizer('paraphrase-MiniLM-L6-v2')
        result = model(body, num_sentences=30)

        output_file=open(os.path.join(output_path,file), 'w+')
        output_file.write(result)
#tranverse all the txt files in the output directory
# dir_path="/home/tongz/google100/process"

# import openai

# for file in os.listdir(dir_path):
#     if file.endswith(".txt"):
#         body = open(os.path.join(dir_path,file), 'r').read()
        
#         rsp = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#         {"role": "system", "content": "a privacy label generator for google play store"},
#         {"role": "user", "content": body}
#             ]
#         )

#         print(rsp.get("choices")[0]["message"]["content"])
        
#         break