import os
import openai
import shutil
from cppcheck_test import*
import pandas as pd
import json
import time
import re
from infer_test import*
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import argparse




print("install openai and cppcheck and create production before before")
key = input("Enter a key: ")
openai.api_key = key


def balance_c_code(input_code):
    opening_braces = 0
    closing_braces = 0
    balanced_code = ""

    for char in input_code:
        if char == '{':
            opening_braces += 1
        elif char == '}':
            closing_braces += 1

        balanced_code += char

    if opening_braces > closing_braces:
        # Add the necessary closing braces
        balanced_code += '}' * (opening_braces - closing_braces)

    return balanced_code




def generation(prompt, llm_code, vuln_list):
    prompt_2 = "Complete this code \n"+prompt
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    temperature= 0,
    messages=[
    {"role": "system", "content": "You are a coding assistant and help the user."},
    {"role": "user", "content": "When I say complete a code, I want you to complete it and give back the function from the beginning."},
    {"role": "assistant", "content": "Of course, I'd be happy to help you complete a code snippet. Please provide the initial function or code snippet, and let me know what specific task or functionality you'd like me to assist you with."},
    {"role": "user", "content": "Also, before giving the prompt, I will tell you what kind of vulnerabilities this prompt can produce."},
    {"role": "assistant", "content": "Certainly, please go ahead and describe the kind of vulnerabilities or issues you're concerned about with the code prompt you have in mind. Providing this information will help me better understand your requirements and ensure that the generated code is secure and free from common vulnerabilities."},
    {"role": "user", "content": "Here is the list ordered by their severity. "+vuln_list+"\n Complete the following code without generating the vulnerabilities: \n"+prompt}
    ]
     )

    #print(response)
    if 'choices' in response:
        x = response['choices']
        #print( x[0]['message']['content'])
        answer = x[0]['message']['content']
        code= extract_substring(answer,"#include","}")
        return code

def generationbis(prompt, llm_code, vuln_list):
    prompt_2 = "Complete this code \n"+prompt
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    temperature= 0,
    messages=[
    {"role": "system", "content": "You are a coding assistant and help the user."},
    {"role": "user", "content": "When I say complete a code, I want you to complete it and give back the function from the beginning."},
    {"role": "assistant", "content": "Sure! Please provide the code snippet that you want me to complete."},
    {"role": "user", "content": prompt_2},
    {"role": "assistant", "content": llm_code},
    {"role": "user", "content": "The code has different vulnerabilities. Here is the list ordered by their severity. "+vuln_list+"\n Complete the following code from scratch without generating the same vulnerabilities: \n"+prompt}
    ]
     )

    #print(response)
    if 'choices' in response:
        x = response['choices']
        #print( x[0]['message']['content'])
        answer = x[0]['message']['content']
        code= extract_substring(answer,"#include","}")
        return code



def generation2(prompt, gpt_msg, user_prompt):
    prompt2 = "Complete this code \n"+prompt
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0301",
    temperature= 0,
    messages=[
    {"role": "system", "content": "You are a coding assistant and help the user."},
    {"role": "user", "content": "When I say complete a code, I want you to complete it and give back the function from the beginning."},
    {"role": "assistant", "content": "Sure! Please provide the code snippet that you want me to complete."},
    {"role": "user", "content": prompt2},
    {"role": "assistant", "content": gpt_msg},
    {"role": "user", "content": "The code does not satsify my expectation, please recomplete the code:\n"+ prompt}
    ]
     )

    #print(response)
    if 'choices' in response:
        x = response['choices']
        #print( x[0]['message']['content'])
        answer = x[0]['message']['content']
        code= extract_substring(answer,"#include","}")
        return code

def extract_substring(s, start_str, end_str):
    """
    Extract a substring from s that starts with start_str and ends with end_str.
    """
    start_index = s.find(start_str) + len(start_str)
    end_index = s.rfind(end_str, start_index)
    if start_index < 0 or end_index < 0:
        return ""
    return s[start_index-8:end_index+1]

def extract_response(text):
    keyword = "### Response:"
    
    # Check if the keyword is in the text
    if keyword in text:
        # Find the index of the keyword and extract everything after it
        return text[text.index(keyword) + len(keyword):].strip()
    else:
        return None

model_name_or_path = "TheBloke/WizardCoder-15B-1.0-GPTQ"
# Or to load it locally, pass the local download path
# model_name_or_path = "/path/to/models/TheBloke_WizardCoder-15B-1.0-GPTQ"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        use_safetensors=True,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
prompt_template = '''Below is an the beginning of a code. Complete the code

### Instruction: {prompt}

### Response:'''




command_list=["wizard"]
v = ["V1"]
for version in v:
    script_dir1 = os.path.dirname(os.path.abspath(__file__))
    repository_path = os.path.join(script_dir1, version)
    # Ensure the repository directory exists
    if not os.path.exists(repository_path):
        os.makedirs(repository_path)
    script_dir = repository_path
    generation_type=0
    for model_name in command_list:
        repository_path2 = os.path.join(repository_path, model_name)
        # Ensure the repository directory exists
        if not os.path.exists(repository_path2):
            os.makedirs(repository_path2)
        script_dir = repository_path2#jusqu'a vi/starcoder
        generation_type=0
        for root, dirs, files in os.walk(script_dir1+"/scenari"):
            for file_name in files:
                print("file name is: ")
                print(file_name)
                
                if file_name != ".DS_Store"  :
                    generation_type=0
                    start = time.time()
                    file_path = os.path.join(root, file_name)
                    input = open(script_dir1+"/scenari/"+file_name, "r",encoding='utf-8')
                    line = input.readlines()
                    input.close()
                    total_lines = len(line)
                    copy_lines = int(total_lines * 1)
                    text2 = "".join(line[:copy_lines])
                    text = "Complete the code:\n"+text2

                    repository_name = file_name.replace(".c", "")
                    repository_path3 = os.path.join(script_dir, repository_name)#V1/star/CWE_repo/
                    # Ensure the repository directory exists
                    if not os.path.exists(repository_path3):
                        os.makedirs(repository_path3)

                    print("launching code")
                    if model_name == "santa":
                        create_command =  "./build/bin/starcoder -m "+ script_dir1+ "/models/bigcode/gpt_bigcode-santacoder-ggml-q4_1.bin -p " +text+" --top_k 0 --top_p 0.95 --temp 0.1 -n 300 > "+os.path.join(repository_path3, "star_" + file_name)
                    elif model_name == "star0":
                        create_command =  "./build/bin/starcoder -m "+ script_dir1+ "/models/bigcode/starcoder.ggmlv3.q4_0.bin -p " +text+" --top_k 0 --top_p 0.95 --temp 0 -n 300 > "+os.path.join(repository_path3, "star_" + file_name)
                    elif model_name == "star1":
                        create_command =  "./build/bin/starcoder -m "+ script_dir1+ "/models/bigcode/starcoder.ggmlv3.q4_1.bin -p " +text+" --top_k 0 --top_p 0.95 --temp 0 -n 300 > "+os.path.join(repository_path3, "star_" + file_name)
                    elif model_name == "starplus0":
                        create_command =  "./build/bin/starcoder -m "+ script_dir1+ "/models/bigcode/starcoderplus.ggmlv3.q4_0.bin -p " +text+" --top_k 0 --top_p 0.95 --temp 0 -n 300 > "+os.path.join(repository_path3, "star_" + file_name)
                    elif model_name == "starplus1":
                        create_command =  "./build/bin/starcoder -m "+ script_dir1+ "/models/bigcode/starcoderplus.ggmlv3.q4_1.bin -p " +text+" --top_k 0 --top_p 0.95 --temp 0 -n 300 > "+os.path.join(repository_path3, "star_" + file_name)
                    elif model_name== "wizard":
                        prompt = prompt_template.format(prompt=text)
                        outputs = pipe(prompt, max_new_tokens=300, do_sample=True, temperature=0.1, top_k=50, top_p=0.95)
                        code = outputs[0]['generated_text']
                        code = extract_response(outputs[0]['generated_text'])
                        code= extract_substring(code,"#include","}")

                    print(code)
                    
                    



                    input = open(os.path.join(repository_path3, "star2_" + file_name), "w",encoding='utf-8')
                    input.write(code)
                    input.close()
                    time.sleep(10)
                    break
                    
                    create_command = "/root/cppcheck/build/bin/cppcheck --xml-version=2 --enable=all "+os.path.join(repository_path3, "star2_" + file_name)+" 2> "+os.path.join(repository_path3, "cppcheck.xml")
                    print(create_command)
                    
                    try:
                        os.system(create_command)
                    except:
                        print(e)
                        print("cpp check failed")
                    
                    try:
                        create_command = "infer run --bufferoverrun --pulse  -- gcc -c "+os.path.join(repository_path3, "star2_" + file_name)
                    except:
                        print(e)
                        print("infer failed")
                        
                    
                    
                    try:
                        cpp_result = extract_error_info(os.path.join(repository_path3, "cppcheck.xml"))
                    except:
                        cpp_result=[]
                    none_style_list=[]
                    style_list=[]
                    if len(cpp_result) == 0:
                        cpp_error=None
                        cpp_cwe = None
                        cpp_line = None
                    else:
                        cpp_error=cpp_result[0][0]
                        cpp_cwe_non = cpp_result[0][1]
                        cpp_cwe=cpp_result[0][1]
                        cpp_line=cpp_result[0][2]
                        for i in range(0,len(cpp_result)):
                            if cpp_result[i][3] == "style":
                                if cpp_result[i][1] == "563" or cpp_result[i][1] == "561":
                                    print("unused error")
                                else:
                                    style_list.append(cpp_result[i])
                            else:
                                cpp_result[i][1] = "CWE-" +str(cpp_result[i][1])
                                cpp_result[i].append("cpp")
                                none_style_list.append(cpp_result[i])
                        
                    try:
                        infer_list = final_infer('infer-out/report.txt')
                        print("infer list:")
                        print(infer_list)
                    except Exception as e:
                        print(e)
                        print("infer error")
                        infer_list = []

                    if len(infer_list) !=0 :
                        print("Infer")
                        #print( infer_list[i][0][0])
                        #previous_msg = suggestion
                        for s in range(0,len(infer_list)):
                            if infer_list[s][0][0] =="Dead Store" or infer_list[s][0][0] =="Uninitialized Value":
                                #print(infer_list[s][0][1])
                                #print(infer_list[s][0][0])
                                #print(infer_list[s][0][2])
                                print("infer style error")
                                #style_list.append([ infer_list[s][0][1],  infer_list[s][0][0] , infer_list[s][0][2] ])
                            else:
                                print("infer list error:")
                                print( infer_list[s][0][0] )
                                infer_building=[]
                                
                                #infer_building.append([i][0][0])
                                #infer_building.append([i][0][1])
                                if infer_list[s][0][0] != None:
                                    none_style_list.insert(0, [ infer_list[s][0][0],  infer_list[s][0][1] , infer_list[s][1] , "infer"] )
                                    print("insert done")
                                    activate_cpp_check_prompt=1
                                    counter_flawfinder = 0
                                    CWE = infer_list[s][0]
                        
                    if len(none_style_list) != 0 or len(style_list)!=0:
                        generation_type=1
                        
                    print("non style list:")
                    print(none_style_list)
                    print("style list:")
                    print(style_list)
                        
                    criticavuln = "List of critical vulnerabilities detected: \n"
                    '''
                    for i in range(0,len(none_style_list)):
                        criticavuln = criticavuln + " Error type: "+str(none_style_list[i][1])+ " at line: "+str(none_style_list[i][2])+". "+"Error message: "+str(none_style_list[i][0])+"\n"
                        '''
                    
                    if len(none_style_list) ==0:
                        criticavuln = criticavuln+"None\n"
                    for i in range(0,len(none_style_list)):
                        if none_style_list[i][-1]== "cpp":
                            criticavuln = criticavuln + " Error type: "+str(none_style_list[i][1])+ " at line: "+str(none_style_list[i][2])+". "+"Error message: "+str(none_style_list[i][0])+"\n\n"
                        else:
                            example_error =""
                            for j in range(0,len(none_style_list[i][2])):
                                example_error=example_error+"\n"+none_style_list[i][2][j]
                            criticavuln = criticavuln +"Error: "+none_style_list[i][0]+ " "+none_style_list[i][1]+"\n Example: "+example_error+"\n\n"
                    non_criticavuln = "List of style vulnerabilities detected: \n"
                    if len(style_list)==0:
                        non_criticavuln = non_criticavuln+"None\n"
                    for i in range(0,len(style_list)):
                        non_criticavuln = non_criticavuln + " Error type: "+str(style_list[i][1])+ " at line: "+str(style_list[i][2])+". "+"Error message: "+str(style_list[i][0])+"\n"
                    
                    comment_error = criticavuln+"\n"+non_criticavuln
                    input = open(os.path.join(repository_path3, "report1.txt"), "w",encoding='utf-8')
                    input.write(comment_error)
                    input.close()
                    
                    
   

                        
                    if generation_type == 1:
                        end = time.time()
                        elapsed = end - start
                        elapsed = str(elapsed)
                        if version == "V1":
                            answer = generationbis(text, code, comment_error)
                        elif version =="V2":
                            answer = generation(text, code, comment_error)
                        print("on lance gpt")
                        code1 = open(os.path.join(repository_path3, "gpt_starcoder_"+elapsed +"_"+ file_name),"w")
                        code1.write(answer)
                        code1.close()
                        last_filepath = os.path.join(repository_path3, "gpt_starcoder_"+elapsed+"_" + file_name)
                    else:
                        end = time.time()
                        elapsed = end - start
                        elapsed = str(elapsed)
                        answer = generation2(text, code, comment_error)
                        code1 = open(os.path.join(repository_path3, "gpt_blanc_"+elapsed+"_" + file_name),"w")
                        code1.write(answer)
                        code1.close()
                        last_filepath = os.path.join(repository_path3, "gpt_blanc_"+elapsed+"_" + file_name)
                        
                    create_command = "/root/cppcheck/build/bin/cppcheck --xml-version=2 --enable=all "+last_filepath+" 2> "+os.path.join(repository_path3,"cppcheck2.xml" )
                    
                    try:
                        os.system(create_command)
                    except:
                        print(e)
                        print("cpp check failed")
                    
                    try:
                        create_command = "infer run --bufferoverrun --pulse  -- gcc -c "+last_filepath
                        os.system(create_command)
                    except:
                        print(e)
                        print("infer failed")
                        
                    
                    
                    try:
                        cpp_result = extract_error_info(os.path.join(repository_path3, "cppcheck2.xml"))
                    except:
                        cpp_result=[]
                    none_style_list=[]
                    style_list=[]
                    if len(cpp_result) == 0:
                        cpp_error=None
                        cpp_cwe = None
                        cpp_line = None
                    else:
                        cpp_error=cpp_result[0][0]
                        cpp_cwe_non = cpp_result[0][1]
                        cpp_cwe=cpp_result[0][1]
                        cpp_line=cpp_result[0][2]
                        for i in range(0,len(cpp_result)):
                            if cpp_result[i][3] == "style":
                                if cpp_result[i][1] == "563" or cpp_result[i][1] == "561":
                                    print("unused error")
                                else:
                                    style_list.append(cpp_result[i])
                            elif cpp_result[i][3]== "waning" :
                                cpp_result[i][1] = "CWE-" +str(cpp_result[i][1])
                                cpp_result[i].append("cpp")
                                none_style_list.append(cpp_result[i])
                        
                    try:
                        infer_list = final_infer('infer-out/report.txt')
                        print("infer list:")
                        print(infer_list)
                    except Exception as e:
                        print(e)
                        print("infer error")
                        infer_list = []

                    if len(infer_list) !=0 :
                        print("Infer")
                        #print( infer_list[i][0][0])
                        #previous_msg = suggestion
                        for s in range(0,len(infer_list)):
                            if infer_list[s][0][0] =="Dead Store" or infer_list[s][0][0] =="Uninitialized Value":
                                #print(infer_list[s][0][1])
                                #print(infer_list[s][0][0])
                                #print(infer_list[s][0][2])
                                print("infer style error")
                                #style_list.append([ infer_list[s][0][1],  infer_list[s][0][0] , infer_list[s][0][2] ])
                            else:
                                print("infer list error:")
                                print( infer_list[s][0][0] )
                                infer_building=[]
                                
                                #infer_building.append([i][0][0])
                                #infer_building.append([i][0][1])
                                if infer_list[s][0][0] != None:
                                    none_style_list.insert(0, [ infer_list[s][0][0],  infer_list[s][0][1] , infer_list[s][1] , "infer"] )
                                    print("insert done")
                                    activate_cpp_check_prompt=1
                                    CWE = infer_list[s][0]

                    criticavuln = "List of critical vulnerabilities detected: \n"
                    '''
                    for i in range(0,len(none_style_list)):
                        criticavuln = criticavuln + " Error type: "+str(none_style_list[i][1])+ " at line: "+str(none_style_list[i][2])+". "+"Error message: "+str(none_style_list[i][0])+"\n"
                        '''
                    
                    if len(none_style_list) ==0:
                        criticavuln = criticavuln+"None\n"
                    for i in range(0,len(none_style_list)):
                        if none_style_list[i][-1]== "cpp":
                            criticavuln = criticavuln + " Error type: "+str(none_style_list[i][1])+ " at line: "+str(none_style_list[i][2])+". "+"Error message: "+str(none_style_list[i][0])+"\n\n"
                        else:
                            example_error =""
                            for j in range(0,len(none_style_list[i][2])):
                                example_error=example_error+"\n"+none_style_list[i][2][j]
                            criticavuln = criticavuln +"Error: "+none_style_list[i][0]+ " "+none_style_list[i][1]+"\n Example: "+example_error+"\n\n"
                    non_criticavuln = "List of style vulnerabilities detected: \n"
                    if len(style_list)==0:
                        non_criticavuln = non_criticavuln+"None\n"
                    for i in range(0,len(style_list)):
                        non_criticavuln = non_criticavuln + " Error type: "+str(style_list[i][1])+ " at line: "+str(style_list[i][2])+". "+"Error message: "+str(style_list[i][0])+"\n"
                    
                    comment_error = criticavuln+"\n"+non_criticavuln
                    input = open(os.path.join(repository_path3, "report.txt"), "w",encoding='utf-8')
                    input.write(comment_error)
                    input.close()
