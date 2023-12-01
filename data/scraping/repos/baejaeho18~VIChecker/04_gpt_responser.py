#!/usr/bin/env python
# coding: utf-8

# In[113]:


import os
import json
import time
import openai
import subprocess

def get_api_key():
    # Read API key from file
    with open('api_key.txt', 'r', encoding='utf-8') as file:
        api_key = file.read().strip()
    # never upload api_key public!
    with open(".gitignore", "a") as gitignore:
        gitignore.write("api_key.txt")
    return api_key

def get_response(prompt, gpt_api_model):
    # Make a question using the API
    response = openai.ChatCompletion.create(
        model=gpt_api_model,
        messages=[
            {"role": "user", "content": prompt}
        ],
    )
    # generated answer
    answer = response['choices'][0]['message']['content'].strip()
    return answer

def record_response(file_path, answer):
    # Record the answer
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(answer)   
    except:
        print(f"Answer Write 오류")
        
def ask_to_gpt(file_path, prompt, gpt_api_model):
    response_file_path = file_path.replace(".java", "_response.txt")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(file_path)
        try:
            answer = get_response(prompt+content, gpt_api_model)
            record_response(response_file_path, answer)
        except Exception as e:
            if isinstance(e, openai.error.RateLimitError):
                print(f"Rate Limit Error: {str(e)}")
                # 대기 후 다시 시도
                time.sleep(30)  # 이 시간을 조정해 주세요
                ask_to_gpt(file_path, prompt, gpt_api_model)
            elif "8192" in str(e):  # 최대가용토큰(파일사이즈) 초과
                print(f"File Size Exceeds: {str(e)}")
                with open("blackList.txt", 'a', encoding='utf-8') as f:
                    f.write(file_path+"\n")
            else:
                print(f"Response Error: {str(e)}")

def get_response_java_files(gpt_api_model):
    blackListFile = "blackList.txt"
    if os.path.exists(blackListFile):
        with open(blackListFile, 'r', encoding='utf-8') as b:
            blackList = b.read()
    else:
        blackList = ""
    
    for root, dirs, files in os.walk("."):
        for file in files:
            if "_after_" in file and file.endswith(".java"):
                file_path = os.path.join(root, file)
                response_file_path = file_path.replace(".java", "_response.txt")
                
                # 가용토큰을 넘는 파일들을 무시
                if file_path in blackList:
                    continue
                # test인 파일들 무시
                if "_test_" in file_path:
                    continue
                # 응답을 가진 파일들을 무시
                if not os.path.exists(response_file_path):
                    file_size = os.path.getsize(file_path)
                    # 30Kb - gpt4.0 / 80kb - gpt3.5
                    if file_size <= 30 * 1024:
                        prompt = "Can you check the following code and if there is any CWE or CVE related vulnerability, can you point it out the number of CWE or CVE and describe it?\n"
                        ask_to_gpt(file_path, prompt, gpt_api_model)
                    else:
                        print(f"Ignored {file_path} - File size exceeds 30KB")
    
    
def get_response_diff_files(gpt_api_model):
    blackListFile = "blackList.txt"
    if os.path.exists(blackListFile):
        with open(blackListFile, 'r', encoding='utf-8') as b:
            blackList = b.read()
    else:
        blackList = ""
    
    for root, dirs, files in os.walk("."):
        for file in files:
            if "_diff_" in file and file.endswith(".java"):
                file_path = os.path.join(root, file)
                response_file_path = file_path.replace(".java", "_response.txt")
                
                # 가용토큰을 넘는 파일들을 무시
                if file_path in blackList:
                    continue
                # test인 파일들 무시
                if "_test_" in file_path:
                    continue
                # 응답을 가진 파일들을 무시
                if not os.path.exists(response_file_path):
                    file_size = os.path.getsize(file_path)
                    # 30Kb - gpt4.0 / 80kb - gpt3.5
                    if file_size <= 30 * 1024:
                        prompt = "Could you read the following diff file and, if there are any security vulnerabilities in changes, point out the related CWE or CVE numbers along with the reason they occurred?\n"
                        ask_to_gpt(file_path, prompt, gpt_api_model)
                    else:
                        print(f"Ignored {file_path} - File size exceeds 30KB")


# In[114]:


if __name__ == "__main__":
    directories = ["guava"]  # ["h2database", "bc-java", "pgjdbc", "junit4", "gson", "guava"]
    working_directory = "commit-files"
    gpt_api_model = "gpt-4" # gpt-3.5-turbo-16k
    # commit_logger(directories)
    
    openai.api_key = get_api_key()
    
    # 3_에서 이미 만들어졌으리라 가정.
    os.chdir(working_directory)
    
    for directory in directories:
        os.chdir(directory)
        # 필요한 작업 선택
        # get_response_java_files(gpt_api_model)
        get_response_diff_files(gpt_api_model)
        os.chdir("..")
    os.chdir("..")


# In[115]:


os.getcwd()


# In[111]:


os.chdir("..")


# In[32]:


os.chdir("VIChecker")


# In[ ]:




