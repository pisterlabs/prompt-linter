#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import nbimporter

def back_to_root():
    root = os.getcwd()
    os.chdir(root)


# In[3]:


import _1_git_repo_cloner as cloner

back_to_root()

# projectList.txt에 저장된 git repo path를 전부 clone 받음
project_list_file = "projectLIst.txt"
project_urls = cloner.get_repo_path(project_list_file)
# clone 받은 project 디렉토리명들을 project_list에 저장함
project_list = cloner.clone_project_and_update_gitignore(project_urls)


# In[ ]:


import _2_git_commit_logger as logger

back_to_root()

# commit-logs폴더에 json형식으로 커밋들의 정보를 저장
output_directory = "commit-logs"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    
# commid_logger(project_list) # 현재 수작업
logger.add_changed_file_list(project_list, output_directory)


# In[ ]:


import _3_git_file_tracker as tracker

back_to_root()

# commit-files폴더에 커밋 시점의 수정된 java파일과 그 diff파일들을 저장
output_directory = "commit-files"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    
tracker.file_tracker(project_list, output_directory)


# In[ ]:


# import openai
import _4_gpt_responser as responser

back_to_root()

# gpt-4 모델에 java/diff파일을 질의하여 응답을 {path}_response.txt에 저장
gpt_api_model = "gpt-4-0613"
openai.api_key = responser.get_api_key()

working_directory = "commit-files"
os.chdir(working_directory)

for directory in project_list:
    os.chdir(directory)
    responser.get_response_java_files(gpt_api_model)
    # responser.get_response_diff_files(gpt_api_model)
    os.chdir("..")
    
os.chdir("..")


# In[27]:


import _5_save2sheet as storer

back_to_root()

# commit-sheets 폴더에 질의파일&응답쌍들을 엑셀파일에 저장
# 향후, google sheet에 자동으로 옮길 수 있으면 좋겠음
# 잘 저장되는지 주의 필요함!
output_directory = "commit-sheets"
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

working_directory = "commit-files"
os.chdir(working_directory)

for directory in project_list:
    os.chdir(directory)
    storer.commits_to_sheet(directory, output_directory, "after")
    os.chdir("..")
    
print("모든 작업이 완료되었습니다.")
os.chdir("..")


# In[14]:


back_to_root()


# In[26]:


os.getcwd()


# In[22]:


print(project_list)


# In[25]:


os.chdir("..")


# In[24]:


project_list = ["pgjdbc"]


# In[ ]:




