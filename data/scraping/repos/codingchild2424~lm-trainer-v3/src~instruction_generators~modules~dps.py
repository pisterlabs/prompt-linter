
"""
dps is a data preprocessor modules.
"""

import json
import os
from langchain.prompts import PromptTemplate


class DpsModule:
    def __init__(
        self
        ):
        pass
    
    def preprocess_with_prompt_chain_generator(
        self, 
        data,
        prompt_chain_maker
        ):
        
        prompt_chain_result_list = []
        
        for i in data:
            # 지금은 바로 prompt에 대화만 넣을 것이니 이렇게 작업하기
            dialogues = "\n".join(
                list(i['talk']['content'].values())
            )
            
            prompt_chain_result = prompt_chain_maker(dialogues)
            
            prompt_chain_result_list.append(prompt_chain_result)
            
        return prompt_chain_result_list
    
    def preprocess_with_prompt_chain_generator_tinystories(
        self, 
        data,
        prompt_chain_maker
        ):
        
        prompt_chain_result_list = []
        
        for i in data:
            var1 = i['var1']
            var2 = i['var2']
            var3 = i['var3']

            prompt_chain_result = prompt_chain_maker(
                var1 = var1,
                var2 = var2,
                var3 = var3
                )
            
            prompt_chain_result_list.append(prompt_chain_result)
            
        return prompt_chain_result_list
    
    def preprocess(
        self, 
        data,
        ):
        
        prompt_chain_result_list = []
        
        for i in data:
            # 지금은 바로 prompt에 대화만 넣을 것이니 이렇게 작업하기
            dialogues = "\n".join(
                list(i['talk']['content'].values())
            )
            
            prompt_chain_result = prompt_chain_maker(dialogues)
            
            prompt_chain_result_list.append(prompt_chain_result)
            
        return prompt_chain_result_list
    
    def postprocess_tinystories(self, data):
        
        data_list = []
        
        for i in data.split("\n"):
            try:       
                data_list.append(
                    i
                )
            except:
                print("[70] postprocess error", i)
                continue
                
        data_list_dict = {
            "text": "\n".join(data_list)
        }
        
        return data_list_dict


    def postprocess(self, data):
        
        data_list = []
        
        for i in data.split("\n"):
            if i == "" or "user" not in i or "bot" not in i:
                continue
            else:
                try:       
                    if i.split(":")[0] == "user":
                        dict_key = "user"
                    else:
                        dict_key = "bot"
                    
                    value = i.split(":")[1]
                    
                    data_list.append(
                        "### " + dict_key + ": " + value
                    )
                except:
                    print("[70] postprocess error", i)
                    continue
                
        data_list_dict = {
            "text": "\n".join(data_list)
        }
        
        return data_list_dict