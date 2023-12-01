#!/usr/bin/env python3

#import config_dir
from config_dir import config as config
#import utils
import tools.request_utils as ut
# import openai libs/modules
#import request text
import raw_code_rq as raw_code

#request model raw code from description
class Feature_Request_Rawcode:
    def __init__(self, common_instance):
        self.common_instance = common_instance

    def prerequest_args_process(self):
        mssg = "Enter Program Description and Features: "
        self.common_instance.program_description = self.common_instance.user_interaction_instance.request_input_from_user(mssg)
        # send additional requests, not back to menu
        return True, False

    def prepare_request_args(self):
        #build args
        summary_new_request = "Request the program code for the program description provided."
        sys_mssg = raw_code.sys_mssg
        request_to_gpt = ut.concat_dict_to_string(raw_code.raw_instructions_dict) + "\n\n" + "Program Description:" + self.common_instance.program_description
        #call base
        return self.common_instance.build_request_args(summary_new_request, sys_mssg, request_to_gpt)

    #send request to model
    def request_code(self, *request_args):
        #override base instance vars
        config.used_api = config.request_rawcode_api
        self.common_instance.model = config.model_request_rawcode
        self.common_instance.model_temp = config.model_request_rawcode_temperature
        #run base request implementation
        return self.common_instance.request_code_enhancement(*request_args)

    def process_successful_response(self):
        self.common_instance.valid_response_file_management(config.module_script_fname, config.full_project_dirname, self.common_instance.gpt_response)
        return True
