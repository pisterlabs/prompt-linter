#!/usr/bin/env python3
#import request text
import custom_req as c_r
#import openai params
#import config_dir
from config_dir import config as config


#custom request by user
class Feature_Request_CustomRequest:

	def __init__(self, common_instance):
		self.common_instance = common_instance
		self.custom_sys_req_input = None
		self.custom_conv_req_input = None

	def prerequest_args_process(self):
		# send additional requests, not back to menu
		# get user custom request
		print(); print(f"\033[44;97mJob: Run Code With Custom Prompt.\033[0m")
		print(
			f"\n\033[1;31m[WARNING]\033[0m A custom request requires that code has been loaded.\033[0m")
		print()
		# update custom json request
		prog_desc_choice = "y"
		if self.common_instance.program_description is None:
			print(f"No program description available but one is required.")
		else:
			mssg = f"Current Program Description: {self.common_instance.program_description} Update? y/n: "
			mssg_option3 = f"Invalid choice. "
			prog_desc_choice = self.common_instance.user_interaction_instance.user_choice_two_options(mssg=mssg, mssg_option1=None, mssg_option2=None, mssg_option3=mssg_option3)
		if prog_desc_choice == "y":
			self.common_instance.program_description = self.common_instance.user_interaction_instance.request_input_from_user(
				"Enter Program Description: ")

		print("Enter request to change the code:")
		self.custom_sys_req_input = self.common_instance.user_interaction_instance.request_input_from_user(
			"Part 1/2 - Enter Short System Prompt: ")
		print()
		self.custom_conv_req_input = self.common_instance.user_interaction_instance.request_input_from_user(
			"Part 2/2 - Enter Request Prompt: ")
		return True

	def prepare_request_args(self):
		#build args
		summary_new_request = "Send a custom request to be applied on the code."
		sys_mssg = c_r.sys_mssg + ". " + self.custom_sys_req_input
		json_required_format = '''JSON Object Template:''' + c_r.json_required_format
		request_to_gpt = str("Your job for this request: " + self.custom_conv_req_input.replace("\n","") + ". "
		+ f"This is the description of what the program does in the the code found in the value for key 'module' of the JSON object':{self.common_instance.program_description}."
		+ f"You will make specific changes to this JSON object: {self.common_instance.gpt_response}."
		+ c_r.json_object_requirements + json_required_format + "\n"
		+ c_r.comments)

		#build request and send
		return self.common_instance.build_request_args(summary_new_request,sys_mssg,request_to_gpt)

	#send request to model
	def request_code(self, *request_args):
		#override base instance vars
		config.used_api = config.request_custom_req_api
		self.common_instance.model = config.model_request_custom_req
		self.common_instance.model_temp = config.model_request_custom_req_temperature
		#run base request implementation
		return self.common_instance.request_code_enhancement(*request_args)

	def process_successful_response(self):
		#call base
		self.common_instance.valid_response_file_management(config.module_script_fname, config.full_project_dirname, self.common_instance.gpt_response)
		return True