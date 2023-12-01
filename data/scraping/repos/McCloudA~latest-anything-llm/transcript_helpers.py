import json
import os
from os.path import join
import re
import requests
import json
import shutil
import time
import openai
#!pip install toktoken
import tiktoken

# TODO: Make a better summary file
# TODO: Take in a set of (phrase tag/name, phrase, exact/fuzzy) and get where/what context of each.
# TODO: create a "fill json" structure that fills a json with requested details on demand.


class DataDetailExtractor:

    def __init__(self,api_key):
        openai.api_key = api_key

    # job extractor functions
    def extract_job_skills_list(self,job_text):
        querystr = "what technical skills does the following job require? Structure the output as a python list, like ['skill1','skill2',...]. Do not write anything before or after this data structure. Job description follows: " +  job_text
        outstr, _ = self.chatgpt_simpleresponse(querystr)
        return outstr
    
    def extract_job_credentials_list(self,job_text):
        querystr = "what certifications, degrees, or other credentials does the following job require? Structure the output as a python list, like ['credential1','credential2',...]. If there are no absolutely required credentials, just return an empty list. Do not write anything before or after this data structure. Job description follows: " +  job_text
        outstr, _ = self.chatgpt_simpleresponse(querystr)
        return outstr

    def extract_job_brief_desc(self,job_text):
        querystr = "Summarize the following job description in 2 sentences. Job description follows: " +  job_text
        outstr, _ = self.chatgpt_simpleresponse(querystr)
        return outstr

    # people extractor functions
    def extract_address(self,resume_text, applicant_name = 'The applicant'):
        instr = "What is " + applicant_name + "'s address in the following resume? Structure the output as a json object, with the keys 'address','city','state', and 'zipcode'. Do not write anything before or after this data structure. Resume Follows: " + resume_text
        zipcode = None
        counter = 0
        while zipcode is None and counter < 2:
            try:
                outstr, _ = self.chatgpt_simpleresponse(instr)
                #print(outstr)
                out_response = json.loads(outstr)#['textResponse']
                #print(out_response)
                #print(type(out_response))
                address = out_response['address'] #if this breaks, we didn't get a correct json format output
                #print(address)
                city = out_response['city']
                state = out_response['state']
                zipcode = out_response['zipcode']
            except Exception as e:
                print('Error in Returned data structure, retrying query: ', e)
        return address, city, state, zipcode
    
    def extract_education(self,resume_text, applicant_name = 'The applicant'):
        instr = "What is " + applicant_name + "'s educational background or professional credentials in the following resume? Structure the output as a list of json objects, with the following structure:  [{institution: 'institution_1',degree_or_credential: 'degree_or_credential_1',fieldOfStudy: 'field_of_study',startDate: ['YYYY'],endDate: ['YYYY']},{...}]  Do not write anything before or after this data structure. Resume Follows: " + resume_text
        educationalBackground = None
        counter = 0
        while educationalBackground is None and counter < 2:
            try:
                outstr, _ = self.chatgpt_simpleresponse(instr)
                #out_response = json.loads(outstr)['textResponse']
                educationalBackground = json.loads(outstr)
            except Exception as e:
                print('Error in Returned data structure, retrying query: ', e)
        return educationalBackground
    
    def extract_workhistory(self,resume_text, applicant_name = 'The applicant'):
        instr = "What is " + applicant_name + "'s work history in the following resume? Format the output as a list of json objects, with the following structure: [{employer: 'employer_company',position: 'job_title',location: 'employment_location',startDate: ['YYYY-MM-DD'],endDate: ['YYYY-MM-DD'],description: 'job_description_overview',responsibilities: ['job_responsibility_1','job_responsibility_2','...']},{...}]  Do not write anything before or after this data structure. Resume Follows: " + resume_text
        workHistory = None
        counter = 0
        while workHistory is None and counter < 2:
            try:
                outstr, _ = self.chatgpt_simpleresponse(instr)
                workHistory = json.loads(outstr)#['textResponse']
                #workHistory = json.loads(out_response) 
                #workHistory = out_response
            except Exception as e:
                print('Error in Returned data structure, retrying query: ', e)
        return workHistory
    
    def extract_skills(self,resume_text, applicant_name = 'The applicant'):
        instr = "What technical skills does " + applicant_name + " have in the following resume? Structure the output as a python list, like ['skill1','skill2',...]. Do not write anything before or after this data structure. Resume Follows: " + resume_text
        skills = None
        counter = 0
        while skills is None and counter < 2:
            try:
                outstr, _ = self.chatgpt_simpleresponse(instr)
                #print('OUTSTR')
                #print(outstr)
                #out_response = json.loads(outstr)#['textResponse']
                # string to list:
                #out_response = outstr.strip('][').split(',')
                #print(out_response)
                skills = [str(i.replace('"','').replace("'",'')) for i in outstr.strip('][').split(', ')]
                # special workaround - occasionally we get a good structure but it's still wrong
                if any(['sorry' in i.lower() for i in skills]):
                    skills = None
            except Exception as e:
                print('Error in Returned data structure, retrying query: ', e)
        return skills

    
    def num_tokens_from_string(self,string, encoding_name):
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    
    def prep_prompts(self,prompt_text):
        prompt_text_out = prompt_text

        #prompt_token_count = tiktoken.count_tokens(prompt_text_out)["n_tokens"]
        #encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        prompt_token_count =self.num_tokens_from_string(prompt_text_out,'p50k_base')

        gpt_3_5_turbo_max_tokens = 4096
        gpt_4_0_turbo_max_tokens = 8192
        gpt_4_0_32k_max_tokens = 32768
        # Decide which model to use based on token count
        selected_model = "gpt-3.5-turbo"
        sleep_time = 0.4 # default to keep us out of trouble
        if prompt_token_count > gpt_3_5_turbo_max_tokens:
            selected_model = "gpt-4"
            sleep_time = 60.0 # 10k tokens/min limit
        if prompt_token_count > gpt_4_0_32k_max_tokens:
            selected_model = "gpt-4-32k"
            sleep_time = 60.0 # 10k tokens/min limit
        # Truncate the prompt if it exceeds the token limit for the largest model
        print( str(prompt_token_count ) , ' tokens, using model: ', selected_model)
        if prompt_token_count > gpt_4_0_32k_max_tokens:
            prompt_text_out = prompt_text_out[:tiktoken.truncate_text(prompt_text_out, gpt_4_0_32k_max_tokens)]
        return selected_model, prompt_text_out, sleep_time
    
    def chatgpt_simpleresponse(self, prompt_text, tries_threshold = 3):
        ret_text = None
        counter = 0
        model, content, sleep_time = self.prep_prompts(prompt_text)
        print('Using Model: ' , model)
        while ret_text is None and counter <= tries_threshold:
            tries_threshold = tries_threshold + 1
            time.sleep(sleep_time) # avoid hitting rate limiter
            #remember 46,000 seconds is about 12 hours, and 
            # the "rate limit" stated on the OpenAI site is 200 tokens per minute
            #https://platform.openai.com/docs/guides/rate-limits/overview
            try:
                completion = openai.ChatCompletion.create(
                    model=model,
                    messages=[
                    {"role": "user", "content":content}
                    ]
                )
                ret_text = completion['choices'][0]['message']['content'].replace('\n','')
            except Exception as e:
                print('Error in API Call, retrying API embedding query: ', e)
        return ret_text , completion

class TranscriptEmbedder:
    # either set this up using a list of files, or pointing at a name.
    # This class will set up (or reference) a title for the "space name," and also 
    # a set of input files that we will use for embedding.
    # this will also 

    # and also check for embedding/breakage, suggest what to do. stabilize this as much as we can!

    # The core algorithm should expect something like :
    # A title to use to describe this thing (like "Cost Concerns" or "competitor - snowflake" or whatever)
    # A list of strings corresponding to the thing (like ['this might be too expensive,' 'can we talk about overall cost, ...'])
    # A switch for determining if this is meant to be an exact match to some of this stuff or if we're trying to match the overall concept/fuzzy-match
    # An optional time frame to look for a second group in, to be delivered along with this one.
    # And then basically to return an object where we can get all the examples of the concept happening....
    def __init__(self):
        self.pinecone_domain_name = None
        self.input_file_set = None
        #self.set_params()
    
    def return_topic_conversations(self,topic):
        outstr = "We're looking for places where " + topic + " is mentioned. Reproduce every statement where " + topic + " is discussed. Structure the output as a list of json objects, one object for each mention, with the keys \'speaker_name\', and \' conversation_text\'. Do not write anything before or after this data structure."
        return outstr

    def return_topic_mentions(self,topic):
        outstr = "What is known about " + topic + "?. Reproduce every statement where the term '" + topic + "' is mentioned exactly. Structure the output as a list of json objects, one object for each mention, with the keys \'speaker_name\', and \' conversation_text\'. Do not write anything before or after this data structure."
        return outstr
    
    def return_address_query(self,applicant_name):
        outstr = "What is " + applicant_name + "'s address? Structure the output as a json object, with the keys 'address','city','state', and 'zipcode'. Do not write anything before or after this data structure."
        #outstr = "What is this person's full malinig address. Structure the output as a json object, with the keys 'address','city','state', and 'zipcode'. Do not write anything before or after this data structure."
        return outstr
    
    def return_educationalBackground_query(self, applicant_name):
        outstr = "What is " + applicant_name + ''''s educational background? Format the output as a list of json objects, with the following structure:  [{institution: 'institution_1',degree: 'degree_1',fieldOfStudy: 'field_of_study',startDate: ['YYYY'],endDate: ['YYYY']},{...}]  Do not write anything before or after this data structure.'''
        return outstr
    
    def return_workHistory_query(self, applicant_name):
        outstr = "What is " + applicant_name + ''''s work history? Format the output as a list of json objects, with the following structure: [{employer: 'employer_company',position: 'job_title',location: 'employment_location',startDate: ['YYYY-MM-DD'],endDate: ['YYYY-MM-DD'],description: 'job_description_overview',responsibilities: ['job_responsibility_1','job_responsibility_2','...']},{...}]  Do not write anything before or after this data structure.'''
        return outstr
    
    def return_skills_query(self,applicant_name):
        outstr = "what technical skills does " + applicant_name + " have? Structure the output as a python list, like ['skill1','skill2',...]. Do not write anything before or after this data structure."
        return outstr

    def set_params(self,anythingllm_summarydir = '/home/sean/repos/playground/sean/transcript_extraction', anythingllm_rootdir = '/home/sean/repos/anything-llm'):
        self.anythingllm_summarydir = anythingllm_summarydir # where to put the input files, and where the summary file will go.
        self.anythingllm_rootdir = anythingllm_rootdir # where is anything-llm installed
        self.anythingllm_input_dir = join(self.anythingllm_rootdir,'collector/hotdir') #where the files to be embededed go in
        self.anythingllm_check_dir = join(self.anythingllm_rootdir,'collector/hotdir/processed') # and where they come out.
        self.summary_file_name = self.pinecone_domain_name + '_sumfile.csv'
        self.summary_file_path = join(self.anythingllm_summarydir,self.summary_file_name)
    
    def set_up_by_domain_name(self, domain_name,anythingllm_summarydir, anythingllm_rootdir ): 
        self.pinecone_domain_name = domain_name
        self.set_params(anythingllm_summarydir, anythingllm_rootdir)
        return self.summary_file_path

    def set_up_by_filelist(self, list_of_input_files,domain_name,anythingllm_summarydir, anythingllm_rootdir):
        self.pinecone_domain_name = domain_name
        self.set_params(anythingllm_summarydir, anythingllm_rootdir)
        self.input_file_set = list_of_input_files

        self.migrate_input_files() # and check they made it and got encoded
        # self.create_summary_file() # and check where it went
        self.make_new_workspace() # create a new workspace for this pinecone_domain_name
        self.add_embedded_files_to_workspace()
        return self.summary_file_path
    
    def make_new_workspace(self):
        url = "http://localhost:3001/api/workspace/new"
        data = '{"name": "' + self.pinecone_domain_name + '"}'
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, data, headers=headers)
        print('Making new workspace:',self.pinecone_domain_name)

        if response.status_code == 200:
            print("POST request successful.")
        else:
            print(f"POST request failed with status code: {response.status_code}")
        content = response.text
        status_code = response.status_code
        headers = response.headers
        content_type = response.headers.get("content-type")
        print("Response Content:", content)
        print("Status Code:", status_code)
        print("Headers:", headers)
        print("Content-Type Header:", content_type)

    def delete_workspace(self,domain_name):
        print('Attempting to delete workspace ', domain_name)
        headers = {"Content-Type": "application/json"}
        response = requests.delete("http://localhost:3001/api/workspace/" + domain_name, headers=headers)
        print(response.status_code)
        print(response.text)

    def does_workspace_exist(self,domain_name):
        headers = {"Content-Type": "application/json"}
        response = requests.get("http://localhost:3001/api/workspace/" + domain_name, {}, headers=headers)
        if response.status_code == 200:
                domain_details = json.loads(response.text)['workspace']
                if domain_details:
                    print("Found domain name: ", domain_name)
                    return True
                else:
                    print('No such domain: ', domain_name)
                    return False
        else:
            print(f"POST request failed with status code: {response.status_code}")
            return False

    def add_embedded_files_to_workspace(self):
        headers = {"Content-Type": "application/json"}
        response = requests.get("http://localhost:3001/api/system/local-files", {}, headers=headers)
        keep_files = []
        for item in json.loads(response.text)['localFiles']['items'][0]['items']:
            #print('ITEM:', item)
            for fnm in self.input_file_set:
                #print('fnm: ',fnm)
                fnm_fragment = fnm.replace('/','').replace(' ','-').replace('_','-').rstrip('.').lstrip('.').lstrip('.').split('.')[0] # good enough?
                #print('fnm_fragment: ',fnm_fragment)
                if fnm_fragment.lower() in item['name'].lower():
                    keep_files.append(item['name'])
                    continue
        print('Keeping files: ')
        print(keep_files) # here are the new files relevant to this applicant

        #now go and add each file to the new workspace:
        for fname in keep_files:
            url = "http://localhost:3001/api/workspace/" + self.pinecone_domain_name + "/update-embeddings"
            data = '{"adds": ["custom-documents/' + fname + '"]}'
            #print(data)
            response = requests.post(url, data, headers=headers)

            if response.status_code == 200:
                print("POST request successful.")
            else:
                print(f"POST request failed with status code: {response.status_code}")
        content = response.text
        status_code = response.status_code
        headers = response.headers
        content_type = response.headers.get("content-type")

        #print("Response Content:", content)
        #print("Status Code:", status_code)
        #print("Headers:", headers)
        #print("Content-Type Header:", content_type)
    
    def migrate_input_files(self):
        if self.input_file_set is None:
            print('This shouldn\'t happen - somehow we tried to move the input files for embedding but we don\'t have any')
            return 1
        else:
            for fte in self.input_file_set:
                # WAIT - first let's check if there's a file like fte in self.anythingllm_check_dir already.
                # if there is, don't move this new set, we already have them.
                existing_encoded_filenames = os.listdir(self.anythingllm_check_dir)
                file_already_encoded = any([fte in i for i in existing_encoded_filenames])
                if file_already_encoded:
                    print('A version of the file was already encoded: ', fte)
                    continue

                source = os.path.abspath(fte)
                print('copying: ',source)
                destination = join(os.path.abspath(self.anythingllm_input_dir))
                print('to: ', destination)
                abc = shutil.copy(source, destination)
                #print(abc)
                #wait a bit to let the embedding happen:
                time.sleep(2.4)
                #now let's make sure this got in and got processed:
                file_moved = os.path.isfile(join(self.anythingllm_input_dir,fte)) # actually this should have embedded
                file_embedded = os.path.isfile(join(self.anythingllm_check_dir,fte)) # and moved to here
                if file_embedded and not file_moved:
                    print('File Successfully Embedded.')
                elif file_moved and not file_embedded:
                    print('File Moved, may not have Embedded - check to be sure.')
                else:
                    print('unknown file error - go check to see what happened to these files.')
            return 0

    def create_summary_file(self):
        # read through files in self.input_file_set, in self.anythingllm_summarydir, 
        # create a summary csv file by parsing them for later,
        # and put it in self.anythingllm_summarydir
        # then return the file name
        # TODO: make this better. Later on we want to use this file for better display of results. Right now it's just a concatenation:
        if self.input_file_set is None:
            print('This shouldn\'t happen - somehow we tried to move the input files for embedding but we don\'t have any')
            return 1
        else:
            with open(self.summary_file_path, 'w') as outfile:
                for fte in self.input_file_set:
                    with open(join(self.anythingllm_check_dir,fte)) as infile:
                        outfile.write(infile.read())
            return 0


    def return_fuzzy_mentions(self,topic_text):
        url = "http://localhost:3001/api/workspace/" + self.pinecone_domain_name + "/chat"
        data = '{"message": "' + self.return_topic_mentions(topic_text) + '","mode":"query"}'
        headers = {"Content-Type": "application/json"}

        

        ret_json = None
        while ret_json is None:
            time.sleep(0.4) # avoid hitting rate limiter
            try:
                response = requests.post(url, data, headers=headers)
                if response.status_code == 200:
                    print("POST request successful.")
                else:
                    print(f"POST request failed with status code: {response.status_code}")
                content = response.text
                status_code = response.status_code
                headers = response.headers
                content_type = response.headers.get("content-type")
                print("Response Content:", content)
                print("Status Code:", status_code)
                print("Headers:", headers)
                print("Content-Type Header:", content_type)
                ret_json = json.loads(content)['textResponse'] #if this breaks, we didn't get a correct json format output
            except Exception as e:
                print('Error in Returned data structure, retrying query: ')
        return ret_json
    
    def get_address_set(self,applicant_name):
        url = "http://localhost:3001/api/workspace/" + self.pinecone_domain_name + "/chat"
        data = '{"message": "' + self.return_address_query(applicant_name) + '","mode":"query"}'
        headers = {"Content-Type": "application/json"}

        address = None
        city = None
        state = None
        zipcode = None
        counter = 0
        max_iter = 3
        while address is None and counter < max_iter:
            counter = counter + 1
            time.sleep(0.4) # avoid hitting rate limiter
            try:
                response = requests.post(url, data, headers=headers)
                if response.status_code == 200:
                    print("POST request successful.")
                else:
                    print(f"POST request failed with status code: {response.status_code}")
                content = response.text
                status_code = response.status_code
                headers = response.headers
                content_type = response.headers.get("content-type")
                print("Response Content:", content)
                #print("Status Code:", status_code)
                #print("Headers:", headers)
                #print("Content-Type Header:", content_type)
                out_response = json.loads(content)['textResponse']
                address = json.loads(out_response)['address'] #if this breaks, we didn't get a correct json format output
                city = json.loads(out_response)['city']
                state = json.loads(out_response)['state']
                zipcode = json.loads(out_response)['zipcode']
                #'address','city','state', and 'zipcode'
            except Exception as e:
                print('Error in Returned data structure, retrying query: ')
        return address,city,state,zipcode
    

    def get_skills_set(self,applicant_name):
        url = "http://localhost:3001/api/workspace/" + self.pinecone_domain_name + "/chat"
        data = '{"message": "' + self.return_skills_query(applicant_name) + '","mode":"query"}'
        headers = {"Content-Type": "application/json"}

        skills = None
        counter = 0
        max_iter = 3
        while skills is None and counter < max_iter:
            counter = counter + 1
            time.sleep(0.4) # avoid hitting rate limiter
            try:
                response = requests.post(url, data, headers=headers)
                if response.status_code == 200:
                    print("POST request successful.")
                else:
                    print(f"POST request failed with status code: {response.status_code}")
                content = response.text
                status_code = response.status_code
                headers = response.headers
                content_type = response.headers.get("content-type")
                print("Response Content:", content)
                #print("Status Code:", status_code)
                #print("Headers:", headers)
                #print("Content-Type Header:", content_type)
                out_response = json.loads(content)['textResponse']
                #skills = json.loads(out_response) #if this breaks, we didn't get a correct json format output
                skills = [str(i.replace('"','').replace("'",'')) for i in out_response.strip('][').split(', ')]
                # special workaround - occasionally we get a good structure but it's still wrong
                if any(['sorry' in i.lower() for i in skills]):
                    skills = None
            except Exception as e:
                print('Error in Returned data structure, retrying query: ')
        return skills

    def get_educationalBackground_set(self,applicant_name):
        url = "http://localhost:3001/api/workspace/" + self.pinecone_domain_name + "/chat"
        data = '{"message": "' + self.return_educationalBackground_query(applicant_name) + '","mode":"query"}'
        headers = {"Content-Type": "application/json"}

        #print('DATA:')
        #print(data)

        educationalBackground = None
        counter = 0
        max_iter = 3
        while educationalBackground is None and counter < max_iter:
            counter = counter + 1
            time.sleep(0.4) # avoid hitting rate limiter
            try:
                response = requests.post(url, data, headers=headers)
                if response.status_code == 200:
                    print("POST request successful.")
                else:
                    print(f"POST request failed with status code: {response.status_code}")
                content = response.text
                status_code = response.status_code
                headers = response.headers
                content_type = response.headers.get("content-type")
                print("Response Content:", content)
                #print("Status Code:", status_code)
                #print("Headers:", headers)
                #print("Content-Type Header:", content_type)
                out_response = json.loads(content)['textResponse']
                educationalBackground = json.loads(out_response) #if this breaks, we didn't get a correct json format output
            except Exception as e:
                print('Error in Returned data structure, retrying query: ')
        return educationalBackground
    
    def get_workHistory_set(self,applicant_name):
        url = "http://localhost:3001/api/workspace/" + self.pinecone_domain_name + "/chat"
        data = '{"message": "' + self.return_workHistory_query(applicant_name) + '","mode":"query"}'
        headers = {"Content-Type": "application/json"}

        #print('DATA:')
        #print(data)

        workHistory = None
        counter = 0
        max_iter = 3
        while workHistory is None and counter < max_iter:
            counter = counter + 1
            time.sleep(0.4) # avoid hitting rate limiter
            try:
                response = requests.post(url, data, headers=headers)
                if response.status_code == 200:
                    print("POST request successful.")
                else:
                    print(f"POST request failed with status code: {response.status_code}")
                content = response.text
                status_code = response.status_code
                headers = response.headers
                content_type = response.headers.get("content-type")
                print("Response Content:", content)
                #print("Status Code:", status_code)
                #print("Headers:", headers)
                #print("Content-Type Header:", content_type)
                out_response = json.loads(content)['textResponse']
                workHistory = json.loads(out_response) #if this breaks, we didn't get a correct json format output
            except Exception as e:
                print('Error in Returned data structure, retrying query: ')
        return workHistory


    def return_simple_query(self,query_text):
        url = "http://localhost:3001/api/workspace/" + self.pinecone_domain_name + "/chat"
        print(url)
        data = '{"message": "' + query_text + '","mode":"query"}'
        print(data)
        headers = {"Content-Type": "application/json"}

        

        ret_json = None
        while ret_json is None:
            time.sleep(0.4) # avoid hitting rate limiter
            try:
                response = requests.post(url, data, headers=headers)
                if response.status_code == 200:
                    print("POST request successful.")
                else:
                    print(f"POST request failed with status code: {response.status_code}")
                content = response.text
                status_code = response.status_code
                headers = response.headers
                content_type = response.headers.get("content-type")
                print("Response Content:", content)
                print("Status Code:", status_code)
                print("Headers:", headers)
                print("Content-Type Header:", content_type)
                ret_json = content #if this breaks, we didn't get a correct json format output
            except Exception as e:
                print('Error in Returned data structure, retrying query: ')
        return ret_json


    #ret, _ = self.chatgpt_simpleresponse(prompt_text)
    #return ret
           
    # def chatgpt_simpleresponse(self, prompt_text):
    #     ret_text = None
    #     while ret_text is None:
    #         time.sleep(0.4) # avoid hitting rate limiter
    #         #remember 46,000 seconds is about 12 hours, and 
    #         # the "rate limit" stated on the OpenAI site is 200 tokens per minute
    #         #https://platform.openai.com/docs/guides/rate-limits/overview
    #         try:
    #             completion = openai.ChatCompletion.create(
    #                 model="gpt-3.5-turbo",
    #                 messages=[
    #                 {"role": "user", "content":prompt_text}
    #                 ]
    #             )
    #             ret_text = completion['choices'][0]['message']['content'].replace('\n','')
    #         except Exception as e:
    #             print('Error in API Call, retrying API embedding query: ')
    #     return ret_text , completion

