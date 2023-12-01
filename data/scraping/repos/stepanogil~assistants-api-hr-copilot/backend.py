import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from tools.retrieve_ot import retrieve_overtime_records
from tools.knowledge_base import retrieve_relevant_kb
from tools.sap_hr import retrieve_employee_data
from tools.snow_ticket import get_ticket_info

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# must be loaded after the env variables as this requires the instance of the OpenAI client
from tools.extract_ocr import extract_ocr

client = OpenAI()

# arg to retrieve employee data from SAP; the email is mapped to the pernr in the SAP HR system; hard coded for now
user_email = "stephen.bonifacio@email.com"

def chat(user_message, thread_id):
    
    # thread id    
    if thread_id is None:
        raise ValueError("No thread ID. Please create a new thread before chatting.")
    
    # set assistant    
    assistant_id = os.getenv('HR_ASSISTANT_ID_DEV_DAY')    
        
    # create thread
    client.beta.threads.messages.create(
        thread_id=thread_id, role="user", content=user_message
    )

    # Create the run
    run = client.beta.threads.runs.create(  
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    while True:
        time.sleep(0.5)
        run = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run.id,
        )

        if run.status in ['completed', 'failed']:
            res = client.beta.threads.messages.list(thread_id=thread_id)

            # Initialize variables to store the extracted information
            extracted_text = None
            file_id = None

            # Check if content is from code interpreter (image + text)
            if len(res.data[0].content) > 1 and res.data[0].content[0].type == 'image_file' and res.data[0].content[1].type == 'text':
                # Extract image file id and LLM text response
                file_id = res.data[0].content[0].image_file.file_id
                extracted_text = res.data[0].content[1].text.value

            # If content is text only (from function call or no tool response)
            elif len(res.data[0].content) == 1 and res.data[0].content[0].type == 'text':
                # Extract LLM text response
                extracted_text = res.data[0].content[0].text.value

            # Return based on what was extracted
            if file_id and extracted_text:
                print(file_id)
                binary_img_file = client.files.content(file_id).content             
                return (binary_img_file, 
                        extracted_text)
                
            elif extracted_text:
                return extracted_text
            
        elif run.status == "requires_action":
            tool_outputs_list = []
            tools_to_call = run.required_action.submit_tool_outputs.tool_calls            
            for tool in tools_to_call:
                tool_call_id = tool.id
                function_name = tool.function.name
                #print(function_name)                            
                function_args = json.loads(tool.function.arguments)                
                #print(function_args)                 

                               
                if function_name == "RetrieveInfoFromHRKnowledgeBase":
                    output = retrieve_relevant_kb(**function_args)

                if function_name == "RetrieveTicketInfoFromServiceNow":
                    output = get_ticket_info(**function_args)
                
                if function_name == "RetrieveEmployeeInfoFromSAPHR":
                    output = retrieve_employee_data(user_email)

                if function_name == "RetrieveReceiptsDataviaOCR":
                    user_prompt = str(function_args.get('user_prompt'))                    
                    urls = function_args.get('urls')                    
                    output = extract_ocr(user_prompt, urls)                
                                
                if output:                    
                    tool_outputs_list.append({"tool_call_id": tool_call_id, "output": output})
            
            run = client.beta.threads.runs.submit_tool_outputs(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs_list
            )

            # Loop continues after submitting tool outputs to check the updated status
            continue

        # If the status is not one of the specified, continue the loop
        else:
            continue

def new_thread():
    thread = client.beta.threads.create()
    # hard coded for now    
    ot_csv = retrieve_overtime_records("Stephen Bonifacio")    
    file = client.files.create(
        file=open(ot_csv, "rb"),
        purpose="assistants"
        )
    # i want to attach the file directly to the thread but i can't. you can only attach it to a message. *shrug emoji*. already requested this feature from OpenAI
    client.beta.threads.messages.create(
            thread_id=thread.id, role="user", content="this is the time data of my direct reports. don't acknolwedge this message in your next reply. Strictly follow this instruction. This is important to my career.", file_ids=[file.id]
        )
    return thread.id