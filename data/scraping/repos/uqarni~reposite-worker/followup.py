import json
import os
import time
from datetime import datetime, timedelta

import openai
from dotenv import load_dotenv

from followup_response import run_conversation
from hubspot_client import Hubspot
from llmwrapper import summarizer
from rag import find_examples
from utilz import (format_messages_to_lead_and_taylor_role,
                   format_messages_to_user_and_assistance_role, generate_response, get_prompt_variables,
                   initial_text_info, is_weekend, unix_to_strftime)

load_dotenv()

api_key = os.environ.get("OPENAI_API_KEY")
openai.api_key = api_key

reposite_hubspot_key = os.environ.get("REPOSITE_HUBSPOT_KEY")
channel_id = os.environ.get("CHANNEL_ID")
channel_account_id = os.environ.get("CHANNEL_ACCOUNT_ID")
sender_actor_id = os.environ.get("SENDER_ACTOR_ID")
sender_email = os.environ.get("SENDER_EMAIL")
subject = os.environ.get("SUBJECT")
threshold_days = os.environ.get("THRESHOLD_DAYS")
threshold_count = os.environ.get("THRESHOLD_COUNT")
#connect to hubspot
crm = Hubspot(reposite_hubspot_key)


def calculate_followup_threshold():
    # Initialize a variable to count the number of business days
    business_days_count = 0
    # Calculate three business days ago
    followup_threshold_count = datetime.now()
    while business_days_count < int(threshold_days):
        followup_threshold_count -= timedelta(days=1)
        if not is_weekend(followup_threshold_count):
            business_days_count += 1
    return followup_threshold_count


def followup(db):
    daily_followup_count = 1
    followup_threshold = calculate_followup_threshold()
    # get all the contacts who hasn't replayed from the last 3 days
    contacts = db.table('contacts').select("*").filter('org_id', 'eq', 'reposite').filter(
        'last_contact', 'lte', followup_threshold).filter('created_at', 'gte', '2023-10-23').execute()
    
    #for individual contact
    for contact in contacts.data:
        try:
            if daily_followup_count > int(threshold_count):
                break
            contact_email = contact["contact_email"]
            if contact_email is None or int(contact["followup_count"]) > 1:
                continue


            #check to see if we already responded within the last 2 minutes
            last_contact = datetime.strptime(contact["last_contact"], '%Y-%m-%d %H:%M:%S')
            if (datetime.now() - last_contact).total_seconds() < 120:
                continue
            
            #get message history
            supabase_messages = db.table("messages").select("*").filter(
                "org_id", "eq", "reposite").filter(
                "org_email", "eq", "tbowdin@reposite.io").filter(
                "contact_email","eq", contact_email).execute()
            #arrange it in order of time
            #In sorted_messages I have full message history of one contact with reposite
            sorted_messages = sorted(supabase_messages.data, key=lambda x: x['utc_datetime'])
            
            llm_sorted_messages = format_messages_to_user_and_assistance_role(sorted_messages)
            
            #check if the last message is from taylor for every contact
            if llm_sorted_messages and llm_sorted_messages[-1].get('role') == 'assistant': 
                nl_messages = format_messages_to_lead_and_taylor_role(llm_sorted_messages)
                followup_check = run_conversation(nl_messages)
                if followup_check == 'error':
                    print('Error occur while getting followup response')
                if followup_check == 'yes':
                    #pull hubspot variables for the contact and any others
                    ntmcampaignbucket, prompt_variables, bot_used, lead_first_name, reseller_org_name, supplier_name, system_prompt, trip_start_date = get_prompt_variables(db, contact["id"], contact_email)
                    if ntmcampaignbucket == 'unknown':
                        continue
                    examples = ""
                    ###PERFORM SIMILARITY SEARCH AND APPEND TO SYSYEM PROMPT BEFORE FORMATTING SYSTEM PROMPT
                    #find last message
                    # if bot_used == "taylorNMQR_RAG":
                    last_message = llm_sorted_messages[-1]['content']
                    #print('\n\nlast message:\n\n' + last_message)
                    #find examples
                    examples = find_examples(last_message, bot_used)
                    #append to system prompt
                    system_prompt += '\n' + examples

                    #format system prompt with these variables
                    formatted_system_prompt = system_prompt.format(**prompt_variables)
                    #print('\n\nformatted system prompt:\n\n' + formatted_system_prompt)

                    #find initial text
                    try:
                        initial_message = initial_text_info(ntmcampaignbucket)
                        initial_message = initial_message.format(lead_first_name = lead_first_name, reseller_org_name = reseller_org_name, supplier_name = supplier_name)
                        llm_sorted_messages.insert(0, {'role': 'assistant', 'content': initial_message})
                    except Exception as e:
                        print(contact_email + ' error adding initial message')
                        print(e)
                    
                    #format and prepend systemprompt
                    llm_sorted_messages.insert(0, {'role': 'system', 'content': formatted_system_prompt})
                    start_date = datetime.fromtimestamp(int(trip_start_date) / 1000).strftime("%Y/%m/%d, %H:%M:%S")
                    if datetime.strptime(start_date, '%Y/%m/%d, %H:%M:%S') <= followup_threshold:
                        secret_message = '''[secret internal message] Hi Taylor, this is Heather, the CEO of Reposite. The lead 
                        you've been emailing hasn't responded in 3 days. Reply to this message with a followup email to the 
                        lead. Remember, the lead doesn't see this message. Do not acknowledge it in your response. Make the followups shorter, 
                        max 3 sentences and do not refer to the quote lead opportunity; it has already passed. just mention the benefits of reposite'''
                    else:
                        secret_message = '''[secret internal message] Hi Taylor, this is Heather, the CEO of Reposite. The lead 
                        you've been emailing hasn't responded in 3 days. Reply to this message with a followup email to the 
                        lead. Remember, the lead doesn't see this message. Do not acknowledge it in your response. Make the followups shorter, 
                        max 3 sentences.'''
                    llm_sorted_messages.append({'role': 'user', 'content': secret_message})
                    print('\n---------\nContact email: ', contact_email,'\nllm_sorted_messages: ', llm_sorted_messages, '\n---------\n')
                    # #generate response

                    response = generate_response(db, llm_sorted_messages, contact_email)
                    if response:
                        usage = response['usage']
                        response = response["choices"][0]["message"]["content"]
                        if bot_used == "taylorNMQR_RAG":
                            response = response.replace("Taylor", "Taylor R Bowdin")
                            response = response.replace("Taylor R Bowdin Bowdin", "Taylor R Bowdin")
                        try:
                            recipient_actor_id = 'V-' + str(contact['id'])
                            thread_info = crm.get_contact_thread_ids(contact['id'])
                            thread_result = thread_info['results']
                            thread_id = thread_result[0]['id']      
                            print('following up with ' + contact_email + ' with message: ' + response)        
                            send = crm.send_message_to_thread(thread_id, channel_id, channel_account_id, response, recipient_actor_id, sender_email, sender_actor_id, subject, contact["contact_email"])
                            save = crm.process_hubspot_message('reposite', contact['id'], send)
                            followup_count = int(contact["followup_count"]) + 1
                            db.table("messages").insert(save).execute()
                            db.table('messages').update({'prompt_tokens': usage['prompt_tokens'], 'completion_tokens': usage['completion_tokens']}).filter('org_id', 'eq', 'reposite').filter(
                            'contact_email', 'eq', contact_email).execute()
                            db.table('contacts').update({'last_contact': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'followup_count': str(followup_count)}).filter('org_id', 'eq', 'reposite').filter(
                                'contact_email', 'eq', contact["contact_email"]).execute()
                            daily_followup_count = daily_followup_count + 1
                        except Exception as e:
                            print(e)

        except Exception as e:
            print('error with ' + contact_email)
            print(e)
            continue