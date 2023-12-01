import json
import logging
import os
import platform
import sys
import time
from datetime import datetime, timedelta

import openai
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

from followup import followup
from hubspot_client import Hubspot
from llmwrapper import run_conversation, summarizer
from rag import find_examples
from supabase_client import SupabaseConnection
from utilz import (format_messages_to_lead_and_taylor_role,
                   format_messages_to_user_and_assistance_role,
                   initial_text_info, unix_to_strftime)

load_dotenv
#hubspot api key
reposite_hubspot_key = os.environ.get("REPOSITE_HUBSPOT_KEY")

#connect to hubspot
crm = Hubspot(reposite_hubspot_key)
supabase = SupabaseConnection()

def get_all_contact_from_crm():
    contact_from_db = []
    contacts = supabase.table('contacts').select("*").filter('org_id', 'eq', 'reposite').execute()

    for contact in contacts.data:
        contact_from_db.append(contact['contact_email'])

    all_contact_in_crm = crm.get_hubspot_contacts_with_property('ntmcampaignbucket', 'EQ', 'Received NMQR')
    all_contact_in_crm.extend(crm.get_hubspot_contacts_with_property('ntmcampaignbucket', 'EQ', 'Newly Onboarded'))
    all_contact_in_crm.extend(crm.get_hubspot_contacts_with_property('ntmcampaignbucket', 'EQ', 'New QR'))
    all_contact_in_crm.extend(crm.get_hubspot_contacts_with_property('ntmcampaignbucket', 'EQ', 'Token Change'))
    all_contact_in_crm.extend(crm.get_hubspot_contacts_with_property('ntmcampaignbucket', 'EQ', 'Quote Hotlist'))
    all_contact_in_crm.extend(crm.get_hubspot_contacts_with_property('ntmcampaignbucket', 'EQ', 'Booking Received'))
    #double checking we arent adding duplicate emails
    for contact in all_contact_in_crm:
        if contact['properties']['email'] in contact_from_db:
            continue

        id = contact['id']
        thread_ids = crm.get_contact_thread_ids(id)
        for thread_id in thread_ids['results']:
            thread_id = thread_id['id']
            thread_info = crm.get_thread_info(thread_id)
            thread_info = json.loads(thread_info)
            contact_email, hubspot_messages = crm.process_hubspot_thread_for_non_responded_contacts('reposite', thread_info)

            if contact_email is None or hubspot_messages is None:
                continue

            print("\n",contact_email, ': ',len(hubspot_messages),"\n")
            # if len(hubspot_messages) <= 2:  
            for message in hubspot_messages:
                try:
                    file_mode = 'a' if os.path.exists('Conversation.txt') else 'w'
                    with open('Conversation.txt', mode=file_mode) as file:
                        file.write("Conversation of message: "+contact_email+"\n" + str(message) + "\n\n")
                except:
                    True

def call_followup():
    followup(supabase)

all_emails = ['bplumb@hotelcantlie.com', 'dan@viprivieramaya.com', 'sue.behnke@acehotel.com', 'atxbeerbus@gmail.com', 'info@achauffeurs.com', 'contact@foodhoodtours.com', 'reservations@neworleansdrunkhistorytours.com', 'tastebudtours@gmail.com', 'info@emberandashphilly.com', 'citiridepc@gmail.com', 'booking@localaromas.com', 'kelly@nashvillebarrelco.com', 'cherrymomery@outlook.com', 'amy.liakaris@hilton.com', 'inquiry@madeinparkcity.com', 'greetings@greetingsfromtn.com', 'annie@banffairporter.com', 'reservations@dreamrideluxury.com', 'hello@littleglassart.co', 'kpeterson@bigonioninc.com', 'events@royalejelly.com', 'info@rennytravel.com', 'reservations@bartonspringsbikerental.com', 'austinkayaktours@gmail.com', 'info@platinumtransferpuntacana.com', 'info@lilahevents.com', 'info@iartedllc.com', 'info@brasseriebrixton.com', 'info@cosmichoneymobilebar.com', 'hello@maufrais.shop', 'doug@feastivitiesevents.com', 'danielle.aden@accor.com', 'melissae@tcrestgroup.com', 'concierge@airssist.com', 'info@seattleelitetowncar.com', 'kthorne@elgaucho.com', 'info@phoenixglasscenter.com', 'qualitytransportservice1@gmail.com', 'bee@pipcoffeeclay.com', 'mckenzie.midtown@gmail.com', 'enquiries@fastrackvip.com', 'fun@westworldpaintball.com', 'matthew@walterstation.beer', 'events@fatebrewing.com', 'sayhello@superbloom.world', 'samantha@shopthieves.com', 'info@prestige-limousine-service.com', 'casandra@riojadenver.com', 'events@mcfarlandmanagementgroup.com', 'tara@fruitionrestaurant.com', 'kenardvipinternational@gmail.com', 'puntacana@occidentalhotels.com', 'elcarmen@excellenceresorts.com', 'concierge.bavaro@princess-hotels.com', 'hola@theredtreehouse.com', 'info@siresortscapcana.com', 'info@dominicanviptransfers.com', 'concierge.dropc@dreamsresorts.com', 'info@romalimousineservice.com', 'reservationscpc-puj@royaltonresorts.com', 'jtorres@puertoricoshuttle.com', 'info@islandjourneys.com', 'josepartybuspr@gmail.com', 'res@car4pr.com', 'princesapr@zaprici.com', 'steven@cocotours.com', 'jtolbert@steinlodge.com', 'mdetling@rockhall.org', 'bruce.bell2@sympatico.ca', 'shawn.fern@pyramidglobal.com', 'groupsales@nurfc.org', 'stoked@cincybrewbus.com', 'ninjabaocatering@gmail.com', 'sales@adventures-tour.com', 'hlainesfdl@gmail.com', 'info@casapancha.com', 'southgatephilly@gmail.com', 'luckyscafemanager@gmail.com', 'forum@nice.worldmarktheclub.com', 'cs1@marinarium.com', 'spost@clevelandchildrensmuseum.org', 'rachel.shelton@hilton.com', 'dscholl@everlineresort.com', 'jillian@jmpnola.com', 'niagaratours@bell.net', 'gshaqbo@posadas.com', 'barrachina@msn.com', 'ted.vondenbenken2@hilton.com', 'breanna.worthen@waldorfastoria.com', 'bavaro.grupos6@barcelo.com', 'events@puertoricogt.com', 'thebangkoklounge@gmail.com', 'bookings@gbairvip.com', 'rick@loggerheadfitness.com', 'torontoairportlimoandtaxis@gmail.com', 'mauricio@vitepresenta.com', 'smiroglotta@clevelandart.org', 'brad@americanlegacytours.com', 'haley@studio154nashville.com', 'cierra.bickel@marriott.com']

def main():
    print('RUNNING AT ' + str(datetime.now()))

    #pull all conversations where variable = True
    contacts = crm.get_hubspot_contacts_with_property('gepeto_last_message_from_contact', 'EQ','true')

    print('\n# of contacts: ' + str(len(contacts)) + '\n')
    #pull all associated threadIds
    thread_count = 0

    
    already_contacted = []
    for contact in contacts:
        try:

            #MANUAL EXCLUSION LIST; TECH DEBT
            contact_email = contact['properties']['email']
            if contact_email in all_emails:
                continue
            
            #we got the id right here!
            id = contact['id']
            thread_ids = crm.get_contact_thread_ids(id)

            #DUPLICATE IN RUN
            if contact['properties']['email'] in already_contacted:
                continue

            already_contacted.append(contact['properties']['email'])

            #IF LAST CONTACT WITHIN LAST 2 MINUTES, DO NOT CONTACT
            try:
                last_contact = supabase.table("contacts").select("last_contact").eq("contact_email", contact_email).execute().data[0]['last_contact']
                last_contact = datetime.strptime(last_contact, "%Y-%m-%d %H:%M:%S")
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if (now - last_contact).total_seconds() < 120:
                    continue
            except: 
                contact_dict = {
                    'id': id,
                    'org_id': 'reposite',
                    'last_contact': now,
                    'owner': 'taylor',
                    'contact_email': contact_email,
                    'first_name': lead_first_name,
                    'last_name': lead_last_name,
                    'followup_count': '0',
                    'custom_data': json.dumps({'last_email': last_inbound})
                }
                add = supabase.table('contacts').insert(contact_dict).execute()
            
            for thread_id in thread_ids['results']:
                thread_id = thread_id['id']
                thread_count+=1
                thread_info = crm.get_thread_info(thread_id)
                thread_info = json.loads(thread_info)

                #need to parse message library to get this into form
                contact_email, hubspot_messages = crm.process_hubspot_thread('reposite', thread_info)

                #pull all messages from supabase for this org_id and email combo
                #print('supabase messages:') uncomment this with line 61 to pull all messages
                supabase_messages, count = supabase.table("messages").select("*").filter("org_id", "eq", "reposite").filter("org_email", "eq", "tbowdin@reposite.io").filter("contact_email","eq", contact_email).execute()
                supabase_messages = supabase_messages[1]

                #write the message ids that dont exist
                # Convert supabase_messages to a set of IDs without dashes
                supabase_ids = {message['id'].replace('-', '') for message in supabase_messages}

                for message in hubspot_messages:
                    #print(message)
                    clean_id = message['id'].replace('-', '')
                    if clean_id not in supabase_ids:
                        #print('\nAdding message with ID:', message['id'], '\n')
                        try:
                            data = supabase.table("messages").insert(message).execute()
                        except:
                            True
            
            #repull across all threadids
            supabase_messages, count = supabase.table("messages").select("*").filter("org_id", "eq", "reposite").filter("org_email", "eq", "tbowdin@reposite.io").filter("contact_email","eq", contact_email).execute()
            supabase_messages = supabase_messages[1]
            # print('supabase messages:')
            # print(supabase_messages)
            if "uzair@hellogepeto.com" in str(supabase_messages) or "mert@hellogepeto.com" in str(supabase_messages) or "heather@reposite.io" in str(supabase_messages):
                #print('\nuh oh red flag: ' + str(supabase_messages) + '\n')
                continue

            #arrange it in order of time
            sorted_messages = sorted(supabase_messages, key=lambda x: x['utc_datetime'])

            #format the conversation for the core openai call
            llm_sorted_messages = format_messages_to_user_and_assistance_role(sorted_messages)

            
            #check to see if we should respond
            #format the message for the check
            nl_messages = format_messages_to_lead_and_taylor_role(llm_sorted_messages)
                
            check = run_conversation(nl_messages)
            print('\nemail: ' + contact_email)
            print('\ncheck: ' + check)
            
            #do something with this
            if check == 'no':
                continue

            #pull hubspot variables for the contact and any others
            bot_name = os.environ.get("BOT_NAME")
            membership_link = os.environ.get("MEMBERSHIP_LINK")
            details = crm.get_contact_info(id, [
                'email', 'supplierorganizationname', 'firstname', 'lastname', 'prior_nmqrs',
                'quote_lead_company_name', 'quote_lead_category', 'lastquoterequestreceiveddate', 'quote_lead_destination', 
                'quote_lead_group_size', 'quote_lead_start_date', 'quote_lead_end_date', 'nmqrurl','ntmcampaignbucket'
            ])
            details = details['properties']
            #gotta use this variable to pull initial message and select system prompt
            ntmcampaignbucket = details.get('ntmcampaignbucket', "unknown")
            if ntmcampaignbucket == None:
                ntmcampaignbucket = 'unknown'
            email = details.get('email', "unknown")
            supplier_name = details.get('supplierorganizationname', "unknown")
            lead_first_name = details.get('firstname', "unknown")
            lead_last_name = details.get('lastname', "unknown")
            nmqr_count = details.get('prior_nmqrs', "unknown")

            #most recent nmqr info
            nmqrurl = details.get('nmqrurl', "N/A")
            reseller_org_name = details.get('quote_lead_company_name', "N/A")
            category = details.get('quote_lead_category', "N/A")
            date = details.get('lastquoterequestreceiveddate', "N/A")

            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            current_date = now
            destination = details.get('quote_lead_destination', "N/A")
            group_size = details.get('quote_lead_group_size', "N/A")
            trip_start_date = details.get('quote_lead_start_date', "N/A")
            trip_end_date = details.get('quote_lead_end_date', "N/A")
            try:
                trip_dates = unix_to_strftime(int(trip_start_date))[:-10] + ' to ' + unix_to_strftime(int(trip_end_date))[:-10]
            except:
                trip_dates = str(details.get('quote_lead_start_date', "N/A")) + ' to ' + str(details.get('quote_lead_end_date', "N/A"))
                print('failed to format trip dates for contact email ' + contact_email)
                print('trip start date: ' + str(trip_start_date))
                print('trip end date: ' + str(trip_end_date))
                print('type of trip start date: ' + str(type(trip_start_date)))
                print('type of trip end date: ' + str(type(trip_end_date)))

            #pull system prompt
            all_campaigns = initial_text_info()
            bot_used = ''
            if ntmcampaignbucket == all_campaigns[0]:
                ##we need to change this to taylorRAG##
                bot_info, count = supabase.table("bots").select("*").filter("id", "eq", "taylorNMQR_RAG").execute()
                bot_used = 'taylorNMQR_RAG' #most of the volume is coming from nmqr taylor
            else:
                bot_info, count = supabase.table("bots").select("*").filter("id", "eq", "taylor_RAG").execute()
                bot_used = 'taylor_RAG'
            bot_info = bot_info[1][0]

            system_prompt = bot_info['system_prompt']
            #print('bot used: ' + bot_used)
            #define dictionary of all system prompts
            prompt_variables = {
                'bot_name': bot_name,
                'membership_link': membership_link,
                'email': email,
                'supplier_name': supplier_name,
                'lead_first_name': lead_first_name,
                'lead_last_name': lead_last_name,
                'nmqr_count': nmqr_count,
                'nmqrurl': nmqrurl,
                'reseller_org_name': reseller_org_name,
                'category': category,
                'date': date,
                'current_date': current_date,
                'destination': destination,
                'group_size': group_size,
                'trip_dates': trip_dates,
            }
            #if bot_used == "TaylorNMQR_RAG":
            examples = ""
            ###PERFORM SIMILARITY SEARCH AND APPEND TO SYSTEM PROMPT BEFORE FORMATTING SYSTEM PROMPT
            #find last message
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
            
            #this is when we check if we already responded to the same message.
            #extract last inbound message
            for message in reversed(llm_sorted_messages):
                if message['role'] == 'user':
                    last_inbound = message['content']
                    break

            #check if we've already responded to this message. we pull the last_email from the contact. if it matches the last inbound message, we skip. if it doesnt, we update the last_email and respond
            #if we get an error when pulling the contact, we add the contact to supabase
            try:
                #if the contact already exists
                #first, ensure we aren't replying to the same message as before
                last_message_on_contact_record = json.loads(supabase.table("contacts").select("custom_data").eq("contact_email", contact_email).execute().data[0]['custom_data'])['last_email']
                if last_message_on_contact_record == last_inbound:
                    #need to update gepeto responder variable in here
                    print('\nskipped contact email: ' + contact_email)
                    continue
                else:
                    True
                    #update the contact record with the last message
                    #UNCOMMENT THIS LINE TO UPDATE CONTACTS
                    update = supabase.table('contacts').update({'custom_data': json.dumps({'last_email': last_inbound}), 'last_contact': now}).filter('contact_email', 'eq', contact_email).execute()
                    #print('\ndidnt skip')
                    #print('\nupdated contact email: ' + contact_email)

            except Exception as e:
                contact_dict = {
                    'id': id,
                    'org_id': 'reposite',
                    'last_contact': now,
                    'owner': 'taylor',
                    'contact_email': contact_email,
                    'first_name': lead_first_name,
                    'last_name': lead_last_name,
                    'followup_count': '0',
                    'custom_data': json.dumps({'last_email': last_inbound})
                }
                add = supabase.table('contacts').insert(contact_dict).execute()
                # print('\nadded contact to supabase: ' + str(contact_dict) + ' and go this response: ' + str(add)) 
                # print('\nlast message error attempt: ' + str(e))
            
            #generate response

            for i in range(5):
                try:
                    response = openai.ChatCompletion.create(model="gpt-4", messages=llm_sorted_messages, max_tokens=600)
                    usage = response['usage']
                    # print('usage: ')
                    # print(usage)
                    #POST USAGE TO ORG DB 
                    break  
                except Exception as e:
                    error_message = f"Attempt {i + 1} failed: {e}"
                    # print(error_message)
                    #handle shortening if error message is context window exceeded:
                    if "tokens. However, you" in str(e):
                        time.sleep(10)
                        try_again = summarizer(llm_sorted_messages, contact_email, 'tbowdin@reposite.io')
                        summary = {'role': 'user', 'content': 'ORIGINAL MESSAGES OMITTED. SUMMARY FOR YOU BELOW:\n\n' + try_again}

                        constructed_history = []
                        constructed_history.append(llm_sorted_messages[0])
                        print('summarizing...')
                        constructed_history.append(summary)
                        print('summarizing...')
                        constructed_history.append(llm_sorted_messages[-1])
                        print('summarized')
                        try:
                            response = openai.ChatCompletion.create(model="gpt-4", messages=constructed_history, max_tokens=600)
                            usage = response['usage']
                        except: 
                            response = None
                        break

                    if i < 4:  # we don't want to wait after the last try
                        time.sleep(5)  # wait for 5 seconds before the next attempt
            else:  # this else block executes when the loop completes normally, i.e., no break statement is encountered
                #
                # print out how many times we tried and failed
                final_error_message = f"Failed all {i + 1} attempts to call the API."
                print(final_error_message)
                #SEND FINAL ERROR MESSAGE TO UZAIR send_text("+16084205020", "+17372740771", final_error_message + "-From: " + us_num + ".\nTo: " + them_num, improovy_api_key, improovy_api_secret)
                raise
            response = response["choices"][0]["message"]["content"]
            response = response.replace("\n","<br>")
            if bot_used == "taylorNMQR_RAG":
                response = response.replace("Taylor", "Taylor R Bowdin")
                response = response.replace("Taylor R Bowdin Bowdin", "Taylor R Bowdin")

            #send the response to the user
            channel_id = os.environ.get("CHANNEL_ID")
            channel_account_id = os.environ.get("CHANNEL_ACCOUNT_ID")
            recipient_actor_id = 'V-' + id
            print('recipient_actor_id: ', recipient_actor_id)
            recipient_email = crm.get_actor_info(recipient_actor_id)['email']
            sender_actor_id = os.environ.get("SENDER_ACTOR_ID")
            sender_email = os.environ.get("SENDER_EMAIL")
            subject = os.environ.get("SUBJECT")

            #print all the variables from below
            print('thread_id: ' + thread_id)
            print('contact_id:' + id)
            print('channel_id: ' + channel_id)
            print('\n\nrecipient_email: ' + recipient_email)
            try:
                print('\n\ncampaign bucket: ' + ntmcampaignbucket)
            except:
                print('campaign bucket: unknown')
            print('\n\nbot used: ' + bot_used)
            if len(examples)>0:
                print('formatted system prompt:\n', formatted_system_prompt)
            #print('\n\nlast message: ' + last_message)
            #print('\n\nformatted system prompt:\n\n' + formatted_system_prompt)
            #print('details:' + str(details))
            # print('channel_account_id: ' + channel_account_id)
            # print('recipient_actor_id: ' + recipient_actor_id)
            
            #print('sender_email: ' + sender_email)
            # print('sender_actor_id: ' + sender_actor_id)
            # print('\n')
            # print('subject: ' + subject)\n\n
            print('\n\nresponse: ' + response)
            #print('messages:' + str(llm_sorted_messages[1:]))
            #print('usage: ' + str(usage))


            if "None" in response:
                print("AHH NONE IN RESPONSE TO: " + contact_email)
                continue 
            
            if not response:
                print('encountered a context limit error most likely while summarizing')

            
            # user_input = input("Do you want to continue? (Y/N): ").strip().upper()
            # if user_input == 'N':
                #print('would have sent message to ' + contact_email + ' with response: ' + response)
            #     continue
            # if user_input == 'Y':
            #     True

            ##send and save contact to supabase
            #manually modify response
            send = crm.send_message_to_thread(thread_id, channel_id, channel_account_id, response, recipient_actor_id, sender_email, sender_actor_id, subject, recipient_email)
            save = crm.process_hubspot_message('reposite', id, send)
            data = supabase.table("messages").insert(save).execute()

        except Exception as e:
            print('error with contact: ' + str(contact))
            print(e)
            continue

    print('FINSHED JOB AT ' + str(datetime.now()))




if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(main, 'interval', minutes = 15)
    # scheduler.add_job(get_all_contact_from_crm, 'cron', hour=12, minute=15)
    # scheduler.add_job(call_followup, 'cron', hour=16, minute=20)

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the scheduler. Will run every 15 minutes.")
    
    scheduler.start()

# call_followup()