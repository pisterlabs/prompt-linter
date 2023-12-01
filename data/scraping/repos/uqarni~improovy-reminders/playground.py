import os
import redis
import platform
import openai 
from main import improovy_reminder
import datetime
import json
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse

# Connect to Redis server
redis_host = os.environ.get("REDIS_1_HOST")
redis_port = 25061
redis_password = os.environ.get("REDIS_1_PASSWORD")

operating_system = platform.system()

# Set the CA certificates file path based on the operating system
if operating_system == 'Linux':
    ssl_ca_certs = "/etc/ssl/certs/ca-certificates.crt"
elif operating_system == 'Darwin':  # macOS
    ssl_ca_certs = "/etc/ssl/cert.pem"  # Update to macOS path
else:
    raise ValueError(f"Unsupported operating system: {operating_system}")

rd = redis.Redis(host=redis_host, port=redis_port, password=redis_password, ssl=True, ssl_ca_certs=ssl_ca_certs)



def sms_webhook(sms_data: Dict):
    #get datetime
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d %H:%M:%S")

    #only add if inbound
    print(str(sms_data))
    
    #hardcoded!!!!
    #uncomment this block
    refnums = ["14065214908", "16504603511","19726669892","18327896969","14067294654", "15673991116"]
    first_test = sms_data['data']['direction'] == 1 or str(sms_data["data"]["content"]).startswith("Hey I")
    if not (first_test and str(sms_data['data']['justcall_number']) in refnums):
        return JSONResponse(content={"status": "success"}, status_code=200)

    #get datetime
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d %H:%M:%S")

    #process data
    sms_data = sms_data['data']
    us_num = str(sms_data['justcall_number']) # DOESN'T contain "+" here
    them_num = sms_data['contact_number']
    content = sms_data['content']

    is_unsubscribing = content.lower() == 'stop' or content.lower() == 'unsubscribe'
    if is_unsubscribing:
        #ensure no followups - REDIS
        rd.hset("last_message-" + us_num, them_num, 'X-' + now)

        #add to customer conversation log - Redis
        key = us_num + '-' + them_num
        to_append = {"role": "user", "content": content}
        rd.rpush(key, json.dumps(to_append))
    else:
        rd.rpush("log_1", json.dumps(sms_data) + "-" + now)# add to log
        print('put on queue)')
        #mq.enqueue('worker.process_sms', sms_data)# add to q

    #############################
    # Update the "contacts" table
    try:
        supabase.table("contacts").insert({
            "created_at": now,
            "contact_phone":them_num,
            "org_id":"awc",
            "contact_email":None,
            "first_name":None,
            "last_name":None,
            "deal_stage":"",
            "custom_data":None, 
            "last_contact":now, 
            "timezone":"UTC",
            "followup_count": "X" if is_unsubscribing else "0",
            "owner":"+"+us_num # append "+"
        }).execute()
    except Exception as e:
        print("/sms_webhook: Couldn't insert {} to contacts table for following reason: {}. Trying to update that row...".format(them_num, e))
        try:
            supabase.table("contacts").update({
                "last_contact":now,
                "followup_count": "X" if is_unsubscribing else "0",
            }).eq("contact_phone", them_num).execute()
        except Exception as e:
            print("/sms_webhook: Couldn't update contacts table for phone number {} for following reason: {}. Passing...".format(them_num, e))

    # Return a success response
    return JSONResponse(content={"status": "success"}, status_code=200)

