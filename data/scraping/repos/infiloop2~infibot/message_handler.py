import time
import os
from system_messages import get_intro_message, under_quota_message, too_long_message,get_privacy_message
from rate_limits import is_within_limits, reset_limits, use_one_limit
from short_term_memory import get_short_term_memory, write_short_term_memory, append_history
from openai_api import get_openai_response
from dynamo_api import get_quota, get_last_intro_message_timestamp, put_last_intro_message_timestamp, get_last_privacy_accepted_timestamp, get_is_private_mode_on, get_is_unsafe_mode_on, get_last_unsafe_accepted_timestamp
from commands import handle_command
from whatsapp_sender import send_whatsapp_text_reply
from system_commands import is_system_command, handle_system_command, get_unsafe_mode_on_message

# Update this whenever you change privacy/unsafe message so that you prompt the user to accept it again
last_privacy_updated_timestamp = 0
last_unsafe_updated_timestamp = 1682950000

# Handle text messages to phone number ID, from, timestamp with message body
def handle_text_message(phone_number_id, from_, timestamp, message, user_secret):
    current_time = int(time.time())
    if current_time - timestamp > 60:
        # Too old messages which may come through because of whatsapp server issues or retries due to errors
        return

    # admin system messages  
    if from_ == os.environ.get("admin_phone_number"):
        if message.startswith("Quota"):
            spl = message.split(" ")
            if len(spl) == 4:
                if spl[3] != os.environ.get("admin_password"):
                    send_whatsapp_text_reply(phone_number_id, from_, "Invalid admin password", is_private_on=False, is_unsafe_on=False)
                    return
                reset_limits(spl[1], spl[2])
                send_whatsapp_text_reply(phone_number_id, from_, "Quota reset for " + spl[1] + " to " + str(spl[2]), is_private_on=False, is_unsafe_on=False)
                return
    
    # Check if within limits
    if not is_within_limits(from_):
        send_whatsapp_text_reply(phone_number_id, from_, under_quota_message(from_), is_private_on=False, is_unsafe_on=False)
        return

    if len(message) > 500:
        send_whatsapp_text_reply(phone_number_id, from_, too_long_message(), is_private_on=False, is_unsafe_on=False)
        return
    
    # Global modes
    is_private_on = get_is_private_mode_on(from_, user_secret)
    is_unsafe_on = get_is_unsafe_mode_on(from_, user_secret)

    # Verify user has accepted privacy policy
    last_privacy_ts = get_last_privacy_accepted_timestamp(from_, user_secret)
    if last_privacy_ts < last_privacy_updated_timestamp:
        send_whatsapp_text_reply(phone_number_id, from_, "Please read and accept privacy policy before continuing", is_private_on, is_unsafe_on)
        send_whatsapp_text_reply(phone_number_id, from_, get_privacy_message(), is_private_on, is_unsafe_on)
        return
    
    # Verify user has accepted unsafe policy if in unsafe mode
    if is_unsafe_on:
        last_unsafe_ts = get_last_unsafe_accepted_timestamp(from_, user_secret)
        if last_unsafe_ts < last_unsafe_updated_timestamp:
            send_whatsapp_text_reply(phone_number_id, from_, "Please read and accept conditions for unsafe mode before proceeding", is_private_on, is_unsafe_on)
            send_whatsapp_text_reply(phone_number_id, from_, get_unsafe_mode_on_message(), is_private_on, is_unsafe_on)
            return

    history = get_short_term_memory(from_, user_secret)
    if len(history) == 0:
        # Send welcome message if not sent within last 7 days already
        last_ts = get_last_intro_message_timestamp(from_, user_secret)
        if current_time - last_ts > 7 * 24 * 3600:
            send_whatsapp_text_reply(phone_number_id, from_, get_intro_message(get_quota(from_)), is_private_on, is_unsafe_on)
            put_last_intro_message_timestamp(from_, current_time, user_secret)

    # Handle system messages from users
    if is_system_command(message):
        further_ai_reponse, updated_user_message = handle_system_command(message, phone_number_id, from_, user_secret, is_private_on, is_unsafe_on)
        if not further_ai_reponse:
            return
        if updated_user_message is not None:
            message = updated_user_message

    ##### Main AI Response #####
    # TODO: Fork if unsafe mode is on
    use_one_limit(from_)
    ai_response, command = get_openai_response(message, history)
    
    #Send assistant reply
    send_whatsapp_text_reply(phone_number_id, from_, ai_response, is_private_on, is_unsafe_on)
    # Append to history
    history = append_history(history, "user", message)
    history = append_history(history, "assistant", ai_response)
    write_short_term_memory(from_, history, user_secret, is_private_on)
   
    if command is not None:
        handle_command(command, phone_number_id, from_, history, user_secret, is_private_on, is_unsafe_on)