import time
from multiprocessing import Process, Event
from flask import Flask, jsonify
from config import CRM_PHONE_NUMBER
from crm_api import CRMAPI
import re
import logging
import os
import openai
import sys

# Define a Handler which writes INFO messages or higher to the sys.stderr (this could be your console)
console = logging.StreamHandler(sys.stderr)
console.setLevel(logging.INFO)

# Define a Handler for the log file
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)

# Set a format which is simpler for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# Tell the handler to use this format
console.setFormatter(formatter)
file_handler.setFormatter(formatter)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, console]
)

app = Flask(__name__)
bot_thread = None
crm_api = CRMAPI()

# Initialize OpenAI with your API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def get_wall_height(sms_text):
    logging.info(f"Analyzing SMS text for wall height: {sms_text}")
    prompt = f"In the following SMS text, a customer is discussing the wall height of a trailer they're interested in. The height will be a number somewhere in their response, either 2, 3, or 4, possibly 2', 4', 5' and will be in their natural language or conversation, here is the text you need to analyze and extract the single digit wall height from: \"{sms_text}\". Could you tell me the wall height the customer is referring to? YOU MUST respond with a single numerical digit only, no additional text or explanation."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant and follow directions perfectly"},
            {"role": "user", "content": prompt},
        ]
    )
    wall_height_response = response['choices'][0]['message']['content'].strip()
    # Extract just the number from the AI's response
    match = re.search(r'\d+', wall_height_response)
    if match is not None:
        wall_height = match.group()
    else:
        wall_height = "No wall height found in the response"
        # Or alternatively, raise a more descriptive error
        # raise ValueError("No wall height found in the response")
    logging.info(f"Extracted wall height: {wall_height}")
    return wall_height


def extract_information(lead_data):
    logging.debug(f"Extracting information from lead data: {lead_data}")
    notes = [note['note'] for note in lead_data.get('notes', [])]
    combined_data = ' '.join(notes)
    hitch_type_pattern = r"(bumper pull|gooseneck)"
    trailer_size_pattern = r"(6x10|6x12|7x14|7x16|7x18|7x20|8x20)"
    hitch_type = re.search(hitch_type_pattern, combined_data, re.IGNORECASE)
    trailer_size = re.search(trailer_size_pattern, combined_data)
    if hitch_type:
        hitch_type = hitch_type.group(0)
    if trailer_size:
        trailer_size = trailer_size.group(0)
    return hitch_type, trailer_size

def select_template(hitch_type, trailer_size, wall_height, templates):
    logging.info(f"Selecting template for hitch type: {hitch_type}, trailer size: {trailer_size}, wall height: {wall_height}")  # Add this line
    # Format the attributes into a string similar to template names
    formatted_attributes = f"{hitch_type} {trailer_size}x{wall_height}"
    # Normalize the attributes string to compare with normalized template names
    normalized_attributes = formatted_attributes.lower().replace(' ', '')
    for template in templates:
        # Normalize the template name
        normalized_template_name = template['name'].lower().replace(' ', '')
        if normalized_attributes in normalized_template_name:
            logging.info(f"Selected template: {template}")  # Add this line
            return template
    logging.info("No matching template found")  # Add this line
    return None


def analyze_data_with_ai(data):
    # Use OpenAI's GPT-4 model to analyze the data
    logging.debug(f"Sending data to AI for analysis: {data}")
    logging.info(f"Analyzing data with AI: {data}")
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": data},
        ]
    )
    ai_response = response['choices'][0]['message']['content'].strip()
    logging.info(f"AI response: {ai_response}")
    return

def run_bot():
    logging.debug("Starting bot run...")
    logging.info("Running the bot...")

    specific_statuses = ['stat_GKAEbEJMZeyQlU7IYFOpd6PorjXupqmcNmmSzQBbcVJ',
                         'stat_6cqDnnaff2GYLV52VABicFqCV6cat7pyJn7wCJALGWz']

    # Fetch all leads with the specific statuses
    lead_ids = crm_api.get_leads_with_specific_statuses(specific_statuses)
    logging.info(f"Fetched {len(lead_ids)} leads with the specific statuses.")

    # Keep track of leads that have been processed
    processed_leads = []

    while True:
        try:
            templates = crm_api.get_sms_templates()
            logging.info(f"Fetched {len(templates)} templates.")
            sent_counter = 0
            human_intervention_counter = 0
            failed_counter = 0

            # Process all leads, not just the new incoming ones
            for lead_id in lead_ids:
                # Skip if lead has already been processed
                if lead_id in processed_leads:
                    continue

                try:
                    logging.info(f"Processing lead {lead_id}...")
                    incoming_sms = crm_api.get_latest_incoming_sms(lead_id)
                    outgoing_sms = crm_api.get_latest_outgoing_sms(lead_id)

                    # Proceed only if there's a new incoming SMS that hasn't been responded to yet
                    if incoming_sms is not None and (outgoing_sms is None or incoming_sms["date_created"] > outgoing_sms["date_created"]):
                        lead_data = crm_api.get_lead_data(lead_id)
                        if lead_data is None:
                            logging.error(f"Failed to get lead data for lead {lead_id}")
                            continue

                        # Extract the first phone number of the first contact
                        contacts = lead_data.get('contacts', [])
                        if contacts and 'phones' in contacts[0] and contacts[0]['phones']:
                            remote_phone = contacts[0]['phones'][0]['phone']
                        else:
                            logging.error(f"No phone number found for lead {lead_id}")
                            continue

                        lead_data['notes'] = crm_api.get_lead_notes(lead_id)

                        hitch_type, trailer_size = extract_information(lead_data)
                        if hitch_type is None or trailer_size is None:
                            logging.info("Insufficient lead data for hitch type or trailer size")
                            continue

                        wall_height = get_wall_height(incoming_sms['text'])
                        if wall_height:
                            template = select_template(hitch_type, trailer_size, wall_height, templates)
                            if template:
                                ai_response = analyze_data_with_ai(incoming_sms['text'])
                                logging.info(f"AI response for incoming SMS: {ai_response}")
                                if crm_api.send_message(lead_id, '', incoming_sms['id'], template['id']):
                                    sent_counter += 1
                                    logging.info(f"Successfully sent SMS template for lead {lead_id}")
                                else:
                                    crm_api.update_lead_status(lead_id, 'stat_w1TTOIbT1rYA24hSNF3c2pjazxxD0C05TQRgiVUW0A3')
                                    human_intervention_counter += 1
                                    logging.info(f"Updated status to 'Human Intervention' for lead {lead_id} due to SMS sending failure")
                            else:
                                crm_api.update_lead_status(lead_id, 'stat_w1TTOIbT1rYA24hSNF3c2pjazxxD0C05TQRgiVUW0A3')
                                human_intervention_counter += 1
                                logging.info(f"Updated status to 'Human Intervention' for lead {lead_id} due to no matching template")
                        else:
                            crm_api.update_lead_status(lead_id, 'stat_w1TTOIbT1rYA24hSNF3c2pjazxxD0C05TQRgiVUW0A3')
                            human_intervention_counter += 1
                            logging.info(f"Updated status to 'Human Intervention' for lead {lead_id} due to no valid wall height found in SMS")
                except Exception as e:
                    logging.exception(f"Failed to process lead {lead_id}")
                    failed_counter += 1

                # Add lead to the list of processed leads
                processed_leads.append(lead_id)

            logging.info(f"Sent {sent_counter} messages, marked {human_intervention_counter} leads for human intervention, failed to process {failed_counter} leads")
        except Exception as e:
            logging.exception("Failed to fetch tasks")
        time.sleep(5)

@app.route('/start', methods=['POST'])
def start_bot():
    logging.debug("Received start request")
    global bot_thread
    if bot_thread is None or not bot_thread.is_alive():
        bot_thread = Thread(target=run_bot)
        bot_thread.start()
        logging.info("Bot thread started.")
    return jsonify(success=True)

@app.route('/stop', methods=['POST'])
def stop_bot():
    logging.debug("Receieved stop request")
    global bot_thread
    if bot_thread is not None and bot_thread.is_alive():
        bot_thread = None
        logging.info("Bot thread stopped.")
    return jsonify(success=True)

@app.route('/logs', methods=['GET'])
def get_logs():
    logging.debug("Received logs request")
    with open('app.log', 'r') as f:
        return f.read()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)