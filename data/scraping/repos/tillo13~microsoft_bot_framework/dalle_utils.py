#passed in from  from slack_events_listener.py
import openai
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from dotenv import load_dotenv
from urllib.parse import unquote, urlparse
from datetime import datetime
import os
import requests
from PIL import Image
from io import BytesIO
import time
import sys
import traceback
import json


# loading environment variables from .env file
load_dotenv('../../.env')

# setting OpenAI variables
openai.api_key = os.getenv('OPENAI_DALLE_API_KEY')
openai.api_type = "azure"
openai.api_base = os.getenv('OPENAI_DALLE_BASE_URL')
openai.api_version = os.getenv('OPENAI_DALLE_VERSION')

# initializing slack client
slack_token = os.getenv('SLACK_BOT_TOKEN')
client = WebClient(token=slack_token)

def parse_dalle_command(command_text):
    n_images = 3  # set the default value
    prompt = command_text.strip()
  
    if '--' in command_text:
        command_parts = command_text.split(' ')
        for index, part in enumerate(command_parts):
            if '--' in part:
                try:
                    n_images = min(int(part.replace('--', '')), 5)  # capping images at 5
                    command_parts.pop(index)  # remove this part from the command
                    prompt = ' '.join(command_parts).strip()  # recreate the prompt
                except ValueError:
                    pass
    return n_images, prompt

def generate_image(event, channel_id, prompt, n_images, VERBOSE_MODE):
    print(f"COMMAND RECEIVED: Ask DALL-E for {n_images} images...")
    start_time = time.time() # records the start time

    # Load the costs dictionary
    with open('openai_costs_2023sept7.json') as f:
        costs = json.load(f)

        # Get the cost of DALL·E image models 1024x1024
        DALL_E_COST_PER_IMAGE = costs["Other Models"]["Image Models"]["1024×1024"]
        estimated_cost = format(DALL_E_COST_PER_IMAGE * n_images, '.4f')

    # Check if entered number was more than limit and send Slack message
    command_parts = event["text"].split(' ')
    for index, part in enumerate(command_parts):
        if '--' in part:
            try:
                entered_number = int(part.replace('--', ''))  
                if entered_number > 5:
                    warning_message = f":exclamation: Doh! You requested {entered_number} images, but the maximum is 5. We'll proceed with 5 images."
                    print(warning_message)  # Output warning message to terminal
                    client.chat_postMessage(channel=channel_id, text=warning_message, thread_ts=event["ts"])  # Send warning to user via Slack
            except ValueError:
                pass

    # Initial message with bot animation and prompt
    initial_message_block = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":robot_face: *Connecting to DALL-E for your {n_images} images, please stand by...*\n\n*...Dall-E is creating for:* `{prompt}`..."
            }
        }
    ]
    
    client.chat_postMessage(
        channel=channel_id,
        thread_ts=event["ts"],
        text="Generating images with DALL-E...",
        blocks=initial_message_block
    )
    # Before entering the for loop
    total_orig_size = 0
    total_final_size = 0
    filename = 'N/A'

    try: 
        # Request image from DALL-E
        response = openai.Image.create(
            prompt=prompt,
            n=n_images
        )

        # Print the complete response from DALL-E
        print("RESPONSE FROM DALLE_OPENAI: ", response)

        if VERBOSE_MODE:   # if VERBOSE_MODE was passed here as argument
            client.chat_postMessage(
                channel=channel_id,
                thread_ts=event["ts"],
                text = "*VERBOSE MODE ENABLED. Posting DETAILED additional information from the call...*",
            )
            client.chat_postMessage(
                channel=channel_id,
                thread_ts=event["ts"],
                text = f"The DALLE-OPENAI Response: {response}",  # perhaps could choose to prettify
            )

        # Check if folder exists, if not, create it
        if not os.path.exists('GENERATED_IMAGES'):
            os.makedirs('GENERATED_IMAGES')

        #process each file
        for index, image_data in enumerate(response["data"]):
            # Initialize these variables at the start of the loop for each image data
            original_size_in_MB = 0 
            final_size_in_MB = 0 

            if 'error' in image_data:
                # image data contains an error
                error_details = image_data['error']
                error_message = f"Problem with image `{index+1}`...\n*{error_details['code']}*: `{error_details['message']}`\nContinuing..."
                client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=event["ts"],
                    text=error_message
                )
                continue  # skip rest of the loop for current image

            image_url = image_data["url"]
            print(f"DALL-E QUERY {index+1} COMPLETED...")
            # Take the first 15 characters of the prompt
            short_prompt = prompt[:15]
            # Replace any non-alphanumeric characters with underscores
            short_prompt = "".join(c if c.isalnum() else "_" for c in short_prompt)
            # Make it lowercase
            short_prompt = short_prompt.lower()

            filename = f"dalle_{short_prompt}_{index+1}_of_{n_images}.png"
            print(f"SHORTENED FILENAME: {filename} \nDALL-E QUERY {index+1} COMPLETED...")

            print("DOWNLOADING GENERATED IMAGE...")
            # Download image
            image_response = requests.get(image_url)
            file_data = image_response.content

            # Original size
            original_size_in_MB = len(file_data) / (1024*1024) # This line was moved up
            total_orig_size += original_size_in_MB  # This line was moved down

            # parsing SAS token for image details
            parsed = urlparse(image_url)
            sas_token = dict((k, unquote(v)) for k, v in (item.split('=') for item in parsed.query.split('&')))

            # Parse the SAS token data into a more human-readable message
            expires_at = datetime.strptime(sas_token.get('se'), '%Y-%m-%dT%H:%M:%SZ')
            now = datetime.utcnow()
            time_remain = expires_at - now
            hours, remainder = divmod(time_remain.total_seconds(), 3600)
            minutes, _ = divmod(remainder, 60)

            sas_details = f'Filename: {filename}\n'  
            sas_details += f'Full-sized Azure version accessible until: {expires_at}.\n'
            sas_details += f'Therefore, expires in about {int(hours)} hours and {int(minutes)} minutes)\n'
            sas_details += f'Azure image URL: `{image_url}`\n'
            #sas_details += f"Allowed Protocols: {sas_token.get('spr')}\n"  # https
            #sas_details += f"Resource type: {sas_token.get('sr')} (b = blob)\n"  # b means blob type
            #sas_details += f"Storage Services Version (sv): {sas_token.get('sv')}\n"
            #sas_details += f"Permissions (sp): {sas_token.get('sp')}\n"  # r means read access
            #sas_details += f"Signature (sig) for the token: [HIDDEN FOR SECURITY]\n"  # Signature should be hidden for security reasons
            #sas_details += f"Storage Service Version ID (skoid): {sas_token.get('skoid')}\n"
            #sas_details += f"Signing Key (sks): {sas_token.get('sks')}\n" 
            #sas_details += f"Key Start Time (skt): {sas_token.get('skt')}\n" 
            #sas_details += f"Tenant ID for Azure Storage Service (sktid): {sas_token.get('sktid')}\n"

            print("DOWNLOADING GENERATED IMAGE...")
            # Download image
            image_response = requests.get(image_url)
            file_data = image_response.content

            # Original size
            #original_size_in_MB = len(file_data) / (1024*1024)
            #total_orig_size += original_size_in_MB  # Add original size to the total

            # if image if over 3MB, let's reduce the size
            if len(file_data) > 3e6:  # 3e6 = 3MB
                print("IMAGE SIZE OVER 3MB, STARTING TO RESIZE...")

                img = Image.open(BytesIO(file_data))
                scale_factor = 1
                
                # original size
                original_size_in_MB = len(file_data) / (1024*1024)

                while len(file_data) > 2e6:
                    scale_factor *= 0.9
                    new_size = (int(img.size[0] * scale_factor), int(img.size[1] * scale_factor))
                    img_resized = img.resize(new_size)
                    print(f"IMAGE RESIZED TO : {new_size}")

                    byte_arr = BytesIO()
                    img_resized.save(byte_arr, format='PNG')
                    file_data = byte_arr.getvalue()
                    short_prompt = prompt[:15].lower()
                    # Replace any non-alphanumeric characters with underscores
                    short_prompt = "".join(c if c.isalnum() else "_" for c in short_prompt)

                    img_resized.save(os.path.join('GENERATED_IMAGES', f"dalle_{short_prompt}_{index+1}_of_{n_images}.png")) 
                    filepath = os.path.join('GENERATED_IMAGES', f"dalle_{short_prompt}_{index+1}_of_{n_images}.png")

                if os.path.isfile(filepath):
                    final_size_in_MB = len(file_data) / (1024*1024)  # converted from Bytes to Megabytes
                    size_reduction = original_size_in_MB - final_size_in_MB
                    total_final_size += final_size_in_MB  # Add final size to the total
                    size_reduction_percent = (size_reduction / original_size_in_MB) * 100  # the percentage of the reduction
                    
                    print(f"Original size: {format(original_size_in_MB, '.2f')} MB")
                    print(f"Final size: {format(final_size_in_MB, '.2f')} MB")
                    print(f"Size reduction: {format(size_reduction, '.2f')} MB - {format(size_reduction_percent, '.2f')}%")
                    print("UPLOADING THE RESIZED IMAGE TO SLACK...")

                    try:
                        with open(filepath, 'rb') as file:
                            files = {'file': file}
                            too_large_message = f"{filename} is over 3MB, it's being reduced in size..."
                            payload = {
                                #"initial_comment": filename,
                                "channels": channel_id,
                                "thread_ts": event["ts"],
                            }
                            headers = {
                                "Authorization": "Bearer {}".format(slack_token)
                            }
                            
                            # Here, you are uploading the image first.
                            response = requests.post(
                                "https://slack.com/api/files.upload",
                                headers=headers, files=files, data=payload
                            )
                            if not response.json()['ok']:
                                raise SlackApiError(response.json()['error'])
                            
                            image_num = index + 1  # We add 1 because `index` starts from 0
                            # Now send the image details block message after successful upload
                            trimmed_image_url = image_url.replace('https://', '')
                            block_message = [
                                {
                                    "type": "context",
                                    "elements": [
                                        {
                                            "type": "mrkdwn",
                                            "text": (f":information_source: *This is image:* _{image_num}_ *of* _{n_images}_.\n"        
                                                    f":robot_face: Your prompt was: `$dalle {prompt}`\n" 
                                                    f"*Filename:* `{filename}`\n"
                                                    f"*Full-sized Azure URL:* `{trimmed_image_url}`\n"
                                                    f"*Azure version accessible until:* `{expires_at}`\n"
                                                    f"*Azure version Expires in:* `{int(hours)} hours and {int(minutes)} minutes`\n"
                                                    f"*Original file size:* `{format(original_size_in_MB, '.2f')} MB`\n"
                                                    f"*Final file size:* `{format(final_size_in_MB, '.2f')} MB`\n" 
                                                    f"*Size reduction:* `{format(size_reduction, '.2f')} MB` - `{format(size_reduction_percent, '.2f')}%`\n"
                                                    )
                                        }
                                    ]
                                },
                                {"type": "divider"}
                            ]

                            client.chat_postMessage(
                                channel=channel_id,
                                thread_ts=event["ts"],
                                text=f"Posting image number {image_num+1} generated by DALL-E...",
                                blocks=block_message,
                            )

                            print("IMAGE AND IMAGE DETAILS SUCCESSFULLY UPLOADED TO SLACK...")
                    except SlackApiError as e:
                        print("FAILED TO UPLOAD THE IMAGE TO SLACK... SENDING THE URL INSTEAD...")
                        client.chat_postMessage(
                            channel=channel_id,
                            thread_ts=event["ts"],
                            text=f"Failed to upload image to Slack: {str(e)}. Here is the URL to your image: {image_url}",
                        )
    except openai.error.OpenAIError as o:
        if "safety system" in str(o):
            error_message = f"Out of an abundance of caution, OpenAI flagged the image `{filename}` as inappropriate. Please try a different prompt."
        else:
            error_message = f"Encountered an error while working with OpenAI API: {str(o)}. Please try again later."
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=event["ts"],
            text=error_message
        )
    except SlackApiError as e:
        error_message = f"Encountered an issue while working with Slack API: {str(e)}. Please try again later."
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=event["ts"],
            text=error_message
        )
    except Exception as e:
        error_type, error_value, error_traceback = sys.exc_info()
        tb_str = traceback.format_exception(error_type, error_value, error_traceback)
        error_message = f"An error occurred: {error_value} \n {''.join(tb_str)}"
        client.chat_postMessage(
            channel=channel_id,
            thread_ts=event["ts"],
            text="We've encountered an unexpected error. Please try again later."
        )

    # Summary block
    total_reduction = total_orig_size - total_final_size
    total_reduction_percent = 0 # set to 0 by default
    if total_orig_size > 0:
        total_reduction_percent = (total_reduction / total_orig_size) * 100
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes, seconds = divmod(elapsed_time, 60)

    # Prepare summary message
    summary_message = [
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f":information_source: Your prompt was: `$dalle {prompt}` \n"
                        f"You asked for {n_images} images.\n"
                        f"*Estimated entire cost for this transaction*: `${estimated_cost}`\n"
                        f"The total size from DALL-E of all the images was `{format(total_orig_size, '.2f')}MB`\n"
                        f"We shrunk the file cumulatively down to: `{format(total_final_size, '.2f')}MB`\n"
                        f"This is an overall reduction of `{format(total_reduction_percent, '.2f')}%`.\n"
                        f"The total time to complete this was `{int(minutes)} minutes and {int(seconds)} seconds`\n"
                        f"Try again with a new `$dalle` prompt.\n"
                        f"❓Get help at any time with `$help`."
                    )
                }
            ]
        },
        {"type": "divider"},
    ]

    # Post the summary message
    client.chat_postMessage(
        channel=channel_id,
        thread_ts=event["ts"],
        text="Summary of DALL-E image generation request...",
        blocks=summary_message,
    )
