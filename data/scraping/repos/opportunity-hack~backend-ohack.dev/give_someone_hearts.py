import logging
from google.cloud import storage
from PIL import ImageFont
import openai
from PIL import ImageDraw
from PIL import Image, ImageEnhance
import urllib.request
from dotenv import load_dotenv
import os
import sys
import uuid

from datetime import datetime
import pytz

sys.path.append("../")
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
CDN_SERVER = os.getenv("CDN_SERVER")
GCLOUD_CDN_BUCKET = os.getenv("GCLOUD_CDN_BUCKET")

# add logger
logger = logging.getLogger(__name__)
# set logger to standard out
logger.addHandler(logging.StreamHandler())
# set log level
logger.setLevel(logging.INFO)
#
from common.utils.firebase import add_hearts_for_user, get_user_by_user_id, add_certificate
from common.utils.slack import send_slack


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(
        source_file_name, if_generation_match=generation_match_precondition)

    logger.info(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

def get_reason_pretty(reason):
    reasons_string = None
    
    # 4 things in "how"
    if reason == "code_reliability":
        reasons_string = "Code Reliability"
    if reason == "customer_driven_innovation_and_design_thinking":
        reasons_string = "Customer Driven Innovation and Design Thinking"    
    if reason == "iterations_of_code_pushed_to_production":
        reasons_string = "Iterations of Code Pushed to Production"
    if reason == "standups_completed":
        reasons_string = "Standups Completed"

    # 8 things in "what"
    if reason == "code_quality":
        reasons_string = "Code Quality"       
    if reason == "design_architecture":
        reasons_string = "Design Architecture"
    if reason == "documentation":
        reasons_string = "Documentation"
    if reason == "observability":
        reasons_string = "Observability"            
    if reason == "productionalized_projects":
        reasons_string = "Productionalized Projects"
    if reason == "requirements_gathering":
        reasons_string = "Requirements Gathering"            
    if reason == "unit_test_coverage":
        reasons_string = "Unit Test Coverage"
    if reason == "unit_test_writing":
        reasons_string = "Unit Test Writing"            

    if reasons_string is None:
        raise Exception(f"Reason {reason} is not valid")    
    
    return reasons_string


def generate_certificate_image(userid, name, reasons, hearts, generate_backround_image=False):
    total_hearts = hearts * len(reasons)

    # generate image for certificate/announcement
    
    header_font = ImageFont.truetype("Gidole-Regular.ttf", size=40)
    large_font = ImageFont.truetype("Gidole-Regular.ttf", size=25)
    small_font = ImageFont.truetype("Gidole-Regular.ttf", size=19)
    smaller_font = ImageFont.truetype("Gidole-Regular.ttf", size=12)
    
    # Used as bullet points for each reason in the image
    ohack_heart = Image.open('images/ohack_logo_april_25x25_2023.png')

    # Main certificate image that has transparent background
    foreground_image = Image.open('images/cert_mask_1024.png')
    
    # Draw allows us to draw text on the image
    draw = ImageDraw.Draw(foreground_image)
    
    Y_OFFSET = 150
    white_color = (255, 255, 255)  # White color
    gold_color = (255, 215, 0)  # Gold color

    # Header
    header_text = f"Congratulations {name}"
    width = draw.textlength(header_text, font=header_font)
    draw.text(((1024/2)-width/2, Y_OFFSET+190), header_text,
            font=header_font, fill=gold_color)

    # Awarded Text
    add_s = "s" if total_hearts > 1 else ""
    awarded_text = f"You have been awarded {total_hearts} heart{add_s}"
    width = draw.textlength(awarded_text, font=header_font)
    draw.text((1024/2-width/2, Y_OFFSET+240), awarded_text,
            font=header_font, fill=white_color)

    draw.text((200, Y_OFFSET+320), "For your contributions in the following areas:",
            font=large_font, fill=white_color)
    for reason in reasons:
        reason_text = get_reason_pretty(reason)
        draw.text((200, Y_OFFSET+360), reason_text,
                font=large_font, fill=white_color)
        foreground_image.paste(
            ohack_heart, (200-30, Y_OFFSET+360+1), mask=ohack_heart)
        Y_OFFSET += 25

    file_id = uuid.uuid1()
    filename = f"{file_id.hex}.png"

    footer_text = "Write code for social good @ ohack.dev"
    width = draw.textlength(footer_text, font=small_font)
    draw.text((1024/2-width/2, 1024-150), footer_text,
            font=small_font, fill=white_color)
    
    socials_text = "Follow us on Facebook, Instagram, and LinkedIn @opportunityhack"
    width = draw.textlength(socials_text, font=small_font)
    draw.text((1024/2-width/2, 1024-130), socials_text,
              font=small_font, fill=white_color)

    # Our nonprofit is based in Arizona
    az_time = datetime.now(pytz.timezone('US/Arizona'))
    iso_date = az_time.isoformat()  # Using ISO 8601 format
    bottom_text = iso_date + " " + file_id.hex
    width = draw.textlength(bottom_text, font=smaller_font)
    draw.text((1024/2-width/2, 1024-25), bottom_text,
            font=smaller_font, fill=white_color)

    # Generate a unique background image
    if generate_backround_image:
        response = openai.Image.create(
            prompt="without text a mesmerizing background with geometric shapes and fireworks no text high resolution 4k",
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']

        urllib.request.urlretrieve(image_url, "./generated_image.png")

    background_image = Image.open('generated_image.png')
    enhancer = ImageEnhance.Brightness(background_image)
    background_image_darker = enhancer.enhance(0.35)

    text_img = Image.new('RGBA', (1024, 1024), (0, 0, 0, 0))
    text_img.paste(background_image_darker, (0, 0))
    text_img.paste(foreground_image, (0, 0), mask=foreground_image)
    text_img.save(filename, format="png")
    upload_blob(GCLOUD_CDN_BUCKET, filename, filename)
    add_certificate(user_id=userid, certificate=filename)
    logger.info(f"Generated certificate for {name} with {total_hearts} hearts with filename {filename}")
    return filename

# Given a user_id, give them hearts for the reason
def give_hearts_to_user(slack_user_id, amount, reasons, create_certificate_image=False, cleanup=True, generate_backround_image=False):
    user = get_user_by_user_id(slack_user_id)
    if "id" not in user:
        error_string = f"User with slack id {slack_user_id} not found"
        logger.error(error_string)
        raise Exception(error_string)
    
    id = user["id"]

    certificate_text = ""
    if create_certificate_image:
        certificate_filename = generate_certificate_image(
            userid=id,
            name=user["name"],
            reasons=reasons,
            hearts=amount,
            generate_backround_image=generate_backround_image)
        certificate_text = f"\nCertificate: {CDN_SERVER}{certificate_filename}"
        if cleanup:
            os.remove(certificate_filename)
        
    
    if len(reasons) >= 1:
        for reason in reasons:
            add_hearts_for_user(id, amount, reason)

        reasons_string = ""
        for reason in reasons:
            reasons_string += get_reason_pretty(reason) + ", "        

        plural = "s" if amount > 1 else ""

        if amount == 0.5:
            heart_list = ":heart:"
        else:
            heart_list = ":heart: " * amount * len(reasons)

        # Intro Message to Opportunity Hack community to encourage more hearts        
        intro_message = ":heart_eyes: *Heart Announcement*! :heart_eyes:\n"
        outro_message = "\n_Thank you for taking the time out of your day to support a nonprofit with your talents_!\nMore on our heart system at https://hearts.ohack.dev"
        # Send a DM
        send_slack(channel=f"{slack_user_id}",
                  message=f"{intro_message}\nHey <@{slack_user_id}> :astronaut-hooray-woohoo-yeahfistpump: You have been given {amount} :heart: heart{plural} each for :point_right: *{reasons_string}* {heart_list}!\n{outro_message} {certificate_text}\nYour profile should now reflect these updates: https://ohack.dev/profile")
        
        # Send to public channel too
        send_slack(channel="general",
                   message=f"{intro_message}\n:astronaut-hooray-woohoo-yeahfistpump: <@{slack_user_id}> has been given {amount} :heart: heart{plural} each for :point_right: *{reasons_string}* {heart_list}!\n{outro_message} {certificate_text}")
    else:
        # Example: ["code_reliability", "iterations_of_code_pushed_to_production
        raise Exception("You must provide at least 1 reasons for giving hearts in a list")


hearts = 1
reasons = [
    ## how
    "standups_completed",
    "code_reliability",
    "customer_driven_innovation_and_design_thinking",
    "iterations_of_code_pushed_to_production",    
    
    ## what
    "productionalized_projects",
    "requirements_gathering",
    # "documentation",
    # "design_architecture",
    "code_quality",
    # "unit_test_writing",
    # "unit_test_coverage",
    # "observability"    
    ]
    

people = ["UC_______"]

for slack_id in people:
    give_hearts_to_user(
        slack_user_id=slack_id,
        amount=hearts,
        reasons=reasons,
        create_certificate_image=True,
        cleanup=True,
        generate_backround_image=True)




