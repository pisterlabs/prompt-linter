from datetime import datetime, timedelta
import json
import re
import openai
import sys
import os
if os.path.dirname(os.path.realpath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import keys
from urllib.parse import urlencode, quote

def getDeepLink(flightDetails):
    command = f"/travelai createoffer {json.dumps(flightDetails)}"
    deeplink = bb_code_link(send_chat_deeplink(command), "Prepare offer draft.")
    return deeplink

def url_encode(params):
    return urlencode(params, quote_via=quote)


def send_chat_deeplink(msg):
    return f"intheloop:///send-chat?{url_encode({'msg': msg})}"

def bb_code_link(link, content, preview: bool = None):
    if preview is not None and isinstance(preview, bool):
        return "[url href=\"{}\" preview={}]{}[/url]".format(link, preview, content)
    else:
        return "[url href=\"{}\"]{}[/url]".format(link, content)
    
def iso_to_custom_date(iso_date):
    parsed_date = datetime.fromisoformat(iso_date)
    return parsed_date.strftime("%d%b").upper()

# Function to calculate duration in hours and minutes from ISO duration string
def iso_to_hours_minutes(iso_duration):
    duration = re.match(r'PT(\d+)H(?:(\d+)M)?', iso_duration)
    if duration:
        hours = int(duration.group(1))
        minutes = int(duration.group(2)) if duration.group(2) else 0
        return f"{hours:02d}h:{minutes:02d}min"
    else:
        return "00h:00min"
    
def generateOffer(emailText, details):
    print("---------------------")
    print(details)
    # Generating the output strings
    flights_string = generateFlightsString(details, usedForDraft=True)

    user_msg = "I will give you a flight tender enquiry email. I want you to generate an offer i can send back. Do NOT make up data. Email should be as short as possible(maximum 80 words) and formal. Do not include subject.\n\nThe following text in curly brackets is flight details which MUST stay exactly the same and should be in this exact format in the final email you write:\n"
    user_msg += "{" + flights_string + "}"

    user_msg += "\n\nEmail I want you to respond to:\n"
    user_msg += emailText
    user_msg += "\n\nRespond in same language as the email you are replying to."
    user_msg += "\n\nYour reply should ONLY be email text and NO other text."
    
    print(user_msg)

    openai.api_key = keys.openAI_APIKEY
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": user_msg}
        ]
    )

    if response.choices:
        print("Offer generated successfuly.")
        generatedOffer = response.choices[0].message.content
        return generatedOffer
    else:
        print("Unexpected or empty response received.")

def generateFlightsString(details, usedForDraft=False, email_comment_id=None):
    flights_string = ""

    for index, offer in enumerate(details["offers"]):
        if not usedForDraft:
            if index == 0:
                flights_string += f"Suggested offer:\n"
            elif index == 1:
                flights_string += f"Alternative offers:\n"
            flights_string += f"Offer {index+1}\n"
        for flight in offer["flights"]:
            departure_date = iso_to_custom_date(flight["departure"]["at"])
            duration = iso_to_hours_minutes(flight["duration"])
            flight_number = flight["carrierCode"] + " " + flight["flightNumber"]
            origin = flight["departure"]["iataCode"]
            destination = flight["arrival"]["iataCode"]
            arrival_time = datetime.fromisoformat(flight["arrival"]["at"]).strftime("%H:%M")
            departure_time = datetime.fromisoformat(flight["departure"]["at"]).strftime("%H:%M")
            
            flights_string += f"{flight_number:<8} {departure_date}  {origin}{destination:<12} {departure_time}-{arrival_time} ({duration})\n"

        flights_string += f"includedCheckedBagsOnly: {offer['luggage']['includedCheckBagsOnly']}\n"
        
        if offer["luggage"]["includedCheckedBags"] is not None:
            flights_string += f"Luggage: {offer['luggage']['includedCheckedBags']}\n"
    
        flights_string += "Total price: " + offer["price"]["grandTotal"] + " " + offer["price"]["billingCurrency"]
        if email_comment_id:
            flights_string += "\n"
            flights_string += getDeepLink(details, email_comment_id)
        flights_string += "\n\n"
        
        if usedForDraft and index == 0:
            break

    return flights_string