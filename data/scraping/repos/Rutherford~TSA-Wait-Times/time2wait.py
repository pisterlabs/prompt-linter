from typing import Dict
from bs4 import BeautifulSoup
import http.client
import time
import json
import requests 
from datetime import datetime
import openai

# Set your OpenAI API key
openai.api_key = 'YOUR_API_KEY'

def format_tweet(wait_times):
    """
    Formats the TSA wait times into a tweet with a timestamp and emojis representing the wait time for each checkpoint.

    Args:
        wait_times (dict): A dictionary of TSA wait times, where the keys are the checkpoint names and the values are the wait times in minutes.

    Returns:
        str: A formatted tweet string that includes the timestamp and wait times for each checkpoint represented by emojis.
    """
    # Get current time
    now = datetime.now()

    # Format the timestamp
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    tweet = f"Current TSA wait times (as of {timestamp}):\n\n"

    for checkpoint, wait_time in wait_times.items():
        # Map wait times to emojis
        wait_time = int(wait_time)
        if wait_time <= 15:
            emoji = "\U0001F7E2"  # Green square
        elif wait_time <= 30:
            emoji = "\U0001F7E1"  # Yellow square
        elif wait_time <= 45:
            emoji = "\U0001F7E0"  # Orange square
        elif wait_time <= 60:
            emoji = "\U0001F7EA"  # Purple square
        else:
            emoji = "\U0001F534"  # Red square

        checkpoint_name = checkpoint.replace("CHECKPOINT", "").replace("PRECHECK ONLY", "(Pre-Check Only)").strip().title()
        tweet += f"{emoji} {checkpoint_name}: {wait_time} minutes\n"
    return tweet

def download_html(url: str) -> str:
    """
    Downloads the HTML content of a given URL and returns it as a string.
    
    Args:
        url: A string representing the URL of the website to download the HTML from.
    
    Returns:
        The HTML content of the website as a string.
    
    Raises:
        ValueError: If the response content type is not HTML.
        requests.exceptions.RequestException: If an error occurs during the request.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        if 'text/html' in response.headers.get('content-type'):
            return response.text
        else:
            raise ValueError('Response content type is not HTML')
    except requests.exceptions.RequestException as e:
        print(f'Error occurred during request: {e}')

def get_wait_times(html):
    soup = BeautifulSoup(html, 'html.parser')

    # Get the DOMESTIC container
    domestic_h1 = soup.find('h1', string=lambda text: 'domestic' in text.lower() if text else False)
    if domestic_h1 is None:
        print("Failed to find DOMESTIC heading")
        return {}
    
    domestic_div = domestic_h1.parent.parent

    # Get the checkpoint elements within the DOMESTIC container
    checkpoint_elements = domestic_div.select('.lomestic > h2')
    time_elements = domestic_div.select('.lomestic.float-right > .declasser3 > button > span')

    checkpoint_names = [elem.get_text(strip=True) for elem in checkpoint_elements]
    wait_times = [elem.get_text(strip=True) for elem in time_elements]

    return dict(zip(checkpoint_names, wait_times))


def send_tweet(tweet: str) -> None:
    """
    Sends a tweet to Twitter using the Twitter API.

    Args:
        tweet: A string containing the formatted tweet to be sent to Twitter.

    Returns:
        None

    Raises:
        Exception: If there is an error sending the tweet.

    """
    try:
        # Establish a connection with the Twitter API
        conn = http.client.HTTPSConnection("api.twitter.com")

        # Format the tweet as a JSON object
        payload = json.dumps({
            "text": tweet
        })

        # Set the headers for the HTTP request
        headers: Dict[str, str] = {
            'Content-Type': 'application/json',
            'Authorization': 'OAuth oauth_consumer_key="S1vWHVawhezr0lWI8I2WFY4m4",oauth_token="1686278180330500096-iN4Hc94OAs9s51NuOcdmufOK09jPBo",oauth_signature_method="HMAC-SHA1",oauth_timestamp="1690875988",oauth_nonce="sNeP196QUh4",oauth_version="1.0",oauth_signature="e4b7kjXPccxsrzRXrDY42NtM3KM%3D"',
            'Cookie': 'guest_id=v1%3A169087413887820853'
        }

        # Send a POST request to the Twitter API with the tweet payload and headers
        conn.request("POST", "/2/tweets", payload, headers)

        # Get the response from the Twitter API
        res = conn.getresponse()

        # Read the response data
        data = res.read()

        # Print a message indicating whether the tweet was successfully sent or not
        print(f"Tweet sent: {data.decode('utf-8')}")

    except Exception as e:
        raise Exception(f"Error sending tweet: {str(e)}")

def tweet_wait_times():
    while True:
        # Get the HTML from the website
        html = download_html("https://www.atl.com/times/")
        
        # Get the wait times
        wait_times = get_wait_times(html)
        print(f"Retrieved wait times: {wait_times}")

        # Format the tweet
        tweet = format_tweet(wait_times)
        print(f"Formatted tweet: {tweet}")

        # Send the tweet
        send_tweet(tweet)

        # Wait for 30 minutes
        time.sleep(30 * 60)

# Run the function
tweet_wait_times()
