from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from twilio.rest import Client 
import cohere
import os
from dotenv import load_dotenv

env_path=os.path.join('../client/', '.env.local')
load_dotenv(env_path)

co_client = cohere.Client(os.getenv("COHERE_KEY"))

twi_account_sid = os.getenv("TWILIO_ACC") 
twi_auth_token = os.getenv("TWILIO_AUTH_TOKEN")
client = Client(twi_account_sid, twi_auth_token) 

def send_msg(msg, image, number):
    message = client.messages.create(  
        messaging_service_sid='MG4a4e60239a96a10a29b551d521abf921', 
        #media_url=[image],
        body=msg,
        to=f'+1{number}' 
    ) 
    #print("MESSAGED!")


def generate_response(name, number, total_price, wp_sf, sp_sf, ps, pw, pf, image):
    response = co_client.generate(
        model='large', 
        prompt='This is a text generator that takes sample data and tells the user the data. \n\nSample: Raghav, total price is 391.9, used water per square foot is 93, used fertilizer per square foot is 91, predicted stress is 30, predicted water is 12, predicted fertilizer is 40\nOutput Text: Hey Raghav, SimpliFarm here. Thanks for using our service. Your estimated 14 day total was $391.9, with water per square foot taking $54, and fertilizer per square foot taking $45. Our analysis shows that predicted plant stress is 30%, predicted water usage is $12, and predicted fertilizer usage is $40. \n--\nSample: Rishi, total price is 31.4, used water per square foot is 48, used fertilizer per square foot is 19, predicted stress is 19, predicted water is 32, predicted fertilizer is 14\nOutput Text: Good afternoon Rishi, thanks for using SimpliFarm! We ran the numbers and came down to the estimated 14 day total being $31.4, with water per square foot at $48, and fertilizer per square foot at $19. Our analysis indicates that plant stress is at 19%, with predicted water at $32, and predicted fertilizer is $14. \n--\nSample: Nam, total price is 123.1, used water per square foot is 93, used fertilizer per square foot is 91, predicted stress is 38, predicted water is 39, predicted fertilizer is 19\nOutput Text: Hey Nam, thanks for using SimpliFarm! Our analysis reports that your estimated two week total was $123.1, with your water per square foot coming out to $19, and your fertilizer per square foot at $48. Our analysis shows that your predicted plant stress is at 38%, predicted water usage at $39, and predicted fertilizer usage at $19. \n--\nSample: '
        +f'{name}, total price is {total_price}, used water per square foot is {wp_sf}, used fertilizer per square foot is {sp_sf}, predicted stress is {ps}, predicted water is {pw}, predicted fertilizer is {pf}\nOutput Text: ',
        max_tokens=100,
        temperature=0.3,
        k=0,
        p=0.75,
        frequency_penalty=0,
        presence_penalty=0.1,
        stop_sequences=["--"],
        return_likelihoods='NONE'
    )
    
    #Thread( target=send_msg, args=(response.generations[0].text, image, number) ).start()
    #send_msg(response.generations[0].text, image, number)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(send_msg, response.generations[0].text, image, number)