import stripe
import openai
import json
import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from temporalio import activity, exceptions

stripe.api_key = os.environ.get('STRIPE_API_KEY')
openai.api_key = os.environ.get('CHATGPT_API_KEY')

@dataclass
class GetFlightDetailsInput:
    origin: str
    destination: str

@dataclass
class GetFlightsInput:
    origin: str
    destination: str
    miles: int

@dataclass
class GetPaymentInput:
    amount: str
    currency: str
    token: str

@dataclass
class GetFlightDetails:
    cost: int
    miles: int

@activity.defn
async def get_flights(input: GetFlightsInput) -> list[dict]:
    current_time = datetime.now()

    # Set correct plane model depending on how far destination is away
    if input.miles > 5000:    
        flights = [{
            'flight_number': f'Flight {i}',
            'origin': input.origin,
            'destination': input.destination,
            'miles': str(input.miles),
            'time': (current_time + timedelta(hours=i)).strftime('%H:%M'),
            'flight_model': 'A330' if i % 2 != 0 else 'B787'
        } for i in range(1, 11)]
    else:
         flights = [{
            'flight_number': f'Flight {i}',
            'origin': input.origin,
            'destination': input.destination,
            'miles': input.miles,            
            'time': (current_time + timedelta(hours=i)).strftime('%H:%M'),
            'flight_model': 'B737' if i % 2 != 0 else 'A321'
        } for i in range(1, 11)]
                
    print(f"Retrieved flights:\n{flights}")
    return flights

@activity.defn
async def get_seat_rows(model: str) -> int:
    
    rows = 0
    if model == 'A321':
        rows = 8
    elif model == 'B737':
        rows = 10
    elif model == 'A330':
        rows = 14
    elif model == 'B787':
        rows = 18
    else:
        raise exceptions.ActivityError("Flight model {model} is invalid, cannot determine seat configuration")       

    print(f"Retrieved seat rows:\n{rows}")
    return rows

@activity.defn
async def create_payment(input: GetPaymentInput):
    await asyncio.sleep(1)

    try:
        # Create a customer
        customer = stripe.Customer.create(
            source=input.token
        )

        # Charge the customer
        charge = stripe.Charge.create(
            amount=input.amount,
            currency=input.currency,
            customer=customer.id
        )

        # Return the charge object
        return charge

    except stripe.error.StripeError as e:
        # Handle any Stripe errors
        raise Exception(e.user_message)

@activity.defn
async def get_flight_details(input: GetFlightDetailsInput) -> GetFlightDetails:
    model_id = 'gpt-3.5-turbo'
    messages = [ {"role": "user", "content": f'provide cost estimate and miles for a flight between {input.origin} and {input.destination}, cost does not need to be real time. Respond JSON only using a field for cost and miles. Value of cost and miles should be a single integer, not a range, number only. No text or note or anything besides JSON as response.'} ]

    # Call the API
    completion = openai.ChatCompletion.create(
    model=model_id,
    messages=messages
    )

    # Extract and return the generated answer
    print(f'ChatGPT {completion.choices[0].message.content}')
    try:
        response = json.loads(completion.choices[0].message.content)
    except:
        pass

    flight_details = GetFlightDetails(
        cost = response["cost"],
        miles = response["miles"]
    )   


    return flight_details   