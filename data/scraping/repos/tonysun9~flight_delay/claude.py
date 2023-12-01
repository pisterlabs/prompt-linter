from dotenv import load_dotenv
import os
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from firewood.core.feature_retrieval import get_online_features
import redis
from typing import Optional

from datetime import datetime


def _convert_to_hours(time: str) -> str:
    minutes = int(float(time) // 60)
    hours = minutes // 60
    minutes = minutes % 60
    if hours > 0:
        return "{} hours {} minutes".format(hours, minutes)
    else:
        return "{} minutes".format(minutes)


def _convert_to_datetime(time: int) -> datetime:
    return datetime.fromtimestamp(float(time) - 3600 * 24 * 16)


def get_features(airport: str, airline: str) -> dict:
    computed_feature_vals = get_online_features(
        entity_keys={"departure_airport": airport, "airline": airline},
        feature_names=[
            # "average_delay_airport_1h",
            # "max_delay_airport_1h",
            "min_delay_airport_airline_1h",
            "average_delay_airport_airline_1h",
            "max_delay_airport_airline_1h",
        ],
    )
    hours_features = {
        "min_delay_airport_airline_1h",
        "average_delay_airport_airline_1h",
        "max_delay_airport_airline_1h",
    }
    return {
        k: _convert_to_hours(v) if k in hours_features else v
        for k, v in computed_feature_vals.items()
    }


def get_flight_info(flight_name: str) -> dict:
    airlines = {
        "UA": "United Airlines",
        "AA": "American Airlines",
        "SW": "Southwest Airlines",
        "DL": "Delta Air Lines",
    }
    airplane_numbers = {920, 360, 481, 720, 526, 755, 192, 214, 388, 823, 109}
    if len(flight_name) < 5:
        raise ValueError("Flight name must be at least 5 characters long")
    if flight_name[:2] not in airlines:
        raise ValueError(
            "Flight name must start with airline code. Allowed codes: UA, AA, SW, DL"
        )
    print("###", flight_name[2:5])
    if int(flight_name[2:5]) not in airplane_numbers:
        raise ValueError(
            "Flight name must have a valid airplane number. Valid numbers: 920, 360, 481, 720, 526, 755, 192, 214, 388, 823, 109"
        )
    airline = airlines[flight_name[:2]]

    r = redis.Redis(port=6379)
    flight_info = r.get(flight_name)
    print(flight_info.decode("utf-8").split(","))
    (
        flight_time,
        expected_takeoff_time,
        actual_takeoff_time,
        d_airport,
        a_airport,
        next_flight_name,
        next_flight_time,
    ) = flight_info.decode("utf-8").split(",")

    # print("arrival time", expected_takeoff_time + flight_time)

    # from datetime import datetime

    # dt_object = datetime.fromtimestamp(int(flight_time))

    # # convert datetime object to string
    # timestamp_str = dt_object.strftime("%Y-%m-%d %H:%M:%S")

    flight_info_dict = {
        "expected_takeoff_time": _convert_to_datetime(expected_takeoff_time),
        "actual_takeoff_time": _convert_to_datetime(actual_takeoff_time),
        "expected_arrival_time": _convert_to_datetime(
            float(expected_takeoff_time) + float(flight_time)
        ),
        "airline": airline,
        "d_airport": d_airport,
        "a_airport": a_airport,
        "flight_time": _convert_to_hours(flight_time),
        "next_flight_name": next_flight_name,
        "next_flight_time": _convert_to_datetime(next_flight_time.strip()),
    }
    print(flight_info_dict)
    return flight_info_dict


def generate_response(
    flight_name: str, connecting_flight_name: Optional[str] = None
) -> str:
    load_dotenv("anthropic.env")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    anthropic = Anthropic(api_key=api_key)

    flight_info = get_flight_info(flight_name=flight_name)
    features = get_features(
        airport=flight_info["d_airport"], airline=flight_info["airline"]
    )

    connecting_flight_info = get_flight_info(flight_name=connecting_flight_name)

    if not connecting_flight_name:
        c_prompt = ""
    else:
        c_prompt = f"""{flight_name} is expected to arrive in {flight_info["a_airport"]} at {flight_info["expected_arrival_time"]}.
        
{connecting_flight_name} is expected to fly from {connecting_flight_info["d_airport"]} to {connecting_flight_info["a_airport"]} at {connecting_flight_info["expected_takeoff_time"]}.
"""
    if not connecting_flight_name:
        c_delay = "If the flight is delayed, give practical advice on what to do."
    else:
        c_delay = f"""If the flight is delayed, give advice on what the user should do about their connecting flight. 
See if they can reschedule to {connecting_flight_info["next_flight_name"]} at {connecting_flight_info["next_flight_time"]} instead.
"""

    prompt = f"""
Human: You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

{flight_name} is expected to fly from {flight_info["d_airport"]} to {flight_info["a_airport"]} at {flight_info["expected_takeoff_time"]}.
{flight_name}'s flight time is {flight_info["flight_time"]}.
{c_prompt}

Minimum {flight_info["airline"]} flight delay at {flight_info["d_airport"]} in the past hour: {features["min_delay_airport_airline_1h"]}
Average {flight_info["airline"]} flight delay at {flight_info["d_airport"]} in the past hour: {features["average_delay_airport_airline_1h"]}
Maximum {flight_info["airline"]} flight delay at {flight_info["d_airport"]} in the past hour: {features["max_delay_airport_airline_1h"]}

Estimate whether or not the flight {flight_name} will be delayed or not. {c_delay}
Wish the user safe travels.

Assistant: Can I think step-by-step?

Human: Yes, please do.

Assistant:
    """
    # return prompt

    completion = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=500,
        temperature=0,
        prompt=prompt,
    )
    return completion.completion
