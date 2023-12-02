# https://github.com/langchain-ai/langchain/issues/9441
# use pydantic.v1
from pydantic.v1 import BaseModel, Field
from langchain.agents import tool
from requests import Response

import json
import requests
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())  # read local .env file

base_location = os.environ['BASE_LOCATION']


class GoogleRoutesInput(BaseModel):
    departure: str = Field(
        ..., description="Departure to get directions if not specified you must use 하단역")
    destination: str = Field(..., description="Destination to get directions")


@tool(args_schema=GoogleRoutesInput)
def get_routes(departure: str, destination: str) -> str:
    """Use Google Routes API to get directions from departure to destination"""

    def get_departure(address: str) -> str:
        return "부산광역시 사하구 낙동남로 지하1415 (하단동)"

    def call_api(departure: str, destination: str) -> Response:
        endpoint = 'https://routes.googleapis.com/directions/v2:computeRoutes'

        headers = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': os.environ['GOOGLE_MAPS_KEY'],
            'X-Goog-FieldMask': 'routes.legs.steps.transitDetails',
        }
        # departure_address = self.get_departure(address)

        payload = {
            "origin": {
                "address": departure
            },
            "destination": {
                "address": destination
            },
            "travelMode": "TRANSIT",
            # "travelMode": "WALK",
            "computeAlternativeRoutes": False,
            "transitPreferences": {
                "allowedTravelModes": ["SUBWAY"],
                "routingPreference": "FEWER_TRANSFERS",
            },
            "languageCode": "ko-KR",
            "units": "METRIC"
        }
        response = requests.post(endpoint, headers=headers,
                                 data=json.dumps(payload))
        return response

    def parse_route_to_string(departure: str, destination: str, recommened_route: dict) -> str:
        steps = recommened_route['steps']
        order_num = 1
        result = f"**{departure}에서 {destination}까지 이동경로**\n"
        for step in steps:
            if not step:
                continue
            # 부산 1호선
            transit_type = step['transitDetails']['transitLine']['vehicle']['type']
            transit_line = step['transitDetails']['transitLine']['nameShort']
            departure_stop = step['transitDetails']['stopDetails']['departureStop']['name']
            arrival_stop = step['transitDetails']['stopDetails']['arrivalStop']['name']
            stop_count = step['transitDetails']['stopCount']
            stop_placeholder = ''
            if transit_type == 'SUBWAY':
                departure_stop = add_station_suffix(departure_stop)
                arrival_stop = add_station_suffix(arrival_stop)
                stop_placeholder = '역'
                transit_line += '으'
            if transit_type == 'BUS':
                transit_line += '번 버스'
                departure_stop += '버스정류장'
                arrival_stop += '버스정류장'
                stop_placeholder = '정류장'
            result += f"{order_num}. {departure_stop} 에서 {arrival_stop}까지 {transit_line}로 {stop_count}개 {stop_placeholder}을 이동합니다.\n"
            order_num += 1
        result += f"{order_num}. {arrival_stop}에서 {destination}까지 도보로 이동합니다.\n"

        return result

    def add_station_suffix(station: str) -> str:
        station = station.replace('경찰서', '')

        if station[-1] == '역':
            return station
        else:
            return station + '역'

    def run(departure: str, destination: str) -> str:
        response = call_api(departure, destination)
        result_prefix = "<<HUMETRO_AI_DIRECTIONS>>\n"
        if response.status_code != 200:
            try:
                error_msg = response.json()['error']['message']
                return result_prefix + "ERROR : " + error_msg
            except:
                return result_prefix + "ERROR : " + "알 수 없는 오류가 발생했습니다."
        if 'routes' not in response.json():
            return result_prefix + "ERROR : " + "해당 경로를 찾지 못했습니다."
        recommened_route = response.json()['routes'][0]['legs'][0]

        return result_prefix + parse_route_to_string(departure, destination, recommened_route)

    return run(departure, destination)


if __name__ == "__main__":
    busan_tourist_spots = [
        "큐병원",
        "태종대",
        "감천문화마을",
        "부산타워",
        "용궁사",
        "부산아쿠아리움",
        "동백섬",
        "영도다리",
        "자갈치시장",
        "부산시립미술관",
        "송도해수욕장",
        "을숙도",
        "용두산공원",
        "부산영화의전당",
        "김해롯데워터파크",
        "이기대공원",
        "남포동",
        "흰여울문화마을",
        "해운대 해수욕장",
        "광안리",
    ]

    for loc in busan_tourist_spots:
        pass
