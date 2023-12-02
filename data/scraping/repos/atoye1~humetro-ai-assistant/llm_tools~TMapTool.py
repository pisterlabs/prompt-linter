import requests
from langchain.tools import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

import os

raise DeprecationWarning(
    'This tool is deprecated, use GoogleRouteTool instead')


class TMapToolInputChecker(BaseModel):
    """Input for TMapTool Check"""
    destination: str = Field(...,
                             description="destination specified by user who is asking directions")


class TMapTool(BaseTool):
    name = "tmap tool for directions"
    description = "you should use this tool when asked about directions"
    hadan_coords = ("128.96673333333334", "35.106230555555555")

    def get_coords(self, destination: str) -> tuple:
        """
            Get coordinates of a given destination using Kakao API
            https://developers.kakao.com/docs/latest/ko/local/dev-guide#search-by-keyword
        """

        kakao_endpoint = "https://dapi.kakao.com/v2/local/search/keyword.json"
        kakao_query = destination

        params = {
            "query": kakao_query,
            "x":  self.hadan_coords[0],
            "y": self.hadan_coords[1],
            # need logics to dynamically change radius
            'radius': 20000,
        }
        headers = {
            "Authorization": "KakaoAK " + os.environ['KAKAO_REST_KEY'],
        }
        res = requests.get(kakao_endpoint, headers=headers, params=params)
        addresses = res.json()['documents']
        if len(addresses) == 0:
            raise Exception("No address found")
        print(addresses[0]['place_name'], addresses[0]['x'], addresses[0]['y'])

        return (addresses[0]['x'], addresses[0]['y'])

    def call_TMapAPI(self, destination: str) -> dict:
        TMap_endpoint = "https://apis.openapi.sk.com/transit/routes"
        dest_coords = self.get_coords(destination)

        headers = {
            "accept": "application/json",
            "appKey": os.environ['TMAP_API_KEY'],
            "Content-Type": "application/json"
        }
        data = {
            "startX": self.hadan_coords[0],
            "startY": self.hadan_coords[1],
            "endX": dest_coords[0],
            "endY": dest_coords[1],
            "count": 10,
            "lang": 0,
            "format": "json"
        }

        res = requests.post(TMap_endpoint, headers=headers, json=data)
        if res.status_code != 200:
            raise Exception("TMap API request failed")
        return res.json()

    def parse_directions_dict(self, directions_dict: dict) -> str:
        fare = directions_dict['fare']['regular']['totalFare']
        legs = directions_dict['legs']

        # enrich data with proper suffix
        for idx, leg in enumerate(legs):
            if leg['mode'] == 'SUBWAY':
                leg['start']['name'] = leg['start']['name'] + '역'
                leg['end']['name'] = leg['end']['name'] + '역'
                if idx - 1 >= 0:
                    legs[idx-1]['end']['name'] = legs[idx - 1]['end']['name'] + '역'
                if idx + 1 < len(legs):
                    legs[idx+1]['start']['name'] = legs[idx +
                                                        1]['start']['name'] + '역'
            elif leg['mode'] == 'BUS':
                leg['start']['name'] = leg['start']['name'] + '정류소'
                leg['end']['name'] = leg['end']['name'] + '정류소'
                if idx - 1 >= 0:
                    legs[idx-1]['end']['name'] = legs[idx -
                                                      1]['end']['name'] + '정류소'
                if idx + 1 < len(legs):
                    legs[idx+1]['start']['name'] = legs[idx +
                                                        1]['start']['name'] + '정류소'

        # generate result string looping over legs
        result = ""
        for idx, leg in enumerate(legs):
            # 하단역에서 하단역까지 걸어가는 것은 말이 안되므로 if문으로 제외한다.
            if leg['mode'] == 'WALK' and leg['end']['name'] != '하단역':
                result += f"{idx}. {leg['start']['name']}에서 {leg['end']['name']}까지 {leg['distance']}m 걸어가세요.\n"
                if idx == 0:
                    flag = False
                    for step in leg['steps']:
                        if '출구' in step['description']:
                            flag = True
                        if flag:
                            result += f" - {step['description']}\n"
            elif leg['mode'] == "SUBWAY":
                result += f"{idx}. {leg['start']['name']}에서 {leg['end']['name']}까지 지하철 {leg['route']} 타고 가세요.\n"
            elif leg['mode'] == "BUS":
                if leg.get('Lane'):
                    routes = [i['route'] for i in leg['Lane']]
                else:
                    routes = [leg['route']]
                result += f"{idx}. {leg['start']['name']}에서 {leg['end']['name']}까지 버스 {' 또는 '.join(routes)}번 타고 가세요.\n"

        result += f"**총 요금** : {fare}\n"

        return "<< HUMETRO AI ASSISTANT 길안내 서비스 >>\n" + result

    def _run(self, destination: str):
        try:
            TMap_response = self.call_TMapAPI(destination)
            if 'result' in TMap_response:
                result_str = TMap_response['result']['message']
            elif 'metaData' in TMap_response:
                itineraries = TMap_response['metaData']['plan']['itineraries']
                target_itinerary = itineraries[0]

                if len(itineraries) > 1:
                    print('multiple itineraries found')
                    # 1. 도시철도만 이용하는 방법이 있는 경우 그것을 우선적으로 제시한다.
                    itineraries.sort(key=lambda x: (
                        x['fare']['regular']['totalFare']))

                    def extract_modes(it):
                        modes = set()
                        for leg in it['legs']:
                            modes.add(leg['mode'])
                        return list(modes)

                    for it in itineraries:
                        target_modes = extract_modes(target_itinerary)
                        current_modes = extract_modes(it)
                        if 'BUS' in target_modes and 'BUS' not in current_modes:
                            target_itinerary = it
                            continue
                        if 'SUBWAY' in target_modes and 'SUBWAY' not in current_modes:
                            continue

                directions_dict = target_itinerary
                result_str = self.parse_directions_dict(directions_dict)
        except Exception as e:
            print(e)
            result_str = f"죄송합니다. {destination}에 대한 길안내를 찾지 못했습니다."

        return result_str

    args_schema: Optional[Type[BaseModel]] = TMapToolInputChecker


if __name__ == "__main__":
    import json
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv())  # read local .env file

    # 키워드로 장소 검색하기
    locations = [
        "OK큐병원", "가락타운1단지", "하단중학교", "곽요양병원", "다이소",
        "하단1동 행정복지센터", "하단초등학교", "가락타운2.3단지", "기업은행",
        "하단119 안전센터", "경남은행", "시에나웨딩뷔페", "스타벅스",
        "하단지구대", "프라임병원", "공영주차장", "우리은행", "새동아직업전문학교",
        "하단교차로", "아트몰링", "에덴공원", "NH농협은행", "경마공원 셔틀버스",
        "동아대학교 순환버스", "LG유플러스 고객지원센터",
        "하단우체국", "국민연금공단 서부산지사", "신용보증기금 사하지점", "버거킹",
        "하단교차로방면", "신평/다대포 버스", "기술보증기금 사하지점",
        "부산여고", "건국중.고교", "동아대학교", "헌혈의 집 하단센터",
        "하단시장", "부산신용보증재단 서부산지점", "신한은행", "부산본병원",
        "하단5일장", "삼성디지털프라자", "스타벅스", "롯데리아",
        "부산은행"
    ]

    busan_tourist_spots = [
        "해운대",
        "광안리",
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
        "영화의거리"
    ]

    tool = TMapTool()
    for loc in busan_tourist_spots:
        result = tool.run(loc)
        print(result)
