import os
import json

from fastapi import APIRouter, HTTPException, Query, Body, Depends
from sqlalchemy.orm import Session

from typing import Any
from schema import Schedule, ScheduleOutput, PointOutput
from model import Point as PointModel
from func.ai import prompts

from db import AMZRDS, Mongo
from pymongo.database import Database as MongoDatabase

from datetime import date, datetime, timedelta


import openai
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat import ChatCompletion
openai.api_type = "azure"
openai.azure_endpoint = "https://hkust.azure-api.net"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.environ.get("AZURE_API_KEY")

router = APIRouter(
    prefix="/schedules")

# uid -> schedule.id
GLOBAL_SCHEDULE_CACHE = {}


def write_schedule_to_nosql(schedule: dict, uid: int = 1):
    mongo_conn: MongoDatabase = Mongo().get_connection()
    s_id = schedule['id']
    schedule['uid'] = uid
    try:
        mongo_conn['schedules'].update_one(
            {'id': s_id}, {"$set": schedule}, upsert=True)
        GLOBAL_SCHEDULE_CACHE[uid] = schedule
    except Exception as e:
        print(
            f'Unable to update schedule {schedule["id"]}: {e}')


def unzip_schedule(schedule: dict):
    mysql_conn = next(AMZRDS().get_connection())
    for day in schedule['days']:
        for block in day['blocks']:
            if block['type'] == "point":
                id = int(block['point']['id'])
                point_db = mysql_conn.query(PointModel).filter(
                    PointModel.id == id).first()
                block['point'] = PointOutput.from_orm(point_db).to_dict()
            elif block['type'] == "route":
                if isinstance(block['route']['origin'], dict) and 'id' in block['route']['origin']:
                    origin_id = int(block['route']['origin']['id'])
                    origin_db = mysql_conn.query(PointModel).filter(
                        PointModel.id == origin_id).first()
                    block['route']['origin'] = PointOutput.from_orm(
                        origin_db).to_dict()
                if isinstance(block['route']['destination'], dict) and 'id' in block['route']['destination']:
                    destination_id = int(block['route']['destination']['id'])
                    destination_db = mysql_conn.query(PointModel).filter(
                        PointModel.id == destination_id).first()
                    block['route']['destination'] = PointOutput.from_orm(
                        destination_db).to_dict()
    return schedule


def convert_json_in_text_to_dict(text):
    # 去除{之前的多余内容，这条语句同时适用于{之前没有多余内容的情况
    t = text[len(text.split('{')[0]):]
    suffix_length = len(text.split('}')[-1])
    # 去除}之后的多余内容, }之后没有多余内容时，则不用处理
    if suffix_length:
        t = t[:-suffix_length]
    # 转换成字典返回
    return json.loads(t)


def zip_schedule(schedule: dict):
    for day in schedule['days']:
        for block in day['blocks']:
            if block['type'] == "point":
                block['point'] = {
                    'id': block['point']['id'],
                    'name': block['point']['name'],
                    'address': block['point']['address'],
                    'latLng': block['point']['latLng'],
                }
            elif block['type'] == "route":
                block['route']['origin'] = {
                    'id': block['route']['origin']['id'],
                    'name': block['route']['origin']['name'],
                    'address': block['route']['origin']['address'],
                    'latLng': block['route']['origin']['latLng'],
                }
                block['route']['destination'] = {
                    'id': block['route']['destination']['id'],
                    'name': block['route']['destination']['name'],
                    'address': block['route']['destination']['address'],
                    'latLng': block['route']['destination']['latLng'],
                }
    return schedule


@router.post("/start")
# TODO: uid放 session 里获取
def start_new_schedule_draft(
    uid: int = Body(1),
    pids: list[int] = Body([]),
    options: dict[str, Any] = Body({}),
    start: date = Body('2023-01-01'),
    end: date = Body('2023-01-02'),
    city: str = Body(default="Beijing"),
    mysql_conn: Session = Depends(AMZRDS().get_connection)
) -> Schedule:

    # 根据 pid，获取 points
    points: list[PointModel] = mysql_conn.query(
        PointModel).filter(PointModel.id.in_(pids)).all()
    if not points:
        raise HTTPException(status_code=400, detail="Invalid point id")

    slim_points = []
    for point in points:
        slim_points.append({
            "id": point.id,
            "name": point.name,
            "latLng": {
                "lat": float(point.latitude),
                "lng": float(point.longitude)
            },
            "address": point.address})

    input = json.dumps({
        'start': str(start),
        'end': str(end),
        'points': slim_points,
    })

    resp: ChatCompletion = openai.chat.completions.create(
        model='gpt-4',
        messages=[
            {
                'role': 'system',
                'content': prompts['start_new_schedule_draft']['system'],
            },
            *prompts['start_new_schedule_draft']['examples'],
            {
                'role': 'user',
                'content': input,
            }
        ],  # type: ignore
        n=1,
        temperature=0,
        top_p=1,
        # response_format={'type': 'text'},
    )
    choice = resp.choices[0]
    content: str = choice.message.content or ""
    res = json.loads(content)
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    res['id'] = f"{uid}-{current_time}"
    res['name'] = f"{city}-{start}-{end}"

    res = unzip_schedule(res)

    w = write_schedule_to_nosql(res, uid)

    return res  # type: ignore


@router.post("/refine")
def refine_schedule(
    uid: int = Body(1),
    refine_chat: str = Body(...),
    draft: dict = Body({}),
):
    if not draft:
        draft = GLOBAL_SCHEDULE_CACHE[uid]
    s_id = draft['id']
    del draft['id']
    draft = zip_schedule(draft)
    resp: ChatCompletion = openai.chat.completions.create(
        model='gpt-4',
        messages=[
            {
                'role': 'system',
                'content': prompts['start_new_schedule_draft']['system'],
            },
            *prompts['start_new_schedule_draft']['examples'],
            *prompts['refine_schedule']['examples'],
            {
                'role': 'user',
                'content': refine_chat + "(reply with only json')\n'+'draft:\n" + json.dumps(draft)
            }
        ],  # type: ignore
        n=1,
        temperature=0,
        top_p=1,
        # response_format={'type': 'text'},
    )
    choice = resp.choices[0]
    content: str = choice.message.content or '{}'

    try:
        res = convert_json_in_text_to_dict(content)
        res['id'] = s_id
        res = unzip_schedule(res)
        write_schedule_to_nosql(res, uid)
        return res
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="OpenAI Result Error")


@router.post("/")
def create_schedule(uid: int = 1, schedule: Schedule = Body(...)):

    return {}


@router.delete("/{id}")
def delete_schedule_by_id(id: str):
    mongo_conn = Mongo().get_connection()
    try:
        mongo_conn['schedules'].delete_one({'id': id})
    except Exception as e:
        print(
            'Unable to delete schedule {id}: {e}')


@router.get("/{id}", response_model=ScheduleOutput,)
def get_schedule_by_id(id: str) -> ScheduleOutput:
    mongo_conn = Mongo().get_connection()
    schedule = mongo_conn['schedules'].find_one({'id': id})
    if schedule is None:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return schedule


@router.get('/history/{uid}')
def get_user_schedule_history(uid: int) -> list[ScheduleOutput]:
    mongo_conn = Mongo().get_connection()
    schedules = mongo_conn['schedules'].find({'uid': uid})

    def clean(schedule: dict) -> ScheduleOutput:
        del schedule['_id']
        return ScheduleOutput(**schedule)

    res: list[ScheduleOutput] = [clean(x) for x in schedules]
    return res

# @router.put("/update")
# def update_schedule_by_id(id: str):
#     return {}


if __name__ == "__main__":

    res = json.loads('''
                    {
        "uid": 2,
        "refine_chat": "help me add some resturants in my schedule",
        "draft": {
            "id": "2-20231127212547",
            "name": "Beijing-2023-07-01-2023-07-03",
            "start": "2023-07-01",
            "end": "2023-07-03",
            "days": [
                {
                    "day": "2023-07-01",
                    "blocks": [
                        {
                            "type": "point",
                            "point": {
                                "id": 8,
                                "name": "The Forbidden City",
                                "latLng": {
                                    "latitude": 39.916344,
                                    "longitude": 116.397155
                                },
                                "address": "4 Jingshan Front St, Dongcheng, Beijing, China",
                                "mapid": null,
                                "pic": [
                                    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/The_Forbidden_City_-_View_from_Coal_Hill.jpg/1920px-The_Forbidden_City_-_View_from_Coal_Hill.jpg",
                                    "https://example.com/forbidden_city2.jpg"
                                ],
                                "tag": [
                                    "historical_sites",
                                    "cultural_experiences"
                                ],
                                "city": "Beijing",
                                "introduction": "The Forbidden City, also known as the Palace Museum, is a large palace complex and a UNESCO World Heritage Site. It served as the imperial palace for 24 emperors during the Ming and Qing dynasties.",
                                "options": {
                                    "pic": [
                                        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/The_Forbidden_City_-_View_from_Coal_Hill.jpg/1920px-The_Forbidden_City_-_View_from_Coal_Hill.jpg",
                                        "https://example.com/forbidden_city2.jpg"
                                    ],
                                    "tag": [
                                        "historical_sites",
                                        "cultural_experiences"
                                    ]
                                }
                            },
                            "route": null,
                            "start": "10:00:00",
                            "end": "12:00:00",
                            "activity": "Visit The Forbidden City"
                        },
                        {
                            "type": "route",
                            "point": null,
                            "route": {
                                "origin": {
                                    "id": 8,
                                    "name": "The Forbidden City",
                                    "latLng": {
                                        "latitude": 39.916344,
                                        "longitude": 116.397155
                                    },
                                    "address": "4 Jingshan Front St, Dongcheng, Beijing, China",
                                    "mapid": null,
                                    "pic": [
                                        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/The_Forbidden_City_-_View_from_Coal_Hill.jpg/1920px-The_Forbidden_City_-_View_from_Coal_Hill.jpg",
                                        "https://example.com/forbidden_city2.jpg"
                                    ],
                                    "tag": [
                                        "historical_sites",
                                        "cultural_experiences"
                                    ],
                                    "city": "Beijing",
                                    "introduction": "The Forbidden City, also known as the Palace Museum, is a large palace complex and a UNESCO World Heritage Site. It served as the imperial palace for 24 emperors during the Ming and Qing dynasties.",
                                    "options": {
                                        "pic": [
                                            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/The_Forbidden_City_-_View_from_Coal_Hill.jpg/1920px-The_Forbidden_City_-_View_from_Coal_Hill.jpg",
                                            "https://example.com/forbidden_city2.jpg"
                                        ],
                                        "tag": [
                                            "historical_sites",
                                            "cultural_experiences"
                                        ]
                                    }
                                },
                                "destination": {
                                    "id": 10,
                                    "name": "Temple of Heaven",
                                    "latLng": {
                                        "latitude": 39.883704,
                                        "longitude": 116.412842
                                    },
                                    "address": "1 Tiantan E Rd, Dongcheng, Beijing, China",
                                    "mapid": null,
                                    "pic": [
                                        "https://example.com/temple_of_heaven1.jpg",
                                        "https://example.com/temple_of_heaven2.jpg"
                                    ],
                                    "tag": [
                                        "historical_sites",
                                        "cultural_experiences"
                                    ],
                                    "city": "Beijing",
                                    "introduction": "The Temple of Heaven is an imperial complex of religious buildings and a UNESCO World Heritage Site. It was visited by the emperors of the Ming and Qing dynasties for annual ceremonies of prayer to Heaven for a good harvest.",
                                    "options": {
                                        "pic": [
                                            "https://example.com/temple_of_heaven1.jpg",
                                            "https://example.com/temple_of_heaven2.jpg"
                                        ],
                                        "tag": [
                                            "historical_sites",
                                            "cultural_experiences"
                                        ]
                                    }
                                },
                                "steps": [
                                    {
                                        "start": "The Forbidden City",
                                        "end": "Tiananmen East Station",
                                        "step": "Metro Line 1, 2 stops",
                                        "duration": 300,
                                        "distance": 3000
                                    },
                                    {
                                        "start": "Tiananmen East Station",
                                        "end": "Temple of Heaven Station",
                                        "step": "Metro Line 5, 2 stops",
                                        "duration": 300,
                                        "distance": 4000
                                    },
                                    {
                                        "start": "Temple of Heaven Station",
                                        "end": "Temple of Heaven",
                                        "step": "500 metres on foot",
                                        "duration": 300,
                                        "distance": 500
                                    }
                                ],
                                "duration": 900,
                                "distance": 7500
                            },
                            "start": "12:00:00",
                            "end": "12:30:00",
                            "activity": "Travel to Temple of Heaven"
                        },
                        {
                            "type": "point",
                            "point": {
                                "id": 10,
                                "name": "Temple of Heaven",
                                "latLng": {
                                    "latitude": 39.883704,
                                    "longitude": 116.412842
                                },
                                "address": "1 Tiantan E Rd, Dongcheng, Beijing, China",
                                "mapid": null,
                                "pic": [
                                    "https://example.com/temple_of_heaven1.jpg",
                                    "https://example.com/temple_of_heaven2.jpg"
                                ],
                                "tag": [
                                    "historical_sites",
                                    "cultural_experiences"
                                ],
                                "city": "Beijing",
                                "introduction": "The Temple of Heaven is an imperial complex of religious buildings and a UNESCO World Heritage Site. It was visited by the emperors of the Ming and Qing dynasties for annual ceremonies of prayer to Heaven for a good harvest.",
                                "options": {
                                    "pic": [
                                        "https://example.com/temple_of_heaven1.jpg",
                                        "https://example.com/temple_of_heaven2.jpg"
                                    ],
                                    "tag": [
                                        "historical_sites",
                                        "cultural_experiences"
                                    ]
                                }
                            },
                            "route": null,
                            "start": "12:30:00",
                            "end": "14:00:00",
                            "activity": "Visit Temple of Heaven"
                        },
                        {
                            "type": "route",
                            "point": null,
                            "route": {
                                "origin": {
                                    "id": 10,
                                    "name": "Temple of Heaven",
                                    "latLng": {
                                        "latitude": 39.883704,
                                        "longitude": 116.412842
                                    },
                                    "address": "1 Tiantan E Rd, Dongcheng, Beijing, China",
                                    "mapid": null,
                                    "pic": [
                                        "https://example.com/temple_of_heaven1.jpg",
                                        "https://example.com/temple_of_heaven2.jpg"
                                    ],
                                    "tag": [
                                        "historical_sites",
                                        "cultural_experiences"
                                    ],
                                    "city": "Beijing",
                                    "introduction": "The Temple of Heaven is an imperial complex of religious buildings and a UNESCO World Heritage Site. It was visited by the emperors of the Ming and Qing dynasties for annual ceremonies of prayer to Heaven for a good harvest.",
                                    "options": {
                                        "pic": [
                                            "https://example.com/temple_of_heaven1.jpg",
                                            "https://example.com/temple_of_heaven2.jpg"
                                        ],
                                        "tag": [
                                            "historical_sites",
                                            "cultural_experiences"
                                        ]
                                    }
                                },
                                "destination": {
                                    "id": 8,
                                    "name": "The Forbidden City",
                                    "latLng": {
                                        "latitude": 39.916344,
                                        "longitude": 116.397155
                                    },
                                    "address": "4 Jingshan Front St, Dongcheng, Beijing, China",
                                    "mapid": null,
                                    "pic": [
                                        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/The_Forbidden_City_-_View_from_Coal_Hill.jpg/1920px-The_Forbidden_City_-_View_from_Coal_Hill.jpg",
                                        "https://example.com/forbidden_city2.jpg"
                                    ],
                                    "tag": [
                                        "historical_sites",
                                        "cultural_experiences"
                                    ],
                                    "city": "Beijing",
                                    "introduction": "The Forbidden City, also known as the Palace Museum, is a large palace complex and a UNESCO World Heritage Site. It served as the imperial palace for 24 emperors during the Ming and Qing dynasties.",
                                    "options": {
                                        "pic": [
                                            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/The_Forbidden_City_-_View_from_Coal_Hill.jpg/1920px-The_Forbidden_City_-_View_from_Coal_Hill.jpg",
                                            "https://example.com/forbidden_city2.jpg"
                                        ],
                                        "tag": [
                                            "historical_sites",
                                            "cultural_experiences"
                                        ]
                                    }
                                },
                                "steps": [
                                    {
                                        "start": "Temple of Heaven",
                                        "end": "Tiantan East Gate Station",
                                        "step": "Metro Line 5, 2 stops",
                                        "duration": 300,
                                        "distance": 4000
                                    },
                                    {
                                        "start": "Tiantan East Gate Station",
                                        "end": "Tiananmen East Station",
                                        "step": "Metro Line 5, 2 stops",
                                        "duration": 300,
                                        "distance": 4000
                                    },
                                    {
                                        "start": "Tiananmen East Station",
                                        "end": "The Forbidden City",
                                        "step": "500 metres on foot",
                                        "duration": 300,
                                        "distance": 500
                                    }
                                ],
                                "duration": 900,
                                "distance": 7500
                            },
                            "start": "14:00:00",
                            "end": "14:30:00",
                            "activity": "Travel back to The Forbidden City"
                        },
                        {
                            "type": "point",
                            "point": {
                                "id": 8,
                                "name": "The Forbidden City",
                                "latLng": {
                                    "latitude": 39.916344,
                                    "longitude": 116.397155
                                },
                                "address": "4 Jingshan Front St, Dongcheng, Beijing, China",
                                "mapid": null,
                                "pic": [
                                    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/The_Forbidden_City_-_View_from_Coal_Hill.jpg/1920px-The_Forbidden_City_-_View_from_Coal_Hill.jpg",
                                    "https://example.com/forbidden_city2.jpg"
                                ],
                                "tag": [
                                    "historical_sites",
                                    "cultural_experiences"
                                ],
                                "city": "Beijing",
                                "introduction": "The Forbidden City, also known as the Palace Museum, is a large palace complex and a UNESCO World Heritage Site. It served as the imperial palace for 24 emperors during the Ming and Qing dynasties.",
                                "options": {
                                    "pic": [
                                        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/The_Forbidden_City_-_View_from_Coal_Hill.jpg/1920px-The_Forbidden_City_-_View_from_Coal_Hill.jpg",
                                        "https://example.com/forbidden_city2.jpg"
                                    ],
                                    "tag": [
                                        "historical_sites",
                                        "cultural_experiences"
                                    ]
                                }
                            },
                            "route": null,
                            "start": "14:30:00",
                            "end": "17:00:00",
                            "activity": "Explore The Forbidden City"
                        }
                    ]
                },
                {
                    "day": "2023-07-02",
                    "blocks": [
                        {
                            "type": "point",
                            "point": {
                                "id": 9,
                                "name": "The Great Wall of China",
                                "latLng": {
                                    "latitude": 40.431908,
                                    "longitude": 116.570374
                                },
                                "address": "Huairou, Beijing, China",
                                "mapid": null,
                                "pic": [
                                    "https://example.com/great_wall1.jpg",
                                    "https://example.com/great_wall2.jpg"
                                ],
                                "tag": [
                                    "historical_sites",
                                    "cultural_experiences"
                                ],
                                "city": "Beijing",
                                "introduction": "The Great Wall of China is a series of fortifications made of stone, brick, and other materials, built to protect China's northern borders. It is a UNESCO World Heritage Site and one of the most famous landmarks in the world.",
                                "options": {
                                    "pic": [
                                        "https://example.com/great_wall1.jpg",
                                        "https://example.com/great_wall2.jpg"
                                    ],
                                    "tag": [
                                        "historical_sites",
                                        "cultural_experiences"
                                    ]
                                }
                            },
                            "route": null,
                            "start": "08:00:00",
                            "end": "17:00:00",
                            "activity": "Visit The Great Wall of China"
                        }
                    ]
                },
                {
                    "day": "2023-07-03",
                    "blocks": [
                        {
                            "type": "point",
                            "point": {
                                "id": 11,
                                "name": "Summer Palace",
                                "latLng": {
                                    "latitude": 39.99944,
                                    "longitude": 116.2755
                                },
                                "address": "19 Xinjiangongmen Rd, Haidian District, Beijing, China",
                                "mapid": null,
                                "pic": [
                                    "https://example.com/summer_palace1.jpg",
                                    "https://example.com/summer_palace2.jpg"
                                ],
                                "tag": [
                                    "historical_sites",
                                    "cultural_experiences",
                                    "citywalk"
                                ],
                                "city": "Beijing",
                                "introduction": "The Summer Palace is a vast ensemble of lakes, gardens, and palaces in Beijing. It is a UNESCO World Heritage Site and a popular tourist destination.",
                                "options": {
                                    "pic": [
                                        "https://example.com/summer_palace1.jpg",
                                        "https://example.com/summer_palace2.jpg"
                                    ],
                                    "tag": [
                                        "historical_sites",
                                        "cultural_experiences",
                                        "citywalk"
                                    ]
                                }
                            },
                            "route": null,
                            "start": "10:00:00",
                            "end": "13:00:00",
                            "activity": "Visit Summer Palace"
                        },
                        {
                            "type": "route",
                            "point": null,
                            "route": {
                                "origin": {
                                    "id": 11,
                                    "name": "Summer Palace",
                                    "latLng": {
                                        "latitude": 39.99944,
                                        "longitude": 116.2755
                                    },
                                    "address": "19 Xinjiangongmen Rd, Haidian District, Beijing, China",
                                    "mapid": null,
                                    "pic": [
                                        "https://example.com/summer_palace1.jpg",
                                        "https://example.com/summer_palace2.jpg"
                                    ],
                                    "tag": [
                                        "historical_sites",
                                        "cultural_experiences",
                                        "citywalk"
                                    ],
                                    "city": "Beijing",
                                    "introduction": "The Summer Palace is a vast ensemble of lakes, gardens, and palaces in Beijing. It is a UNESCO World Heritage Site and a popular tourist destination.",
                                    "options": {
                                        "pic": [
                                            "https://example.com/summer_palace1.jpg",
                                            "https://example.com/summer_palace2.jpg"
                                        ],
                                        "tag": [
                                            "historical_sites",
                                            "cultural_experiences",
                                            "citywalk"
                                        ]
                                    }
                                },
                                "destination": {
                                    "id": 8,
                                    "name": "The Forbidden City",
                                    "latLng": {
                                        "latitude": 39.916344,
                                        "longitude": 116.397155
                                    },
                                    "address": "4 Jingshan Front St, Dongcheng, Beijing, China",
                                    "mapid": null,
                                    "pic": [
                                        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/The_Forbidden_City_-_View_from_Coal_Hill.jpg/1920px-The_Forbidden_City_-_View_from_Coal_Hill.jpg",
                                        "https://example.com/forbidden_city2.jpg"
                                    ],
                                    "tag": [
                                        "historical_sites",
                                        "cultural_experiences"
                                    ],
                                    "city": "Beijing",
                                    "introduction": "The Forbidden City, also known as the Palace Museum, is a large palace complex and a UNESCO World Heritage Site. It served as the imperial palace for 24 emperors during the Ming and Qing dynasties.",
                                    "options": {
                                        "pic": [
                                            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/The_Forbidden_City_-_View_from_Coal_Hill.jpg/1920px-The_Forbidden_City_-_View_from_Coal_Hill.jpg",
                                            "https://example.com/forbidden_city2.jpg"
                                        ],
                                        "tag": [
                                            "historical_sites",
                                            "cultural_experiences"
                                        ]
                                    }
                                },
                                "steps": [
                                    {
                                        "start": "Summer Palace",
                                        "end": "Beigongmen Station",
                                        "step": "Metro Line 4, 5 stops",
                                        "duration": 600,
                                        "distance": 8000
                                    },
                                    {
                                        "start": "Beigongmen Station",
                                        "end": "Tiananmen East Station",
                                        "step": "Metro Line 4, 5 stops",
                                        "duration": 600,
                                        "distance": 8000
                                    },
                                    {
                                        "start": "Tiananmen East Station",
                                        "end": "The Forbidden City",
                                        "step": "500 metres on foot",
                                        "duration": 300,
                                        "distance": 500
                                    }
                                ],
                                "duration": 1500,
                                "distance": 16500
                            },
                            "start": "13:00:00",
                            "end": "14:00:00",
                            "activity": "Travel back to The Forbidden City"
                        },
                        {
                            "type": "point",
                            "point": {
                                "id": 8,
                                "name": "The Forbidden City",
                                "latLng": {
                                    "latitude": 39.916344,
                                    "longitude": 116.397155
                                },
                                "address": "4 Jingshan Front St, Dongcheng, Beijing, China",
                                "mapid": null,
                                "pic": [
                                    "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/The_Forbidden_City_-_View_from_Coal_Hill.jpg/1920px-The_Forbidden_City_-_View_from_Coal_Hill.jpg",
                                    "https://example.com/forbidden_city2.jpg"
                                ],
                                "tag": [
                                    "historical_sites",
                                    "cultural_experiences"
                                ],
                                "city": "Beijing",
                                "introduction": "The Forbidden City, also known as the Palace Museum, is a large palace complex and a UNESCO World Heritage Site. It served as the imperial palace for 24 emperors during the Ming and Qing dynasties.",
                                "options": {
                                    "pic": [
                                        "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/The_Forbidden_City_-_View_from_Coal_Hill.jpg/1920px-The_Forbidden_City_-_View_from_Coal_Hill.jpg",
                                        "https://example.com/forbidden_city2.jpg"
                                    ],
                                    "tag": [
                                        "historical_sites",
                                        "cultural_experiences"
                                    ]
                                }
                            },
                            "route": null,
                            "start": "14:00:00",
                            "end": "17:00:00",
                            "activity": "Explore The Forbidden City"
                        }
                    ]
                }
            ],
            "options": {}
        }
    }
                    ''')
    with open('test.json', 'w') as f:
        f.write(json.dumps(zip_schedule(res['draft'])))
