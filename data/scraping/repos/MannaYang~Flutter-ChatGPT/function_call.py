import requests
from langchain import PromptTemplate

default_template = """请用中文回答问题

历史对话:
{history}
Human: {input}
AI:
"""

default_prompt = PromptTemplate(input_variables=["history", "input"], template=default_template)

# 参数
# functions call
functionCall = [
    'query_attendance_data',
    'query_inventory_data',
    'submit_leave_data'
]
# functions info
functionInfo = [
    {
        "name": "query_attendance_data",
        "description":
            "Query departmental attendance data for the current time",
        "parameters": {
            "type": "object",
            "properties": {
                "attendance_date": {
                    "type": "string",
                    "description":
                        "Attendance dates, such as：2023-07-17，2023-07-16，2023-07-15，2023-07-14，format is{yyyy-MM-dd}"
                },
                "attendance_depart": {
                    "type": "string",
                    "description":
                        "Attendance departments,such as：研发部(R&D),市场部(Marketing),人力资源(HR)"
                }
            },
            "required": ["attendance_date", 'attendance_depart']
        }
    },
    {
        "name": "query_inventory_data",
        "description": "Query Zeiss lens inventory data",
        "parameters": {
            "type": "object",
            "properties": {
                "brand": {
                    "type": "string",
                    "description":
                        "Brand name,such as：Zeiss，Essilor，format is{brand：}"
                },
                "sku_code": {
                    "type": "string",
                    "description":
                        "Sku code,such as：78969993499538304,format is{skuCode：}"
                }
            },
            "required": ["brand"]
        }
    },
    {
        "name": "submit_leave_data",
        "description":
            "Submission of leave requests based on the given date-time-reason",
        "parameters": {
            "type": "object",
            "properties": {
                "date_start": {
                    "type": "string",
                    "description":
                        "Leave start date,such as：2023-07-18，2023-07-17，2023-07-16，format is{yyyy-MM-dd}"
                },
                "time_start": {
                    "type": "string",
                    "description":
                        "Leave start time,such as：09:00,10:00,11:00,format is{HH:mm}"
                },
                "date_end": {
                    "type": "string",
                    "description":
                        "Leave end date,such as：2023-07-18，2023-07-17，2023-07-16，format is{yyyy-MM-dd}"
                },
                "time_end": {
                    "type": "string",
                    "description":
                        "Leave end time,such as：16:00,17:00,18:00,format is{HH:mm}"
                },
                "leave_reason": {
                    "type": "string",
                    "description":
                        "Leave reason,such as：Unable to go to work normally due to hot weather，Need to go to the hospital if you are not feeling well，Children are sick and need to be taken care of"
                },
            },
            "required": [
                "date_start",
                "time_start",
                "date_end",
                "time_end",
                "leave_reason"
            ]
        }
    },
]


#
# Query Zeiss lens inventory data with ownerName and skuCode
#
async def query_inventory_data(brand: str, sku_code: str):
    print('Query Zeiss lens inventory data with ownerName and skuCode')
    url = "https://jsonplaceholder.typicode.com/posts"
    headers = {
        "content-type": "application/json"
        # 'Content-Type': 'application/json; charset=UTF-8'
    }
    body = {
        "code": 0,
        "success": True,
        "result": [
            {
                "skuCode": sku_code,
                "brand": brand,
                "model": "Zeiss-2023",
                "lensType": "lens001",
                "stockQuantity": 3000
            },
            {
                "skuCode": sku_code,
                "brand": brand,
                "model": "Zeiss-2023",
                "lensType": "lens002",
                "stockQuantity": 200
            },
            {
                "skuCode": sku_code,
                "brand": brand,
                "model": "Zeiss-2023",
                "lensType": "lens003",
                "stockQuantity": 100
            },
        ]
    }
    print(f"Query Zeiss lens inventory params - {body}")
    return requests.post(url, json=body, headers=headers)


#
# Query attendance data
#
async def query_attendance_data(date: str, depart: str):
    print('Query attendance data')
    url = 'https://jsonplaceholder.typicode.com/posts'
    headers = {"content-type": "application/json"}
    body = {
        "code": 0,
        "success": True,
        "result": {
            "depart": depart,
            "dateStart": date,
            "dateEnd": date,
            "lateArrival": 30,
            "earlyDeparture": 3,
            "absenteeism": 1,
            "leave": 2,
            "businessTrips": 10
        }
    }
    return requests.post(url, json=body, headers=headers)


#
# Submit leave data
#
async def submit_leave_data(
        date_start: str,
        time_start: str,
        date_end: str,
        time_end: str,
        leave_reason: str
):
    print('Submit leave data')
    url = 'https://jsonplaceholder.typicode.com/posts'
    headers = {
        'Content-Type': 'application/json; charset=UTF-8'
    }
    body = {
        'code': 0,
        'success': True,
        'result': {
            "dateStart": date_start,
            "timeStart": time_start,
            "dateEnd": date_end,
            "timeEnd": time_end,
            "leaveReason": leave_reason,
            "leaveStatus": "Leave submitted successfully"
        }
    }
    return requests.post(url, json=body, headers=headers)
