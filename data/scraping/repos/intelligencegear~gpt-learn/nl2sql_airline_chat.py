# ref: https://tianchi.aliyun.com/dataset/1078
import os
import openai

from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

import openai

prompt = """
Postgres SQL tables, with their properties:
Airline
 - id: 航班ID
 - date: 格式 YYYY-MM-dd
 - foreign: 国内/国际
 - from_city: 起飞城市
 - to_city: 到达城市
 - from_time: 起飞时间
 - to_time: 降落时间

Plane
 - id: 飞机ID
 - airline_id: 航班ID
 - plane_type: 飞机机型
 - date: 格式 YYYY-MM-dd

Traveler
 - id: 用户ID
 - airline_id: 航班ID
 - date: 格式 YYYY-MM-dd
 

请列出2022-03-01这一天，北京到杭州的所有航班id，飞机的机型，起飞和降落时间，以及乘客乘坐这趟航班的总数量
"""

response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "作为一名资深数据分析师, 给出了数据表的描述, 请写出一个详细并且争取的sql查询."},
        {"role": "user", "content": prompt},
    ]
)

print(response["choices"][0]["message"]["content"])


# 以下是查询语句：

# ```sql
# SELECT 
#     a.id AS flight_id, 
#     p.plane_type, 
#     a.from_time, 
#     a.to_time, 
#     COUNT(DISTINCT t.id) AS traveler_count
# FROM 
#     Airline a
# JOIN 
#     Plane p ON a.id = p.airline_id
# LEFT JOIN
#     Traveler t ON a.id = t.airline_id AND a.date = t.date
# WHERE 
#     a.date = '2022-03-01'
#     AND a.from_city = '北京'
#     AND a.to_city = '杭州'
# GROUP BY 
#     a.id, 
#     p.plane_type, 
#     a.from_time, 
#     a.to_time
# ORDER BY 
#     a.id ASC;
# ```

# 这个查询语句使用了 `JOIN` 和 `LEFT JOIN`，来将三个表联结起来并按照条件过滤数据。`JOIN` 连接 `Airline` 和 `Plane` 表，取出结果包含了所有的航班信息以及飞机机型；`LEFT JOIN` 连接 `Traveler` 表，则统计了当前这个航班下有多少个乘客。

# 筛选条件包括日期、出发城市和到达城市。结果按照航班id的升序排列，显示了每个符合条件的航班对应的飞机机型、起飞和降落时间，以及乘客数量。

# 注意到这里使用了 `COUNT(DISTINCT)`，这样可以确保每个乘客只会被计数一次。如果不加 `DISTINCT`，同一个用户多次乘坐同一趟航班将会被重复计数。