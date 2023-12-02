import csv
import os
from dotenv import load_dotenv
import openai
from time import sleep


def get_gpt_response(content, num):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "assistant", "name": "trial6" +
                str(num), "content": content}
        ],
        temperature=1.2
    )
    print("RESPONSE !")
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


col = ['id', 'kr_name', 'sci_name', 'grw_type', 'grw_speed',
       'min_temp', 'max_temp', 'winter_min_temp',
       'min_humidity', 'max_humidity', 'light_demand',
       'water_cycle', 'mgmt_level', 'mgmt_demand', 'place',
       'mgmt_tip', 'grw_season', 'characteristics',
       'insect_info', 'toxic_info', 'smell_deg', 'height', 'area',
       'description'
       ]

load_dotenv()

# load file
filename = os.environ.get('OUTPUT_DIR') + 'dry_preproc.csv'
fr = open(filename, 'r', encoding='utf-8-sig')
reader = csv.reader(fr)

# output file
filename = os.environ.get('OUTPUT_DIR') + 'dry_preproc_v2.csv'
fw = open(filename, 'w', encoding='utf-8-sig', newline='')
# fw = open(filename, 'a', encoding='utf-8-sig', newline='')
writer = csv.writer(fw)

# output columns
# writer.writerow(col)

# base URL
openai.api_key = os.environ.get('CHATGPT_API_KEY')

# read line
cnt = 0
api_cnt = 0
for row in reader:
    if row[0] == 'id':
        continue

    cnt += 1
    data = row

    # convert via Chat GPT
    message = "아래 내용의 문장을 친절하게 바꿔줘.\n"
    if data[15].strip() != '':
        tip_message = message + row[15]
        data[15] = get_gpt_response(tip_message, cnt)
        api_cnt += 1

    if api_cnt > 2:
        print("INFO : API call limits! wait 1 minute")
        api_cnt = 0
        sleep(60)

    if data[17].strip() != '':
        char_message = message + row[17]
        data[17] = get_gpt_response(char_message, cnt)
        api_cnt += 1

    if api_cnt > 2:
        print("INFO : API call limits! wait 1 minute")
        api_cnt = 0
        sleep(60)

    writer.writerow(data)
    print("[%3d] row done : " % (cnt))


print("DONE")
