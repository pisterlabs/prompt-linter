import random
import requests
import sqlite3
import os,sys
from pathlib import Path
import openai
import time
from .Notes import Note
from .db_operation import (
    init_db,
    insert_data,
    update_db,
    get_latest_cursor_pid,
    fill_blank_content,
)
from utils.Headers import HEADERS
sys.path.append(str(Path(__file__).resolve().parent.parent))

api_key = "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"
openai.api_key = api_key

# timestamp_to_datetime(1691243943000)
#  note_num = session.get(url=URL, params=params, headers=HEADERS).json()['data']['total_note_count'] #当前笔记总数


URL = "https://www.xiaohongshu.com/web_api/sns/v3/page/notes"

categories = [
    "OTA功能分享：在这个类别中，用户分享特斯拉车辆通过在线升级（OTA）获得的新功能或改进。这可能涉及到车辆功能的扩展、性能的优化以及新的驾驶辅助功能的引入等。用户可以分享他们的使用体验和对新功能的评价。",
    "充电：这个类别讨论特斯拉车辆的充电方式、充电速度、充电站点等相关话题。用户可能分享他们的充电策略、使用超级充电站的体验，或者讨论在不同情况下的充电效率和电池寿命等问题。",
    "导航：该类别关注特斯拉车载导航系统的使用体验、导航路线规划、导航数据准确性等讨论。用户可以分享他们的导航体验，是否准确地导航到目的地，以及导航系统的功能是否满足需求等。",
    "自用好物：这个类别是分享用户购买的与特斯拉车辆相关的个人用品、周边产品或改装配件等。用户可能推荐实用的车内用品、车载充电器、车身贴纸等产品，或者分享他们自己的改装经历和效果。重点是不含广告或者引流倾向的内容",
    "驾驶体验：在这个类别中，用户分享特斯拉车辆驾驶体验、操控感受、加速性能、舒适性等方面的内容。他们可能描述驾驶特斯拉的乐趣，操控与其他车型的对比，以及在不同道路条件下的表现。",
    "咨询问题：这个类别涉及用户对特斯拉车辆或公司的疑问、问题咨询与解答。其他用户或专业人士可能回答问题，提供帮助和解决方案。",
    "提车经历：用户在这个类别分享提取新车的经历、购车流程、交付体验等内容。这可能包括预约交付，交付过程中的顾虑和惊喜，以及对特斯拉交付服务的评价。",
    "特斯拉中国官方精品：该类别介绍特斯拉中国官方推出的精品商品、周边产品、限量版车型等。用户可能分享新产品的外观和功能，或者讨论特斯拉与其他品牌的合作。",
    "能耗：在这个类别讨论特斯拉车辆的能耗情况、电池续航里程、充电效率等。用户可能分享不同模式下的能耗数据，充电对续航里程的影响，以及一些节能的驾驶技巧。",
    "影音娱乐：这个类别与特斯拉车载影音娱乐系统、音响配置、多媒体功能相关。用户可能讨论车载娱乐体验，音响效果的好坏，以及与手机连接的功能。",
    "购车相关：该类别涉及购买特斯拉车辆的流程、政策、购车优惠等信息。用户可能分享购车经验，购车时需要注意的事项，以及政策变化对购车的影响。",
    "吐槽：用户在这个类别发表对特斯拉车辆、服务或公司的吐槽、批评与不满意之处。这可以是对特定问题的抱怨，或者对产品或服务的改建议。",
    "马斯克：在这个类别讨论特斯拉公司CEO埃隆·马斯克的言论、行动、公司动态等相关话题。用户可能分享关于马斯克的采访、社交媒体上的言论，或者对他领导风格的看法。",
    "召回：该类别包含特斯拉车辆召回信息、维修保养、售后服务等内容。用户可能分享召回经历，售后服务质量，以及维修保养的建议。",
    "刹车：在这个类别与特斯拉车辆刹车性能、刹车系统、刹车噪音等相关讨论。用户可能分享刹车效果，刹车系统的工作原理，以及刹车维护注意事项。",
    "新闻评论：这个类别允许用户对特斯拉相关新闻报道进行评论、观点分享等。用户可以表达对新闻事件的看法，讨论新闻对特斯拉未来发展的影响等。",
    "卖车：该类别包含用户出售特斯拉车辆、二手交易相关信息。用户可以发布车辆出售信息，或者询问关于二手车交易的相关问题。",
    "广告：这个类别涉及第三方商家对车内用品，附件，改装，保险，服务的广告和推广，和自用好物的区别是有明显的广告倾向，比如商家地址，联系方式，店铺名称，购买渠道，手机号等等",
]


class Request_note:
    def __init__(self):
        pass


def generate_req(pid, _cursor, req_num):
    req = f"""GET /你的端点URL?page_size={req_num + 1}&sort=time&page_id={pid}&cursor={_cursor} HTTP/1.1
        Host: 你的主机
        User-Agent: 你的用户代理
        Authorization: 你的授权令牌"""
    return req


def get_data_from_web(pid, _cursor, req_num):
    data_list = []
    session = requests.session()
    params = {
        "page_size": req_num + 1,
        "sort": "time",
        "page_id": pid,
        "cursor": _cursor,
        
    }
    try:
        response = session.get(url=URL, params=params, headers=HEADERS)
        response.raise_for_status()  # Raise an exception for non-200 status codes
        notes = response.json()["data"]["notes"]

        cur_cursor = response.json()["data"]["cursor"]
        cursor_list.append(cur_cursor)
        if len(notes) != req_num + 1:
            print(f"Not enough data, we only receive {len(notes)} of {req_num + 1}")

        for note in notes:
            data_list.append(
                Note(
                    note["id"],
                    note["type"],
                    note["likes"],
                    note["title"],
                    note["user"]["nickname"],
                    note["create_time"],
                    note["image_count"],
                )
            )

    except requests.exceptions.RequestException as e:
        print("Request Error:", e)
    except KeyError as ke:
        print("KeyError:", ke)
    except Exception as ex:
        print("An error occurred:", ex)
    finally:
        session.close()

    return data_list


def finish_data(notes):
    for note in notes:
        try:
            note.get_date()
        except Exception as date_exception:
            print("Error while getting date:", date_exception)

        try:
            note.get_content()
        except Exception as content_exception:
            print("Error while getting content:", content_exception)

        # try:
        #     note.get_category(categories)
        # except Exception as category_exception:
        #     print("Error while getting category:", category_exception)

    return notes


global file_path
file_path = "..database/xhs_tesla_notes.db"


def main(num=10, loops=10):
    if os.path.exists(file_path):
        print(f"检测到数据库{file_path}存在")
    else:
        print(f"检测到数据库{file_path}不存在，正在初始化数据库")
        init_db("xhs_tesla_notes.db")
        print(f"{file_path}数据库初始化完成")

    conn = sqlite3.connect(file_path)
    cursor = conn.cursor()

    page_id = "5be6e4bcdb601f0001f72453"  # 页面id
    global cursor_list
    # cursor_list = ['mA-kVb-s87jDFkCthC9RQg']
    cursor_list = ["1spX9G02HLT3E4fzy35Aww"]

    for _ in range(loops):
        notes = get_data_from_web(page_id, cursor_list[-1], num)
        finish_data(notes)
        insert_data(conn, cursor, notes)
        time.sleep(0.5)

    conn.close()

    print(f"The next cursor as initator is : {cursor_list[-1]}")


if __name__ == "__main__":
    # main(num = 10,loops= 100)
    fill_blank_content()

# update_db('xhs_tesla_notes.db')


# get_latest_cursor("xhs_tesla_notes.db")


