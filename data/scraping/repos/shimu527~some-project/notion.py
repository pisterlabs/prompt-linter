import openai_secret_manager
import openai
from notion_client import Client

# 获取 Notion API 访问令牌
secrets = openai_secret_manager.get_secret("notion")
notion = Client(auth=secrets["api_key"])

# 定义数据库 ID 和属性名称
database_id = "your_database_id"
target_property_name = "target_property_name"
tag_property_name = "tag_property_name"

# 查询数据库
results = notion.databases.query(
    **{
        "database_id": database_id,
        "filter": {
            "property": target_property_name,
            "title": {
                "contains": "your_keyword"
            }
        }
    }
).get("results")

# 遍历查询结果并更新属性
for result in results:
    tags = result.get(tag_property_name, [])
    if "your_tag" not in tags:
        tags.append("your_tag")
        result.set(properties={tag_property_name: tags})
