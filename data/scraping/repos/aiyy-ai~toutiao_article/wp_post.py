#随机生成文章写入wordpress下的分类
import requests
from requests.auth import HTTPBasicAuth
import openai
import time
import random
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import pickle
import re
from datetime import datetime
import mysql.connector

# 读取根目录下的.env文件
load_dotenv()
# AI 主题账号
CATE = "AI"
# 获取API_KEY
openai.api_key = os.getenv('API_KEY')
# 当前时间
current_datetime = datetime.now()
current_date = str( current_datetime.date() )
file_dir = './title/' + current_date + '/'
file_path = file_dir + "title.txt"
file_path_bak = file_dir + "title_bak.txt"

if not os.path.exists( file_dir ):
    os.makedirs( file_dir )

def insert_data( title, content, cate = 0, source_id = 0 ):
    # 连接到MySQL数据库
    conn = mysql.connector.connect(
        host= os.getenv('MYSQL_HOST'),        # 数据库主机地址
        user=os.getenv('MYSQL_USERNAME'),     # 数据库用户名
        password=os.getenv('MYSQL_PASSWORD'), # 数据库密码
        database=os.getenv('MYSQL_DATABASE')  # 数据库名称
    )
    # 创建游标对象
    cursor = conn.cursor()
    # 插入数据的SQL语句
    sql = "INSERT INTO oc_article (title, content, cate, source_id, created_time) VALUES (%s, %s, %s, %s, %s)"
    # 插入的数据
    data = (title, content, cate, source_id, int(time.time()) )
    # 执行插入操作
    cursor.execute(sql, data)
    # 提交事务
    conn.commit()
    # 关闭游标和连接
    cursor.close()
    conn.close()

# 从数据库获取主题
def get_mysql_data( cate = 'AI' ):
    # 连接到MySQL数据库
    conn = mysql.connector.connect(
        host= os.getenv('MYSQL_HOST_IP'),        # 数据库主机地址
        user=os.getenv('MYSQL_USERNAME'),     # 数据库用户名
        password=os.getenv('MYSQL_PASSWORD'), # 数据库密码
        database=os.getenv('MYSQL_DATABASE')  # 数据库名称
    )
    # 创建游标对象
    cursor = conn.cursor( dictionary=True )
    # 查询数据的SQL语句
    sql = "SELECT * FROM oc_title WHERE cate=%s LIMIT 1"
    params = (cate,)
    # 执行查询操作
    cursor.execute( sql, params )
    # 获取查询结果（一条数据）
    row = cursor.fetchone()
    # 关闭游标和连接
    cursor.close()
    conn.close()
    return row

# 创建文章
def post_to_wp( post_title, post_content, category_id ):
    # 设置相关参数
    wp_base_url = os.getenv('WP_BASE_URL')  # 用您的 WordPress 网站地址替换
    wp_username = os.getenv('WP_USERNAME')  # 用您的用户名替换
    wp_password = os.getenv('WP_PASSWORD')  # 用您的密码替换
    wp_api_posts_url = f"{wp_base_url}/wp-json/wp/v2/posts"

    # 准备文章数据
    post_data = {
        "title": post_title,
        "content": post_content,
        "categories": [category_id],  # 将文章分配到指定分类
        "status": "publish",  # 设置文章状态为已发布
    }

    # 发送请求，将文章写入 WordPress v6.2 文章表
    response = requests.post(
        wp_api_posts_url, auth=HTTPBasicAuth(wp_username, wp_password), json=post_data
    )
    print( response )
    if response.status_code == 201:
        print("文章已成功发布！")
    else:
        print("发布文章时发生错误。请检查您的设置和输入。")

def check_file_exists(file_path):
    return os.path.exists(file_path)

# 从文件中读取标题
def get_title_from_file():
    with open(file_path, "r", encoding="utf-8") as file:
        titles = file.readlines()
    if not titles:
        return None
    # 取出第一条标题
    title = titles.pop(0).strip()
    # 重新写入剩余标题到文件
    with open(file_path, "w", encoding="utf-8") as file:
        file.writelines( titles )
    return title

#调用ChatGPT创建标题和内容
def generate_post_title():
    if check_file_exists( file_path_bak ):
        print(f"文件 {file_path_bak} 存在")
    else:
        print(f"文件 {file_path_bak} 不存在")
        # 调用ChatGPT进行问题创建
        row = get_mysql_data( CATE )
        #question = "写5个关于AI科普文章的吸引眼球中文标题"
        question = row['title']
        print("主题：", question)
        response = openai.ChatCompletion.create(  
            model="gpt-4",   
            messages=[{"role": "user", "content": question}]  
        )
        text = response["choices"][0]["message"]["content"]
        #text = '1.12323'
        with open( file_path, "w", encoding="utf-8") as f:
            f.write ( text )
        with open( file_path_bak, "w", encoding="utf-8") as f:
            f.write ( text )
         
#调用ChatGPT创建标题和内容
def generate_post():
    #title = get_title_from_file()
    #取标题
    with open(file_path, "r", encoding="utf-8") as file:
        titles = file.readlines()
    if not titles:
        return None
    # 取出第一条标题
    title = titles.pop(0).strip()
    if title:
        print("取出的标题：", title)
        line = ''
        # 去除序号
        match = re.match(r"(\d+)\. (.*)", title)
        if match:
            line = match.group(2)
        else:
            line = title
        if line != "":
            # 调用ChatGPT进行回答
            response = openai.ChatCompletion.create(  
                model="gpt-4",   
                messages=[{"role": "user", "content": "用2000字详细描述下：" + line}]  
            )
            text = response["choices"][0]["message"]["content"]
            if text != "":
                # 重新写入剩余标题到文件
                with open(file_path, "w", encoding="utf-8") as file:
                    file.writelines( titles )
                #text = '232323'
                # 保存到数据库
                insert_data( line, text, CATE )
                print("mysql 插入成功")
                # 保存到wordpress
                post_to_wp( line, text,  150 )
                print("wordpress 插入成功")
            else:
                print("没有内容可取了")
                
    else:
        print("没有标题可取了")

# 每天只生成一次标题
generate_post_title()   
# 每天生成N篇文章
generate_post()

