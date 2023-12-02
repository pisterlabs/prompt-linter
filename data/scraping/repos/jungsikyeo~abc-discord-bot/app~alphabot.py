import datetime
import time
import pymysql
import discord
import requests
import os as operating_system
import openai
import random
import cloudscraper
import json
import pytz
import urllib3
from pytz import all_timezones, timezone
from discord.ui import Button, View
from discord.ext import commands
from discord import Embed
from Paginator2 import Paginator, PageEmojis, NavigationType, Page
from pymysql.cursors import DictCursor
from dbutils.pooled_db import PooledDB
from urllib.parse import quote, urlparse
from dotenv import load_dotenv

load_dotenv()

command_flag = operating_system.getenv("SEARCHFI_BOT_FLAG")
bot_token = operating_system.getenv("SEARCHFI_BOT_TOKEN")
mysql_ip = operating_system.getenv("MYSQL_IP")
mysql_port = operating_system.getenv("MYSQL_PORT")
mysql_id = operating_system.getenv("MYSQL_ID")
mysql_passwd = operating_system.getenv("MYSQL_PASSWD")
mysql_db = operating_system.getenv("MYSQL_DB")
bot_domain=operating_system.getenv("SEARCHFI_BOT_DOMAIN")
discord_client_id = operating_system.getenv("DISCORD_CLIENT_ID")

class UpDownView(View):
    def __init__(self, ctx, embed_message, embed, db, project_id):
        super().__init__(timeout=None)
        self.ctx = ctx
        self.embed_message = embed_message
        self.embed = embed
        self.db = db
        self.project_id = project_id
        self.regUser = f"{ctx.message.author.name}#{ctx.message.author.discriminator}"
        self.user_id = ctx.author.id
        if self.embed_message is not None:
            self.update_message()

    async def on_timeout(self):
        await self.embed_message.edit(view=None)

    def update_message(self):
        self.embed_message.edit(embed=self.embed, view=self)

    @discord.ui.button(label="UP", style=discord.ButtonStyle.green)
    async def up_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        buttonView = ButtonView(self.ctx, self.db, "")
        Queries.merge_recommend(self.db, self.project_id, self.regUser, self.user_id, "UP")
        item = Queries.select_one_project(self.db, self.project_id)
        try:
            avatar_url = await buttonView.get_member_avatar(int(item['user_id']))
        except Exception as e:
            avatar_url = "https://pbs.twimg.com/profile_images/1544400407731900416/pmyhJIAx_400x400.jpg"
        item["avatar_url"] = avatar_url
        embed = buttonView.makeEmbed(item)
        await self.embed_message.edit(embed=embed, view=self)

    @discord.ui.button(label="DOWN", style=discord.ButtonStyle.red)
    async def down_button(self, button: discord.ui.Button, interaction: discord.Interaction):
        buttonView = ButtonView(self.ctx, self.db, "")
        Queries.merge_recommend(self.db, self.project_id, self.regUser, self.user_id, "DOWN")
        item = Queries.select_one_project(self.db, self.project_id)
        try:
            avatar_url = await buttonView.get_member_avatar(int(item['user_id']))
        except Exception as e:
            avatar_url = "https://pbs.twimg.com/profile_images/1544400407731900416/pmyhJIAx_400x400.jpg"
        item["avatar_url"] = avatar_url
        embed = buttonView.makeEmbed(item)
        await self.embed_message.edit(embed=embed, view=self)

class ButtonView(discord.ui.View):
    def __init__(self, ctx, db, day):
        super().__init__()
        self.ctx = ctx
        self.db = db
        self.day = day
        self.id = ctx.message.author.id
        self.username = f"{ctx.message.author.name}#{ctx.message.author.discriminator}"
        self.desktop = ctx.message.author.desktop_status
        self.mobile = ctx.message.author.mobile_status

    async def get_member_avatar(self, user_id: int):
        member = bot.get_user(user_id)
        if member is None:
            return "https://pbs.twimg.com/profile_images/1544400407731900416/pmyhJIAx_400x400.jpg"
        else:
            return member.avatar

    def makeEmbed(self, item):
        if item['hasTime'] == "True":
            mintTime = f"<t:{int(item['unixMintDate'])}>"
        else:
            mintTime = "NoneTime"

        link_url = f"[Twitter]({item['twitterUrl']})"
        if item['discordUrl'] and item['discordUrl'] != '-':
            link_url = f"{link_url}  |  [Discord]({item['discordUrl']})"
        if item['walletCheckerUrl'] != '-':
            link_url = f"{link_url}  |  [Checker]({item['walletCheckerUrl']})"

        call_url = None
        if item['callUrl'] != '-':
            call_url = item['callUrl']

        if str(self.mobile) == "online":
            embed=discord.Embed(title=f"{item['name']}\n@{item['twitterUrl'].split('/')[-1]}", description=f"""{mintTime} | {link_url}\n> **Supply**             {item['supply']} \n> **WL Price**         {item['wlPrice']} {item['blockchain']} \n> **Public Price**   {item['pubPrice']} {item['blockchain']}\n:thumbsup: {item['goodCount']}     :thumbsdown: {item['badCount']}""", color=0x04ff00)
            if call_url:
                embed.add_field(name="SearchFi Call", value=f"{call_url}", inline=True)
            embed.set_thumbnail(url=item['twitterProfileImage'])
            embed.set_author(name=f"{item['regUser']}", icon_url=f"{item['avatar_url']}")
            embed.set_footer(text="Powered by 으노아부지#2642")
        else:
            embed=discord.Embed(title=f"{item['name']}\n@{item['twitterUrl'].split('/')[-1]}", description=f"{mintTime} | {link_url}", color=0x04ff00)
            embed.set_thumbnail(url=item['twitterProfileImage'])
            embed.set_author(name=f"{item['regUser']}", icon_url=f"{item['avatar_url']}")
            embed.add_field(name=f"""Supply       """, value=f"{item['supply']}", inline=True)
            embed.add_field(name=f"""WL Price     """, value=f"{item['wlPrice']} {item['blockchain']}", inline=True)
            embed.add_field(name=f"""Public Price """, value=f"{item['pubPrice']} {item['blockchain']}", inline=True)
            embed.add_field(name="Up", value=f":thumbsup: {item['goodCount']}", inline=True)
            embed.add_field(name="Down", value=f":thumbsdown: {item['badCount']}", inline=True)
            if call_url:
                embed.add_field(name="SearchFi Call", value=f"{call_url}", inline=True)
            embed.set_footer(text="Powered by 으노아부지#2642")
        return embed

class Database:
    def __init__(self, host, port, user, password, db):
        self.pool = PooledDB(
            creator=pymysql,
            maxconnections=5,
            mincached=2,
            host=host,
            port=int(port),
            user=user,
            password=password,
            database=db,
            charset='utf8mb4',
            cursorclass=DictCursor
        )

    def get_connection(self):
        return self.pool.connection()

class Queries:
    def select_search_projects(db, day, week):
        select_query = f"""
        SELECT  
            A.*,  
            case when mintTime24 > 12 then 'PM' else 'AM' end timeType
        FROM ( 
             SELECT
                id, 
                name, 
                ifnull(discordUrl, '-') discordUrl,  
                ifnull(twitterUrl, '-') twitterUrl,  
                ifnull(walletCheckerUrl, '-') walletCheckerUrl,  
                ifnull(callUrl, '-') callUrl,  
                ifnull(twitterProfileImage, '-') twitterProfileImage,  
                ifnull(nullif(supply, ''), '-') supply,  
                ifnull(nullif(wlPrice, ''), '-') wlPrice,  
                ifnull(nullif(pubPrice, ''), '-') pubPrice,  
                ifnull(blockchain, '-') blockchain,  
                ifnull(starCount, '0') starCount,  
                (select count(1) from recommends where projectId = AA.id and recommendType = 'UP') goodCount,  
                (select count(1) from recommends where projectId = AA.id and recommendType = 'DOWN') badCount, 
                mintDate/1000 unixMintDate,
                case when mintDate = 'TBA' then mintDate else FROM_UNIXTIME(mintDate/1000, '%Y-%m-%d') end mintDay,
                FROM_UNIXTIME(mintDate/1000, '%H:%i') mintTime24,  
                FROM_UNIXTIME(mintDate/1000, '%h:%i') mintTime12,
                regUser,
                user_id,
                hasTime  
             FROM projects AA
             WHERE 1=1 
             AND FROM_UNIXTIME(mintDate/1000, '%Y-%m-%d') = '{day}' 
             /*AND hasTime = 'True' */
             ORDER BY mintDate ASC 
        ) A 
        WHERE 1=1 
        AND case when mintTime24 > 12 then 'PM' else 'AM' end = '{week}'
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query)
                result = cursor.fetchall()
                return result

    def select_all_projects(db, today, tomorrow):
        select_query = f"""
        SELECT  
            A.*,  
            case when mintTime24 > 12 then 'PM' else 'AM' end timeType
        FROM ( 
             SELECT
                id, 
                name, 
                ifnull(discordUrl, '-') discordUrl, 
                ifnull(twitterUrl, '-') twitterUrl,  
                ifnull(walletCheckerUrl, '-') walletCheckerUrl,  
                ifnull(callUrl, '-') callUrl,  
                ifnull(twitterProfileImage, '-') twitterProfileImage,  
                ifnull(nullif(supply, ''), '-') supply,  
                ifnull(nullif(wlPrice, ''), '-') wlPrice,  
                ifnull(nullif(pubPrice, ''), '-') pubPrice,  
                ifnull(blockchain, '-') blockchain,  
                ifnull(starCount, '0') starCount,  
                (select count(1) from recommends where projectId = AA.id and recommendType = 'UP') goodCount,  
                (select count(1) from recommends where projectId = AA.id and recommendType = 'DOWN') badCount,  
                mintDate/1000 unixMintDate,
                case when mintDate = 'TBA' then mintDate else FROM_UNIXTIME(mintDate/1000, '%Y-%m-%d') end mintDay, 
                FROM_UNIXTIME(mintDate/1000, '%H:%i') mintTime24,  
                FROM_UNIXTIME(mintDate/1000, '%h:%i') mintTime12,
                regUser,
                user_id,
                hasTime  
             FROM projects AA
             WHERE 1=1 
             AND FROM_UNIXTIME(mintDate/1000, '%Y-%m-%d %H:%i') >= '{today}' 
             AND FROM_UNIXTIME(mintDate/1000, '%Y-%m-%d') <= '{tomorrow}' 
             /*AND hasTime = 'True' */
             /*AND AA.mintDate >= concat(UNIX_TIMESTAMP(now()), '000')*/
             ORDER BY mintDate ASC 
        ) A 
        WHERE 1=1 
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query)
                result = cursor.fetchall()
                return result

    def select_today_projects(db, today, tomorrow):
        select_query = f"""
        SELECT  
            A.*,  
            case when mintTime24 > 12 then 'PM' else 'AM' end timeType
        FROM ( 
             SELECT
                id, 
                name, 
                ifnull(discordUrl, '-') discordUrl, 
                ifnull(twitterUrl, '-') twitterUrl,  
                ifnull(walletCheckerUrl, '-') walletCheckerUrl,  
                ifnull(callUrl, '-') callUrl,  
                ifnull(twitterProfileImage, '-') twitterProfileImage,  
                ifnull(nullif(supply, ''), '-') supply,  
                ifnull(nullif(wlPrice, ''), '-') wlPrice,  
                ifnull(nullif(pubPrice, ''), '-') pubPrice,  
                ifnull(blockchain, '-') blockchain,  
                ifnull(starCount, '0') starCount,  
                (select count(1) from recommends where projectId = AA.id and recommendType = 'UP') goodCount,  
                (select count(1) from recommends where projectId = AA.id and recommendType = 'DOWN') badCount,
                mintDate/1000 unixMintDate,  
                case when mintDate = 'TBA' then mintDate else FROM_UNIXTIME(mintDate/1000, '%Y-%m-%d') end mintDay, 
                FROM_UNIXTIME(mintDate/1000, '%H:%i') mintTime24,  
                FROM_UNIXTIME(mintDate/1000, '%h:%i') mintTime12,
                regUser,
                user_id,
                hasTime  
             FROM projects AA
             WHERE 1=1 
             AND FROM_UNIXTIME(mintDate/1000, '%Y-%m-%d %H:%i') >= '{today}' 
             AND FROM_UNIXTIME(mintDate/1000, '%Y-%m-%d') <= '{tomorrow}' 
             /*AND hasTime = 'True' */
             AND AA.mintDate >= concat(UNIX_TIMESTAMP(now()), '000')
             ORDER BY mintDate ASC 
        ) A 
        WHERE 1=1 
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query)
                result = cursor.fetchall()
                return result

    def select_one_project(db, project_id):
        select_query = f"""
        SELECT  
            A.*,  
            case when mintTime24 > 12 then 'PM' else 'AM' end timeType
        FROM ( 
             SELECT
                id, 
                name, 
                ifnull(discordUrl, '-') discordUrl, 
                ifnull(twitterUrl, '-') twitterUrl,  
                ifnull(walletCheckerUrl, '-') walletCheckerUrl,  
                ifnull(callUrl, '-') callUrl,  
                ifnull(twitterProfileImage, '-') twitterProfileImage,  
                ifnull(nullif(supply, ''), '-') supply,  
                ifnull(nullif(wlPrice, ''), '-') wlPrice,  
                ifnull(nullif(pubPrice, ''), '-') pubPrice,  
                ifnull(blockchain, '-') blockchain,  
                ifnull(starCount, '0') starCount,  
                (select count(1) from recommends where projectId = AA.id and recommendType = 'UP') goodCount,  
                (select count(1) from recommends where projectId = AA.id and recommendType = 'DOWN') badCount,
                mintDate/1000 unixMintDate, 
                case when mintDate = 'TBA' then mintDate else FROM_UNIXTIME(mintDate/1000, '%Y-%m-%d') end mintDay, 
                FROM_UNIXTIME(mintDate/1000, '%H:%i') mintTime24,  
                FROM_UNIXTIME(mintDate/1000, '%h:%i') mintTime12,
                regUser,
                user_id,
                hasTime  
             FROM projects AA
             WHERE 1=1 
             AND id = '{project_id}'
        ) A 
        WHERE 1=1 
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query)
                result = cursor.fetchone()
                return result

    def select_search_project(db, project_name):
        select_query = f"""
        SELECT  
            A.*,  
            case when mintTime24 > 12 then 'PM' else 'AM' end timeType
        FROM ( 
             SELECT
                id, 
                name, 
                ifnull(discordUrl, '-') discordUrl, 
                ifnull(twitterUrl, '-') twitterUrl,  
                ifnull(walletCheckerUrl, '-') walletCheckerUrl,  
                ifnull(callUrl, '-') callUrl,  
                ifnull(twitterProfileImage, '-') twitterProfileImage,  
                ifnull(nullif(supply, ''), '-') supply,  
                ifnull(nullif(wlPrice, ''), '-') wlPrice,  
                ifnull(nullif(pubPrice, ''), '-') pubPrice,  
                ifnull(blockchain, '-') blockchain,  
                ifnull(starCount, '0') starCount,  
                (select count(1) from recommends where projectId = AA.id and recommendType = 'UP') goodCount,  
                (select count(1) from recommends where projectId = AA.id and recommendType = 'DOWN') badCount, 
                mintDate/1000 unixMintDate,
                case when mintDate = 'TBA' then mintDate else FROM_UNIXTIME(mintDate/1000, '%Y-%m-%d') end mintDay, 
                FROM_UNIXTIME(mintDate/1000, '%H:%i') mintTime24,  
                FROM_UNIXTIME(mintDate/1000, '%h:%i') mintTime12,
                regUser,
                user_id,
                hasTime  
             FROM projects AA
             WHERE 1=1 
             AND (
                 upper(replace(name,' ', '')) like upper(replace('%{project_name}%', ' ', '')) 
                 or upper(replace(twitterUrl,'https://twitter.com/', '')) like upper(replace('%{project_name}%', ' ', ''))
             )
        ) A 
        WHERE 1=1 
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query)
                result = cursor.fetchall()
                return result

    def select_change_date(db, date):
        select_query = f"""
        select 
           a.date_string, 
           STR_TO_DATE(a.date_string, '%Y-%m-%d') date_date 
        from ( 
          select DATE_FORMAT('{date}','%Y-%m-%d') as date_string 
        ) a 
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query)
                result = cursor.fetchone()
                return result

    def merge_recommend(db, project_id, regUser, user_id, recommend_type):
        insert_query = f"""
            insert into recommends
            (
                projectId, regUser, user_id, recommendType
            ) 
            values 
            (
                '{project_id}', '{regUser}', '{user_id}', '{recommend_type}'
            )
            ON DUPLICATE KEY UPDATE recommendType='{recommend_type}';
        """
        try:
            with db.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(insert_query)
                    conn.commit()
            return {"status":"OK"}
        except Exception as e:
            conn.rollback()
            print(e)
            return {"status": "ERROR", "msg": e}

    def select_my_up(db, user_id, today, tomorrow):
        select_query = f"""
            SELECT  
                A.*,  
                case when mintTime24 > 12 then 'PM' else 'AM' end timeType
            FROM ( 
                 SELECT
                    AA.id, 
                    name, 
                    ifnull(discordUrl, '-') discordUrl,  
                    ifnull(twitterUrl, '-') twitterUrl,     
                    ifnull(walletCheckerUrl, '-') walletCheckerUrl,  
                    ifnull(callUrl, '-') callUrl,  
                    ifnull(twitterProfileImage, '-') twitterProfileImage,  
                    ifnull(nullif(supply, ''), '-') supply,  
                    ifnull(nullif(wlPrice, ''), '-') wlPrice,  
                    ifnull(nullif(pubPrice, ''), '-') pubPrice,  
                    ifnull(blockchain, '-') blockchain,  
                    ifnull(starCount, '0') starCount,  
                    (select count(1) from recommends where projectId = AA.id and recommendType = 'UP') goodCount,  
                    (select count(1) from recommends where projectId = AA.id and recommendType = 'DOWN') badCount, 
                    mintDate/1000 unixMintDate,
                    case when mintDate = 'TBA' then mintDate else FROM_UNIXTIME(mintDate/1000, '%Y-%m-%d') end mintDay, 
                    FROM_UNIXTIME(mintDate/1000, '%Y년 %m월 %d일') mintDayKor, 
                    FROM_UNIXTIME(mintDate/1000, '%H:%i') mintTime24,  
                    FROM_UNIXTIME(mintDate/1000, '%h:%i') mintTime12,
                    AA.regUser,
                    AA.user_id,
                    AA.hasTime
                 FROM projects AA
                 INNER JOIN recommends BB ON BB.projectId = AA.id
                 WHERE 1=1 
                 AND BB.user_id = '{user_id}'
                 AND BB.recommendType = 'UP'
                 /*AND AA.mintDate >= concat(UNIX_TIMESTAMP(now()), '000')*/
                 AND FROM_UNIXTIME(mintDate/1000, '%Y-%m-%d') >= '{today}' 
                 AND FROM_UNIXTIME(mintDate/1000, '%Y-%m-%d') <= '{tomorrow}'
                 ORDER BY mintDate ASC 
            ) A 
            WHERE 1=1 
            """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query)
                result = cursor.fetchall()
                return result

    def select_ranking(db):
        select_query = f"""
        SELECT
            DENSE_RANK() OVER (ORDER BY (up_score - down_score) DESC) AS ranking,
            id,
            name,
            twitterUrl,
            discordUrl,
            walletCheckerUrl,
            case when mintDate = 'TBA' then mintDate else FROM_UNIXTIME(mintDate/1000, '%%Y-%%m-%%d %%H:%%i') end mintDate,
            mintDate/1000 unixMintDate,
            up_score,
            down_score,
            star_score
        FROM (
                 SELECT
                     c.id,
                     c.name,
                     c.mintDate,
                     c.twitterUrl,
                     c.discordUrl,
                     c.walletCheckerUrl,
                     SUM(c.up_score) AS up_score,
                     SUM(c.down_score) AS down_score,
                     MAX(c.star_score) AS star_score
                 FROM (
                          SELECT
                              a.id,
                              a.name,
                              a.mintDate,
                              a.twitterUrl,
                              a.discordUrl,
                              a.walletCheckerUrl,
                              CASE WHEN b.recommendType = 'UP' THEN 1
                                   ELSE 0
                                  END up_score,
                              CASE WHEN b.recommendType = 'DOWN' THEN 1
                                   ELSE 0
                                  END down_score,
                              CASE WHEN COALESCE(a.starCount, 0) = '' THEN 0
                                  ELSE COALESCE(a.starCount, 0)
                                END star_score
                          FROM projects a
                                   LEFT OUTER JOIN recommends b ON a.id = b.projectId
                           WHERE a.mintDate >= concat(UNIX_TIMESTAMP(now()), '000')
                      ) c
                 GROUP BY c.id, c.name, c.twitterUrl, c.discordUrl, c.walletCheckerUrl
                 having (up_score + down_score) > 0
             ) d
        ORDER BY (up_score - down_score) DESC
        LIMIT 50;
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query)
                result = cursor.fetchall()
                return result

    def select_my_ranking(db, user_id):
        select_query = f"""
        SELECT f.*
        FROM (
                 SELECT
                     DENSE_RANK() OVER (ORDER BY (up_score - down_score) DESC) AS ranking,
                     user_id,
                     id,
                     name,
                     twitterUrl,
                     discordUrl,
                     walletCheckerUrl,
                     case when mintDate = 'TBA' then mintDate else FROM_UNIXTIME(mintDate/1000, '%%Y-%%m-%%d %%H:%%i') end mintDate,
                     mintDate/1000 unixMintDate,
                     up_score,
                     down_score,
                     star_score
                 FROM (
                          SELECT
                              c.user_id,
                              c.id,
                              c.name,
                              c.mintDate,
                              c.twitterUrl,
                              c.discordUrl,
                              c.walletCheckerUrl,
                              SUM(c.up_score) AS up_score,
                              SUM(c.down_score) AS down_score,
                              MAX(c.star_score) AS star_score
                          FROM (
                                   SELECT
                                       a.id,
                                       a.name,
                                       a.mintDate,
                                       a.twitterUrl,
                                       a.discordUrl,
                                       a.walletCheckerUrl,
                                       CASE WHEN b.recommendType = 'UP' THEN 1
                                            ELSE 0
                                           END up_score,
                                       CASE WHEN b.recommendType = 'DOWN' THEN 1
                                            ELSE 0
                                           END down_score,
                                       CASE WHEN COALESCE(a.starCount, 0) = '' THEN 0
                                            ELSE COALESCE(a.starCount, 0)
                                           END star_score,
                                       a.regUser,
                                       a.user_id
                                   FROM projects a
                                            LEFT OUTER JOIN recommends b ON a.id = b.projectId
                                   WHERE a.mintDate >= concat(UNIX_TIMESTAMP(now()), '000')
                               ) c
                          GROUP BY c.id, c.name, c.twitterUrl, c.discordUrl, c.walletCheckerUrl, c.user_id
                          having (up_score + down_score) > 0
                      ) d
                 ORDER BY (up_score - down_score) DESC
                 LIMIT 50
             ) f
        WHERE user_id = %s
        ORDER BY ranking ASC
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query, (user_id,))
                result = cursor.fetchall()
                return result

    def select_my_updown(db, user_id, type):
        select_query = f"""
        SELECT f.*
        FROM (
                 SELECT
                     DENSE_RANK() OVER (ORDER BY (up_score - down_score) DESC) AS ranking,
                     user_id,
                     id,
                     name,
                     twitterUrl,
                     discordUrl,
                     walletCheckerUrl,
                     case when mintDate = 'TBA' then mintDate else FROM_UNIXTIME(mintDate/1000, '%%Y-%%m-%%d %%H:%%i') end mintDate,
                     mintDate/1000 unixMintDate,
                     up_score,
                     down_score,
                     star_score
                 FROM (
                          SELECT
                              c.user_id,
                              c.id,
                              c.name,
                              c.mintDate,
                              c.twitterUrl,
                              c.discordUrl,
                              c.walletCheckerUrl,
                              SUM(c.up_score) AS up_score,
                              SUM(c.down_score) AS down_score,
                              MAX(c.star_score) AS star_score
                          FROM (
                                   SELECT
                                       a.id,
                                       a.name,
                                       a.mintDate,
                                       a.twitterUrl,
                                       a.discordUrl,
                                       a.walletCheckerUrl,
                                       CASE WHEN b.recommendType = 'UP' THEN 1
                                            ELSE 0
                                           END up_score,
                                       CASE WHEN b.recommendType = 'DOWN' THEN 1
                                            ELSE 0
                                           END down_score,
                                       CASE WHEN COALESCE(a.starCount, 0) = '' THEN 0
                                            ELSE COALESCE(a.starCount, 0)
                                           END star_score,
                                       a.regUser,
                                       a.user_id
                                   FROM projects a
                                            LEFT OUTER JOIN recommends b ON a.id = b.projectId
                                   WHERE a.mintDate >= concat(UNIX_TIMESTAMP(now()), '000')
                               ) c
                          GROUP BY c.id, c.name, c.twitterUrl, c.discordUrl, c.walletCheckerUrl, c.user_id
                          having (up_score + down_score) > 0
                      ) d
                 ORDER BY (up_score - down_score) DESC
             ) f
            INNER JOIN recommends r ON f.id = r.projectId
        WHERE r.user_id = %s
        AND r.recommendType = %s
        ORDER BY ranking ASC
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query, (user_id, type))
                result = cursor.fetchall()
                return result

    def add_recommendation(db, project_id, reg_user, user_id, recommend_type):
        insert_query = f"""
        INSERT INTO recommends (projectId, regUser, user_id, recommendType)
        VALUES (%s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE recommendType=%s;
        """

        previous_recommendation = Queries.get_previous_recommendation(db, project_id, user_id)
        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(insert_query, (project_id, reg_user, user_id, recommend_type, recommend_type))
                conn.commit()

        return previous_recommendation

    def get_previous_recommendation(db, project_id, user_id):
        select_query = f"""
        SELECT recommendType FROM recommends WHERE projectId=%s AND user_id=%s;
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query, (project_id, user_id))
                result = cursor.fetchone()

        if result:
            return result['recommendType']
        return None

    def get_project_id_by_twitter_handle(db, twitter_handle):
        select_query = f"""
        SELECT *
        FROM projects
        WHERE twitterUrl LIKE replace(replace(%s, '@', ''), ' ', '');
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query, (f"%{twitter_handle}",))
                result = cursor.fetchone()

        if result is None:
            return None

        return result

    def update_wallet_checker_url(db, project_id, wallet_checker_url, user_id):
        update_query = "UPDATE projects SET walletCheckerUrl = %s, walletCheckerUserId = %s WHERE id = %s"

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(update_query, (wallet_checker_url, user_id, project_id))
                conn.commit()

    def update_call_url(db, project_id, call_url, user_id):
        update_query = "UPDATE projects SET callUrl = %s, callUrlUserId = %s WHERE id = %s"

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(update_query, (call_url, user_id, project_id))
                conn.commit()

    def get_tier_by_blockchain(db, blockchain):
        select_query = f"""
        SELECT imageUrl
        FROM tiers
        WHERE blockchain = case when upper(%s) = null then 'ETH' else upper(%s) end;
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query, (blockchain, blockchain,))
                result = cursor.fetchone()

        if result is None:
            return None

        return result

    def update_tier_url(db, blockchain, image_url, reg_user, user_id):
        select_query = f"""
        SELECT count(1) lock_cnt
        FROM tiers t
        WHERE blockchain = %s
        AND t.lock = 1
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query, blockchain)
                result = cursor.fetchone()

                if result['lock_cnt'] > 0:
                    return {"lock_cnt": 1}

        update_query = """
        INSERT INTO tiers (blockchain, imageUrl, regUser, user_id)
        VALUES (upper(%s), %s, %s, %s)
        ON DUPLICATE KEY UPDATE imageUrl = %s, user_id = %s
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(update_query, (blockchain, image_url, reg_user, user_id, image_url, user_id,))
                conn.commit()
        return {"lock_cnt": 0}

    def select_keyword(db, keyword):
        select_query = f"""
        SELECT *
        FROM keywords
        WHERE keyword = %s or symbol = %s
        limit 1
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query, (keyword, keyword,))
                result = cursor.fetchone()

        if result is None:
            return {"symbol": keyword, "blockchain": "ETH"}

        return result

    def update_keyword(db, blockchain, keyword, symbol, reg_user, user_id):
        update_query = """
        INSERT INTO keywords (blockchain, keyword, symbol, regUser, user_id)
        VALUES (upper(%s), %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE blockchain = upper(%s), symbol = %s, user_id = %s
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(update_query, (blockchain, keyword, symbol, reg_user, user_id, blockchain, symbol, user_id,))
                conn.commit()

    def insert_message(db, user_id, role, content):
        update_query = """
        INSERT INTO messages (user_id, role, content) VALUES (%s, %s, %s)
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(update_query, (user_id, role, content,))
                conn.commit()

    def select_message(db, user_id):
        select_query = """
        SELECT role, content, timestamp FROM messages WHERE user_id = %s ORDER BY timestamp ASC
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query, (user_id,))
                results = cursor.fetchall()

        if results is None:
            return []

        return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]} for r in results]

    def select_stats(db):
        select_query = f"""
        with main as (
            select a.user_id, a.type, a.cnt
            from (
                     select user_id, 'REG' type, count(1) cnt
                     from projects
                     where isAlphabot <> 'Y'
                     group by user_id
                     union
                     select user_id, recommendType, count(1) cnt
                     from recommends
                     group by user_id, recommendType
                     union
                     select walletCheckerUserId, 'CHECKER', count(1) cnt
                     from projects
                     where walletCheckerUserId is not null
                     group by walletCheckerUserId
                     union
                     select callUrlUserId, 'SEARCHFI_CALL', count(1) cnt
                     from projects
                     where callUrlUserId is not null
                     group by callUrlUserId
                 ) a
            where user_id is not null
        ),
        stats as (
            select
                user_id,
                ifnull((select cnt from main where user_id = m.user_id and type = 'REG'), 0) REG,
                ifnull((select cnt from main where user_id = m.user_id and type = 'CHECKER'), 0) CHECKER,
                ifnull((select cnt from main where user_id = m.user_id and type = 'SEARCHFI_CALL'), 0) SEARCHFI_CALL,
                ifnull((select cnt from main where user_id = m.user_id and type = 'UP'), 0) UP,
                ifnull((select cnt from main where user_id = m.user_id and type = 'DOWN'), 0) DOWN
            from main m
            group by user_id
        ),
        ranks as (
            select
                user_id,
                REG,
                CHECKER,
                SEARCHFI_CALL,
                UP,
                DOWN,
                ((REG * 2) + (CHECKER * 1.5) + (SEARCHFI_CALL * 1.5) + (UP * 0.1) + (DOWN * 0.1)) RANK_POINT
            FROM stats
        )
        select
            DENSE_RANK() OVER (ORDER BY RANK_POINT DESC) AS ranking,
            user_id,
            REG,
            CHECKER,
            SEARCHFI_CALL,
            UP,
            DOWN,
            RANK_POINT
        from ranks
        order by ranking
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query)
                result = cursor.fetchall()
                return result

    def select_tarots(db, user_id):
        select_query = f"""
        SELECT draw_date, card_index FROM tarots WHERE user_id = %s
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(select_query, (user_id))
                result = cursor.fetchone()
                return result

    def insert_tarots(db, user_id, current_date, frame_index):
        update_query = """
        INSERT INTO tarots (user_id, draw_date, card_index) VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE draw_date = VALUES(draw_date), card_index = VALUES(card_index)
        """

        with db.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(update_query, (user_id, current_date, frame_index))
                conn.commit()


bot = commands.Bot(command_prefix=f"{command_flag}", intents=discord.Intents.all())

db = Database(mysql_ip, mysql_port, mysql_id, mysql_passwd, mysql_db)
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@bot.event
async def on_ready():
    print("다음으로 로그인합니다: ")
    print(bot.user.name)
    print("connection was succesful")
    await bot.change_presence(status=discord.Status.online, activity=None)

@bot.command()
async def mint(ctx, *, arg="today"):
    if arg == "today":
        target_date = datetime.datetime.now()

        today = target_date
        tomorrow = target_date + datetime.timedelta(days=1)
        today_string = today.strftime("%Y-%m-%d %H:%M")
        tomorrow_string = tomorrow.strftime("%Y-%m-%d")
    else:
        try:
            target_date = datetime.datetime.strptime(arg, "%Y%m%d").date()

            today = target_date
            tomorrow = target_date + datetime.timedelta(days=1)
            today_string = today.strftime("%Y-%m-%d")
            tomorrow_string = tomorrow.strftime("%Y-%m-%d")
        except ValueError:
            await ctx.reply("```❌ Invalid date format. Please try again. (yyyymmdd)\n\n잘못된 날짜 형식입니다. 다시 시도해주세요. (yyyymmdd)```", mention_author=True)
            return

    buttonView = ButtonView(ctx, db, "")
    pages = []
    projects = Queries.select_all_projects(db, today_string, tomorrow_string) # removed the if-else statement and only use select_all_projects method
    before_mint_day = ""
    color = "-"
    for item in projects:
        try:
            avatar_url = await buttonView.get_member_avatar(int(item['user_id']))
        except Exception as e:
            avatar_url = "https://pbs.twimg.com/profile_images/1544400407731900416/pmyhJIAx_400x400.jpg"
        item["avatar_url"] = avatar_url
        embed=buttonView.makeEmbed(item)

        if before_mint_day == "":
            before_mint_day = item['mintDay']
        if before_mint_day != item['mintDay']:
            color = "+"
        cal = Page(content=f"```diff\n{color}[{item['mintDay']}]{color}```", embed=embed)
        pages.append(cal)
    if len(projects) > 0:
        paginator = Paginator(bot)
        await paginator.send(ctx.channel, pages, type=NavigationType.Buttons)
    else:
        embed=discord.Embed(title="", description="")
        embed.add_field(name="", value=f"❌ There is no mint project for today's date.\n\n❌ 오늘 날짜의 민팅 프로젝트가 없습니다.", inline=True)
        await ctx.reply(embed=embed, mention_author=True)

@bot.command()
async def msearch(ctx, *, project_name):
    buttonView = ButtonView(ctx, db, "")
    pages = []
    projects = Queries.select_search_project(db, project_name)
    before_mint_day = ""
    color = "-"
    if len(projects) > 0:
        for item in projects:
            try:
                avatar_url = await buttonView.get_member_avatar(int(item['user_id']))
            except Exception as e:
                avatar_url = "https://pbs.twimg.com/profile_images/1544400407731900416/pmyhJIAx_400x400.jpg"
            item["avatar_url"] = avatar_url
            embed=buttonView.makeEmbed(item)

            if before_mint_day == "":
                before_mint_day = item['mintDay']
            if before_mint_day != item['mintDay']:
                if color == "+":
                    color = "-"
                else:
                    color = "+"
            cal = Page(content=f"```diff\n{color}[{item['mintDay']}]{color}```", embed=embed)
            pages.append(cal)

        paginator = Paginator(bot)
        await paginator.send(ctx.channel, pages, type=NavigationType.Buttons)
    else:
        embed=discord.Embed(title="", description="")
        embed.add_field(name="", value=f"❌ No projects have been searched as `{project_name}`.\nPlease search for another word.\n\n❌ `{project_name}`(으)로 검색된 프로젝트가 없습니다.\n다른 단어를 검색하십시오.", inline=True)
        await ctx.reply(embed=embed, mention_author=True)

@bot.command()
async def mrank(ctx):
    results = Queries.select_ranking(db)

    num_pages = (len(results) + 9) // 10

    pages = []

    for page in range(num_pages):
        embed = Embed(title=f"Top {page * 10 + 1} ~ {page * 10 + 10} Rank\n", color=0x00ff00)

        for i in range(10):
            index = page * 10 + i
            if index >= len(results):
                break

            item = results[index]
            link_url = f"[Twitter]({item['twitterUrl']})"
            if item['discordUrl']:
                link_url = f"{link_url}  |  [Discord]({item['discordUrl']})"
            if item['walletCheckerUrl']:
                link_url = f"{link_url}  |  [Checker]({item['walletCheckerUrl']})"

            field_name = f"`{item['ranking']}.` {item['name']} (@{item['twitterUrl'].split('/')[-1]}) :thumbsup: {item['up_score']}  :thumbsdown: {item['down_score']}"
            if item['mintDate'] == 'TBA':
                field_value = f"{item['mintDate']}  |  {link_url}"
            else:
                field_value = f"<t:{int(item['unixMintDate'])}>  |  {link_url}"
            embed.add_field(name=field_name, value=field_value, inline=False)
            embed.set_footer(text=f"by SearchFI Bot")

        cal = Page(content=f"**🏆 Project Ranking Top 50 🏆**", embed=embed)
        pages.append(cal)

    paginator = Paginator(bot)
    await paginator.send(ctx.channel, pages, type=NavigationType.Buttons)

@bot.command()
async def mreg(ctx):
    embed = Embed(title="Warning", description="ℹ️ Please register the project with the button below.\n\nℹ️ 아래 버튼으로 프로젝트를 등록해주세요.", color=0xFFFFFF)
    embed.set_footer(text="Powered by 으노아부지#2642")
    await ctx.reply(embed=embed, mention_author=True)

    button_url = f'https://discord.com/api/oauth2/authorize?client_id={discord_client_id}&redirect_uri={quote(f"{bot_domain}/discord-callback/register")}&response_type=code&scope=identify'
    button = discord.ui.Button(style=discord.ButtonStyle.green, label="Go to Registration", url=button_url)
    view = discord.ui.View()
    view.add_item(button)
    await ctx.send(view=view)

@bot.command()
async def mmod(ctx):
    embed = Embed(title="Warning", description="ℹ️ Please correct the project with the button below.\n\nℹ️ 아래 버튼으로 프로젝트를 수정해주세요.", color=0xFFFFFF)
    embed.set_footer(text="Powered by 으노아부지#2642")
    await ctx.reply(embed=embed, mention_author=True)

    button_url = f'https://discord.com/api/oauth2/authorize?client_id={discord_client_id}&redirect_uri={quote(f"{bot_domain}/discord-callback/modify")}&response_type=code&scope=identify'
    button = discord.ui.Button(style=discord.ButtonStyle.red, label="Go to Modify", url=button_url)
    view = discord.ui.View()
    view.add_item(button)
    await ctx.send(view=view)

@bot.command()
async def mup(ctx, *, twitter_handle: str):
    regUser = f"{ctx.message.author.name}#{ctx.message.author.discriminator}"
    user_id = ctx.author.id

    project_info = Queries.get_project_id_by_twitter_handle(db, twitter_handle)

    if project_info is None:
        embed = Embed(title="Error", description=f"❌ No project found for `{twitter_handle}`.\n Click `!mreg` to register the project.\n\n❌ `{twitter_handle}`에 대한 프로젝트를 찾을 수 없습니다.\n `!mreg`를 눌러서 프로젝트를 등록해주세요.", color=0xff0000)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)

        button_url = f'https://discord.com/api/oauth2/authorize?client_id={discord_client_id}&redirect_uri={quote(f"{bot_domain}/discord-callback/register")}&response_type=code&scope=identify'
        button = discord.ui.Button(style=discord.ButtonStyle.green, label="Go to Registration", url=button_url)
        view = discord.ui.View()
        view.add_item(button)
        await ctx.send(view=view)

        return

    project_id = project_info['id']

    previous_recommendation = Queries.add_recommendation(db, project_id, regUser, user_id, "UP")

    if previous_recommendation is None:
        embed = Embed(title="Success", description=f":thumbup: Successfully recommended `{twitter_handle}` project!\n\n:thumbup: `{twitter_handle}` 프로젝트를 추천했습니다!", color=0x37E37B)
    elif previous_recommendation == "UP":
        embed = Embed(title="Warning", description=f"ℹ️ You have already recommended `{twitter_handle}` project.\n\nℹ️ 이미 `{twitter_handle}` 프로젝트를 추천하셨습니다.", color=0xffffff)
    else:
        embed = Embed(title="Changed", description=f":thumbup: Changed your downvote to an upvote for `{twitter_handle}` project!\n\n:thumbup: `{twitter_handle}` 프로젝트에 대한 비추천을 추천으로 변경했습니다!", color=0x37E37B)
    embed.set_footer(text="Powered by 으노아부지#2642")
    await ctx.reply(embed=embed, mention_author=True)

@bot.command()
async def mdown(ctx, *, twitter_handle: str):
    regUser = f"{ctx.message.author.name}#{ctx.message.author.discriminator}"
    user_id = ctx.author.id

    project_info = Queries.get_project_id_by_twitter_handle(db, twitter_handle)

    embed=discord.Embed(title="", description="")

    if project_info is None:
        embed = Embed(title="Error", description=f"❌ No project found for `{twitter_handle}`.\n Click `!mreg` to register the project.\n\n❌ `{twitter_handle}`에 대한 프로젝트를 찾을 수 없습니다.\n `!mreg`를 눌러서 프로젝트를 등록해주세요.", color=0xff0000)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)

        button_url = f'https://discord.com/api/oauth2/authorize?client_id={discord_client_id}&redirect_uri={quote(f"{bot_domain}/discord-callback/register")}&response_type=code&scope=identify'
        button = discord.ui.Button(style=discord.ButtonStyle.green, label="Go to Registration", url=button_url)
        view = discord.ui.View()
        view.add_item(button)
        await ctx.send(view=view)

        return

    project_id = project_info['id']

    previous_recommendation = Queries.add_recommendation(db, project_id, regUser, user_id, "DOWN")

    if previous_recommendation is None:
        embed = Embed(title="Success", description=f":thumbdown: Successfully downvoted `{twitter_handle}` project!\n\n:thumbdown: `{twitter_handle}` 프로젝트를 비추천했습니다!", color=0x37E37B)
    elif previous_recommendation == "DOWN":
        embed = Embed(title="Warning", description=f"ℹ️ You have already downvoted `{twitter_handle}` project.\n\nℹ️ 이미 `{twitter_handle}` 프로젝트를 비추천하셨습니다.", color=0xffffff)
    else:
        embed = Embed(title="Changed", description=f":thumbdown: Changed your upvote to a downvote for `{twitter_handle}` project!\n\n:thumbdown: `{twitter_handle}` 프로젝트에 대한 추천을 비추천으로 변경했습니다!", color=0x37E37B)
    embed.set_footer(text="Powered by 으노아부지#2642")
    await ctx.reply(embed=embed, mention_author=True)

@bot.command()
async def mylist(ctx):
    try:
        regUser = f"{ctx.message.author.name}#{ctx.message.author.discriminator}"
        user_id = ctx.author.id
        today = datetime.datetime.now().date()
        today_string = today.strftime("%Y-%m-%d")
        tomorrow = (datetime.datetime.now() + datetime.timedelta(days=1)).date()
        tomorrow_string = tomorrow.strftime("%Y-%m-%d")

        embed=discord.Embed(title=f"**Today {regUser} Mint List**", description="")

        my_up_list = Queries.select_my_up(db, user_id, today_string, tomorrow_string)
        before_date = ""
        before_time = ""
        list_massage = "\n"
        if len(my_up_list) > 0:
            for item in my_up_list:
                if len(list_massage) > 900:
                    embed.add_field(name="", value=list_massage, inline=True)
                    await ctx.send(embed=embed)
                    embed=discord.Embed(title="", description="")
                    list_massage = "\n"
                item_date = f"{item['mintDay']}"
                item_time = f"{item['mintTime24']}"
                if before_date != item_date:
                    list_massage = list_massage + f"""\n\n"""
                    before_date = item_date
                    before_time = ""
                if before_time != item_time:
                    if before_time != "":
                        list_massage = list_massage + "\n"
                    list_massage = list_massage + f"""<t:{int(item['unixMintDate'])}>\n"""
                    before_time = item_time
                list_massage = list_massage + f"""> [{item['name']}]({item['twitterUrl']})  /  Supply: {item['supply']}  / WL: {item['wlPrice']} {item['blockchain']}  /  Public: {item['pubPrice']} {item['blockchain']}\n"""
                # print(len(list_massage))
            list_massage = list_massage + ""
        else:
            # update_channel = await bot.fetch_channel(1089590412164993044)
            # mention_string = update_channel.mention
            list_massage = list_massage + f"❌ No projects have been recommend.\nPlease press `!mup @twitter_handle` for the project you want to recommend.\n\n❌ 추천한 프로젝트가 없습니다.\n추천할 프로젝트는 `!mup @twitter_handle`을 눌러주세요."
            embed=discord.Embed(title="", description="")
            embed.add_field(name="", value=list_massage, inline=True)
            await ctx.reply(embed=embed, mention_author=True)
            return
    except Exception as e:
        print("Error:", e)
        return

    embed.add_field(name="", value=list_massage, inline=True)
    await ctx.send(embed=embed)

@bot.command()
async def youlist(ctx, dc_id):
    try:
        print(dc_id[2:-1])
        user_id = dc_id[2:-1]
        user = await bot.fetch_user(user_id)
        print(user)
        if user is not None:
            print(f"이름: {user.name}")
            print(f"디스크리미네이터: {user.discriminator}")
            regUser = user.name + "#" + user.discriminator
        else:
            regUser = dc_id


        embed=discord.Embed(title=f"**Today {regUser} Mint List**", description="")

        today = datetime.datetime.now().date()
        today_string = today.strftime("%Y-%m-%d")
        tomorrow = (datetime.datetime.now() + datetime.timedelta(days=1)).date()
        tomorrow_string = tomorrow.strftime("%Y-%m-%d")

        my_up_list = Queries.select_my_up(db, user_id, today_string, tomorrow_string)
        before_date = ""
        before_time = ""
        list_massage = "\n"
        if len(my_up_list) > 0:
            for item in my_up_list:
                if len(list_massage) > 900:
                    embed.add_field(name="", value=list_massage, inline=True)
                    await ctx.send(embed=embed)
                    embed=discord.Embed(title="", description="")
                    list_massage = "\n"
                item_date = f"{item['mintDay']}"
                item_time = f"{item['mintTime24']}"
                if before_date != item_date:
                    list_massage = list_massage + f"""\n\n"""
                    before_date = item_date
                    before_time = ""
                if before_time != item_time:
                    if before_time != "":
                        list_massage = list_massage + "\n"
                    list_massage = list_massage + f"""<t:{int(item['unixMintDate'])}>\n"""
                    before_time = item_time
                list_massage = list_massage + f"""> [{item['name']}]({item['twitterUrl']})  /  Supply: {item['supply']}  / WL: {item['wlPrice']} {item['blockchain']}  /  Public: {item['pubPrice']} {item['blockchain']}\n"""
                # print(len(list_massage))
            list_massage = list_massage + ""
        else:
            list_massage = list_massage + f"❌ `{regUser}` has no recommended project.\n\n`❌ {regUser}`가 추천한 프로젝트는 없습니다."
            embed=discord.Embed(title="", description="")
            embed.add_field(name="", value=list_massage, inline=True)
            await ctx.reply(embed=embed, mention_author=True)
            return
    except Exception as e:
        print("Error:", e)
        return

    embed.add_field(name="", value=list_massage, inline=True)
    await ctx.send(embed=embed)

@bot.command()
async def myrank(ctx, *, dc_id=None):
    if dc_id == None:
        user_id = ctx.author.id
    else:
        user_id = int(dc_id[2:-1])

    user = await bot.fetch_user(user_id)

    buttonView = ButtonView(ctx, db, "")
    results = Queries.select_my_ranking(db, user_id)

    num_pages = (len(results) + 9) // 10

    pages = []

    if num_pages > 0:
        for page in range(num_pages):
            embed = Embed(title="", color=0x0061ff)

            for i in range(10):
                index = page * 10 + i
                if index >= len(results):
                    break

                item = results[index]
                link_url = f"[Twitter]({item['twitterUrl']})"
                if item['discordUrl']:
                    link_url = f"{link_url}  |  [Discord]({item['discordUrl']})"
                if item['walletCheckerUrl']:
                    link_url = f"{link_url}  |  [Checker]({item['walletCheckerUrl']})"

                field_name = f"`{item['ranking']}.` {item['name']} (@{item['twitterUrl'].split('/')[-1]}) :thumbsup: {item['up_score']}  :thumbsdown: {item['down_score']}"
                if item['mintDate'] == 'TBA':
                    field_value = f"{item['mintDate']}  |  {link_url}"
                else:
                    field_value = f"<t:{int(item['unixMintDate'])}>  |  {link_url}"
                embed.add_field(name=field_name, value=field_value, inline=False)

            try:
                avatar_url = await buttonView.get_member_avatar(user_id)
                if avatar_url == None:
                    avatar_url = "https://pbs.twimg.com/profile_images/1544400407731900416/pmyhJIAx_400x400.jpg"
            except Exception as e:
                avatar_url = "https://pbs.twimg.com/profile_images/1544400407731900416/pmyhJIAx_400x400.jpg"
            embed.set_author(name=f"{user.name}#{user.discriminator}\n Total {len(results)} Project in Top 50 rank", icon_url=f"{avatar_url}")
            embed.set_thumbnail(url=avatar_url)
            embed.set_footer(text=f"by SearchFI Bot")

            cal = Page(content=f"", embed=embed)
            pages.append(cal)
    else:
        embed = Embed(title="", color=0x0061ff)
        try:
            avatar_url = await buttonView.get_member_avatar(user_id)
            if avatar_url == None:
                avatar_url = "https://pbs.twimg.com/profile_images/1544400407731900416/pmyhJIAx_400x400.jpg"
        except Exception as e:
            avatar_url = "https://pbs.twimg.com/profile_images/1544400407731900416/pmyhJIAx_400x400.jpg"
        embed.set_author(name=f"{user.name}#{user.discriminator}\n Total {len(results)} Project in Top 50 rank", icon_url=f"{avatar_url}")
        embed.set_thumbnail(url=avatar_url)
        embed.set_footer(text=f"by SearchFI Bot")

        cal = Page(content=f"", embed=embed)
        pages.append(cal)

    paginator = Paginator(bot)
    await paginator.send(ctx.channel, pages, type=NavigationType.Buttons)

@bot.command()
async def myup(ctx, *, dc_id=None):
    if dc_id == None:
        user_id = ctx.author.id
    else:
        user_id = int(dc_id[2:-1])

    user = await bot.fetch_user(user_id)

    buttonView = ButtonView(ctx, db, "")
    results = Queries.select_my_updown(db, user_id, 'UP')

    num_pages = (len(results) + 9) // 10

    pages = []

    if num_pages > 0:
        for page in range(num_pages):
            embed = Embed(title="", color=0x0061ff)

            for i in range(10):
                index = page * 10 + i
                if index >= len(results):
                    break

                item = results[index]
                link_url = f"[Twitter]({item['twitterUrl']})"
                if item['discordUrl']:
                    link_url = f"{link_url}  |  [Discord]({item['discordUrl']})"
                if item['walletCheckerUrl']:
                    link_url = f"{link_url}  |  [Checker]({item['walletCheckerUrl']})"

                field_name = f"`{item['ranking']}.` {item['name']} (@{item['twitterUrl'].split('/')[-1]}) :thumbsup: {item['up_score']}  :thumbsdown: {item['down_score']}"
                if item['mintDate'] == 'TBA':
                    field_value = f"{item['mintDate']}  |  {link_url}"
                else:
                    field_value = f"<t:{int(item['unixMintDate'])}>  |  {link_url}"
                embed.add_field(name=field_name, value=field_value, inline=False)

            try:
                avatar_url = await buttonView.get_member_avatar(user_id)
                if avatar_url == None:
                    avatar_url = "https://pbs.twimg.com/profile_images/1544400407731900416/pmyhJIAx_400x400.jpg"
            except Exception as e:
                avatar_url = "https://pbs.twimg.com/profile_images/1544400407731900416/pmyhJIAx_400x400.jpg"
            embed.set_author(name=f"{user.name}#{user.discriminator}\n Total {len(results)} Project in Top 50 rank", icon_url=f"{avatar_url}")
            embed.set_thumbnail(url=avatar_url)
            embed.set_footer(text=f"by SearchFI Bot")

            cal = Page(content=f"", embed=embed)
            pages.append(cal)
    else:
        embed = Embed(title="", color=0x0061ff)
        try:
            avatar_url = await buttonView.get_member_avatar(user_id)
            if avatar_url == None:
                avatar_url = "https://pbs.twimg.com/profile_images/1544400407731900416/pmyhJIAx_400x400.jpg"
        except Exception as e:
            avatar_url = "https://pbs.twimg.com/profile_images/1544400407731900416/pmyhJIAx_400x400.jpg"
        embed.set_author(name=f"{user.name}#{user.discriminator}\n Total {len(results)} UP", icon_url=f"{avatar_url}")
        embed.set_thumbnail(url=avatar_url)
        embed.set_footer(text=f"by SearchFI Bot")

        cal = Page(content=f"", embed=embed)
        pages.append(cal)

    paginator = Paginator(bot)
    await paginator.send(ctx.channel, pages, type=NavigationType.Buttons)

@bot.command()
async def mydown(ctx, *, dc_id=None):
    if dc_id == None:
        user_id = ctx.author.id
    else:
        user_id = int(dc_id[2:-1])

    user = await bot.fetch_user(user_id)

    buttonView = ButtonView(ctx, db, "")
    results = Queries.select_my_updown(db, user_id, 'DOWN')

    num_pages = (len(results) + 9) // 10

    pages = []

    if num_pages > 0:
        for page in range(num_pages):
            embed = Embed(title="", color=0x0061ff)

            for i in range(10):
                index = page * 10 + i
                if index >= len(results):
                    break

                item = results[index]
                link_url = f"[Twitter]({item['twitterUrl']})"
                if item['discordUrl']:
                    link_url = f"{link_url}  |  [Discord]({item['discordUrl']})"
                if item['walletCheckerUrl']:
                    link_url = f"{link_url}  |  [Checker]({item['walletCheckerUrl']})"

                field_name = f"`{item['ranking']}.` {item['name']} (@{item['twitterUrl'].split('/')[-1]}) :thumbsup: {item['up_score']}  :thumbsdown: {item['down_score']}"
                if item['mintDate'] == 'TBA':
                    field_value = f"{item['mintDate']}  |  {link_url}"
                else:
                    field_value = f"<t:{int(item['unixMintDate'])}>  |  {link_url}"
                embed.add_field(name=field_name, value=field_value, inline=False)

            try:
                avatar_url = await buttonView.get_member_avatar(user_id)
                if avatar_url == None:
                    avatar_url = "https://pbs.twimg.com/profile_images/1544400407731900416/pmyhJIAx_400x400.jpg"
            except Exception as e:
                avatar_url = "https://pbs.twimg.com/profile_images/1544400407731900416/pmyhJIAx_400x400.jpg"
            embed.set_author(name=f"{user.name}#{user.discriminator}\n Total {len(results)} Project in Top 50 rank", icon_url=f"{avatar_url}")
            embed.set_thumbnail(url=avatar_url)
            embed.set_footer(text=f"by SearchFI Bot")

            cal = Page(content=f"", embed=embed)
            pages.append(cal)
    else:
        embed = Embed(title="", color=0x0061ff)
        try:
            avatar_url = await buttonView.get_member_avatar(user_id)
            if avatar_url == None:
                avatar_url = "https://pbs.twimg.com/profile_images/1544400407731900416/pmyhJIAx_400x400.jpg"
        except Exception as e:
            avatar_url = "https://pbs.twimg.com/profile_images/1544400407731900416/pmyhJIAx_400x400.jpg"
        embed.set_author(name=f"{user.name}#{user.discriminator}\n Total {len(results)} UP", icon_url=f"{avatar_url}")
        embed.set_thumbnail(url=avatar_url)
        embed.set_footer(text=f"by SearchFI Bot")

        cal = Page(content=f"", embed=embed)
        pages.append(cal)

    paginator = Paginator(bot)
    await paginator.send(ctx.channel, pages, type=NavigationType.Buttons)

@bot.command()
@commands.has_any_role('SF.Team', 'SF.Super', 'SF.Pioneer', 'SF.Guardian', 'SF.dev')
async def mchecker(ctx, twitter_handle: str = None, wallet_checker_url: str = None):
    if twitter_handle is None or wallet_checker_url is None:
        embed = Embed(title="Error", description="❌ Usage: `!mchecker <Twitter_Handle> <Wallet_Checker_URL>`\n\n❌ 사용방법: `!mchecker <트위터 핸들> <지갑체크 URL>`", color=0xff0000)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)
        return

    # Validate the URL
    parsed_url = urlparse(wallet_checker_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        embed = Embed(title="Error", description=f"❌ Please enter a `{wallet_checker_url}` valid URL format.\n\n❌ `{wallet_checker_url}`은 유효한 URL형식이 아닙니다.", color=0xff0000)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)
        return

    # Find the project ID using the Twitter handle
    project_info = Queries.get_project_id_by_twitter_handle(db, twitter_handle)
    project_id = project_info['id']
    wallet_checker_user_id = project_info['walletCheckerUserId']
    user_id = ctx.author.id

    if project_info is None:
        embed = Embed(title="Error", description="❌ Cannot find a project corresponding to `{twitter_handle}`.\n\n❌ `{twitter_handle}`에 해당하는 프로젝트를 찾을 수 없습니다.", color=0xff0000)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)
        return

    if wallet_checker_user_id is not None and wallet_checker_user_id != str(user_id):
        embed = Embed(title="Error", description=f"❌ The wallet check URL is already registered by <@{wallet_checker_user_id}>. Only <@{wallet_checker_user_id}> can be changed.\n\n❌ 이미 <@{wallet_checker_user_id}>의 의해 지갑 체크 URL이 등록되어 있습니다. <@{wallet_checker_user_id}>만 URL변경이 가능합니다.", color=0xff0000)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)
        return

    # Update the Wallet Checker URL
    Queries.update_wallet_checker_url(db, project_id, wallet_checker_url, user_id)

    embed = Embed(title="Success", description=f"✅ Wallet Checker URL for the `{twitter_handle}` project has been updated!\n\n✅ `{twitter_handle}` 프로젝트의 Wallet Checker URL이 업데이트되었습니다!", color=0x37e37b)
    embed.set_footer(text="Powered by 으노아부지#2642")
    await ctx.reply(embed=embed, mention_author=True)

@bot.command()
@commands.has_any_role('SF.Team', 'SF.Super', 'SF.Pioneer', 'SF.Guardian', 'SF.dev')
async def mcall(ctx, twitter_handle: str = None, call_url: str = None):
    if twitter_handle is None or call_url is None:
        embed = Embed(title="Error", description="❌ Usage: `!mcall <Twitter_Handle> <Call_Massage_Link>`\n\n❌ 사용방법: `!mcall <트위터 핸들> <Call 메시지 링크>`", color=0xff0000)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)
        return

    # Validate the URL
    nft_alpha_channels = [
        "https://discord.com/channels/961242951504261130/1059449160262234153",
        "https://discord.com/channels/961242951504261130/1059431422349291530",
        "https://discord.com/channels/961242951504261130/1059474081310838804",
        "https://discord.com/channels/961242951504261130/1059431299393265685",
    ]

    url_error = True
    for channel in nft_alpha_channels:
        if channel in call_url:
            url_error = False
            break

    if url_error:
        embed = Embed(title="Error", description=f"❌ Only messages from the channel below can be registered for Call message link. \n\n"
                                                 f"❌ Call 메시지 링크는 아래 채널의 메시지만 등록할 수 있습니다.\n\n"
                                                 f"{nft_alpha_channels[0]}\n"
                                                 f"{nft_alpha_channels[1]}\n"
                                                 f"{nft_alpha_channels[2]}\n"
                                                 f"{nft_alpha_channels[3]}\n", color=0xff0000)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)
        return

    # Find the project ID using the Twitter handle
    project_info = Queries.get_project_id_by_twitter_handle(db, twitter_handle)

    if project_info is None:
        embed = Embed(title="Error", description=f"❌ Cannot find a project corresponding to `{twitter_handle}`.\n\n❌ `{twitter_handle}`에 해당하는 프로젝트를 찾을 수 없습니다.", color=0xff0000)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)
        return

    project_id = project_info['id']
    call_user_id = project_info['callUrlUserId']
    user_id = ctx.author.id

    if call_user_id is not None and call_user_id != str(user_id):
        embed = Embed(title="Error", description=f"❌ This link is already registered by <@{call_user_id}>. Only <@{call_user_id}> can be changed.\n\n❌ 이미 <@{call_user_id}>의 의해 링크가 등록되어 있습니다. <@{call_user_id}>만 URL변경이 가능합니다.", color=0xff0000)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)
        return

    # Update the Wallet Checker URL
    Queries.update_call_url(db, project_id, call_url, user_id)

    embed = Embed(title="Success", description=f"✅ Call message link for the `{twitter_handle}` project has been updated!\n\n✅ `{twitter_handle}` 프로젝트의 Call 메시지 링크가 업데이트되었습니다!", color=0x37e37b)
    embed.set_footer(text="Powered by 으노아부지#2642")
    await ctx.reply(embed=embed, mention_author=True)

@bot.command()
@commands.has_any_role('SF.Team', 'SF.Super', 'SF.Guardian', 'SF.dev')
async def mt(ctx, blockchain: str = "ETH", tier_url: str = None):
    regUser = f"{ctx.message.author.name}#{ctx.message.author.discriminator}"
    user_id = ctx.author.id

    if tier_url:
        update_result = Queries.update_tier_url(db, blockchain, tier_url, regUser, user_id)
        if int(update_result["lock_cnt"]) > 0:
            embed = Embed(title="Error", description=f"❌ The `{blockchain}` keyword is locked and cannot be changed.\n\n❌ `{blockchain}` 키워드는 잠금 처리 되어있어 변경할 수 없습니다. ", color=0x37e37b)
            embed.set_footer(text="Powered by 으노아부지#2642")
            await ctx.reply(embed=embed, mention_author=True)
            return
        embed = Embed(title="Success", description=f"✅ `{blockchain}` has been updated!\n\n✅ `{blockchain}` 내용이 업데이트되었습니다!", color=0x37e37b)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)
    else:
        result = Queries.get_tier_by_blockchain(db, blockchain)
        await ctx.reply(f"{result['imageUrl']}", mention_author=True)

def get_current_price(token):
    url = f"https://api.bithumb.com/public/ticker/{token}_KRW"
    headers = {"accept": "application/json"}
    response = requests.get(url, headers=headers)
    data = response.json()

    if data["status"] == "0000":
        return float(data["data"]["closing_price"])
    else:
        return None

@bot.command()
async def lm(ctx, amount: float = 1):
    current_price = get_current_price('LM')
    if current_price is not None:
        current_price_rounded = round(current_price, 1)
        total_price = current_price * amount
        total_price_rounded = round(total_price, 1)

        embed = Embed(title="LM Price", color=0x3498db)
        embed.add_field(name="1 LM", value=f"```\n{format(int(str(current_price_rounded).split('.')[0]), ',')}.{str(current_price_rounded).split('.')[1]} KRW\n```", inline=True)
        embed.add_field(name=f"{amount} LM", value=f"```\n{format(int(str(total_price_rounded).split('.')[0]), ',')}.{str(total_price_rounded).split('.')[1]} KRW\n```", inline=True)
        embed.set_footer(text="Data from Bithumb", icon_url="https://content.bithumb.com/resources/img/comm/seo/favicon-96x96.png?v=bithumb.2.0.4")

        await ctx.reply(embed=embed, mention_author=True)
    else:
        embed = Embed(title="Error", description="❌ Could not fetch the price.\n\n❌ 가격을 가져올 수 없습니다.", color=0xff0000)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)

@bot.command()
async def sui(ctx, amount: float = 1):
    current_price = get_current_price('SUI')
    if current_price is not None:
        current_price_rounded = round(current_price, 1)
        total_price = current_price * amount
        total_price_rounded = round(total_price, 1)

        embed = Embed(title="SUI Price", color=0x3498db)
        embed.add_field(name="1 SUI", value=f"```\n{format(int(str(current_price_rounded).split('.')[0]), ',')}.{str(current_price_rounded).split('.')[1]} KRW\n```", inline=True)
        embed.add_field(name=f"{amount} SUI", value=f"```\n{format(int(str(total_price_rounded).split('.')[0]), ',')}.{str(total_price_rounded).split('.')[1]} KRW\n```", inline=True)
        embed.set_footer(text="Data from Bithumb", icon_url="https://content.bithumb.com/resources/img/comm/seo/favicon-96x96.png?v=bithumb.2.0.4")

        await ctx.reply(embed=embed, mention_author=True)
    else:
        embed = Embed(title="Error", description="❌ Could not fetch the price.\n\n❌ 가격을 가져올 수 없습니다.", color=0xff0000)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)

@bot.command()
async def bnb(ctx, amount: float = 1):
    current_price = get_current_price('BNB')
    if current_price is not None:
        current_price_rounded = round(current_price, 1)
        total_price = current_price * amount
        total_price_rounded = round(total_price, 1)

        embed = Embed(title="BNB Price", color=0x3498db)
        embed.add_field(name="1 BNB", value=f"```\n{format(int(str(current_price_rounded).split('.')[0]), ',')}.{str(current_price_rounded).split('.')[1]} KRW\n```", inline=True)
        embed.add_field(name=f"{amount} BNB", value=f"```\n{format(int(str(total_price_rounded).split('.')[0]), ',')}.{str(total_price_rounded).split('.')[1]} KRW\n```", inline=True)
        embed.set_footer(text="Data from Bithumb", icon_url="https://content.bithumb.com/resources/img/comm/seo/favicon-96x96.png?v=bithumb.2.0.4")

        await ctx.reply(embed=embed, mention_author=True)
    else:
        embed = Embed(title="Error", description="❌ Could not fetch the price.\n\n❌ 가격을 가져올 수 없습니다.", color=0xff0000)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)

def format_time_difference(sale_time):
    # 현재 시간과 판매 시간의 차이를 계산
    elapsed_time = datetime.datetime.now(datetime.timezone.utc) - sale_time

    # 시간 차이를 초 단위로 계산
    total_seconds = int(elapsed_time.total_seconds())

    # 시간 차이를 문자열로 포맷팅
    if total_seconds < 60:  # less than a minute
        return f"{total_seconds}초 전"
    elif total_seconds < 3600:  # less than an hour
        return f"{total_seconds // 60}분 전"
    else:  # show in hours
        return f"{total_seconds // 3600}시간 전"

def fetch_and_format_sales(activities):
    from datetime import datetime, timedelta, timezone

    index = 1
    sales = []
    for sale in activities:
        if index > 5:
            break
        try:
            name = sale['token']['meta']['name']
        except:
            name = f"Inscription #{sale['token']['inscriptionNumber']}"
        price = float(sale['listedPrice']) / 100000000
        sale_time = datetime.strptime(sale['createdAt'], "%a, %d %b %Y %H:%M:%S GMT")
        sale_time = sale_time.replace(tzinfo=timezone.utc)
        elapsed_time = datetime.now(tz=timezone.utc) - sale_time

        if elapsed_time < timedelta(minutes=1):
            time_string = f"{elapsed_time.seconds} sec ago"
        elif elapsed_time < timedelta(hours=1):
            time_string = f"{elapsed_time.seconds // 60} min ago"
        elif elapsed_time < timedelta(days=1):
            time_string = f"{elapsed_time.seconds // 3600} hrs ago"
        elif elapsed_time < timedelta(days=30):
            time_string = f"{elapsed_time.days} days ago"
        else:
            months_elapsed = elapsed_time.days // 30
            time_string = f"{months_elapsed} months ago"

        sales.append({
            "Name": name,
            "Price": price,
            "Time": time_string
        })
        index += 1
    return sales

def create_table(formatted_sales):
    output = "```\n"
    output += "{:<24s}{:<10s}{:<10s}\n".format("Name", "Price", "Time")
    output += "-"*44 + "\n"  # 24 characters + 10 characters + 10 characters

    for row in formatted_sales:
        # print(row, len(row.values()))  # 각 행과 그에 해당하는 값의 개수를 출력
        output += "{:<24s}{:<10.5f}{:<10s}\n".format(*row.values())

    output += "```"

    return output

async def me_btc(ctx, symbol):
    api_key = operating_system.getenv("MAGICEDEN_API_KEY")
    scraper = cloudscraper.create_scraper(delay=10, browser={
        'browser': 'chrome',
        'platform': 'android',
        'desktop': False,
    })
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    response = scraper.get(f"https://api-mainnet.magiceden.dev/v2/ord/btc/collections/{symbol}", headers=headers).text
    # print(response)
    data = json.loads(response)
    # print(data)

    try:
        if not data:
            embed = Embed(title="Not Found", description=f"Collection with slug `{symbol}` not found.", color=0xff0000)
            embed.set_footer(text="Powered by 으노아부지#2642")
            await ctx.reply(embed=embed, mention_author=True)
            return
    except:
        pass

    projectName = data["name"]
    projectImg = data['imageURI']
    projectChain = data['chain'].upper()
    projectTwitter = data['twitterLink']
    projectDiscord = data['discordLink']
    projectWebsite = data['websiteLink']
    projectLinks = f"[MegicEden](https://magiceden.io/ordinals/marketplace/{symbol})"
    if projectWebsite:
        projectLinks += f" | [Website]({projectWebsite})"
    if projectDiscord:
        projectLinks += f" | [Discord]({projectDiscord})"
    if projectTwitter:
        projectLinks += f" | [Twitter]({projectTwitter})"

    time.sleep(0.1)
    response = scraper.get(f"https://api-mainnet.magiceden.dev/v2/ord/btc/stat?collectionSymbol={symbol}", headers=headers).text
    # print(response)
    data = json.loads(response)

    projectFloorPrice = float(data['floorPrice']) / 100000000
    projectSupply = data['supply']
    projectOwners = data['owners']

    embed = Embed(title=f"{projectName}", color=0xbc2467, url=f"https://magiceden.io/ordinals/marketplace/{symbol}")
    embed.set_thumbnail(url=f"{projectImg}")
    embed.add_field(name=f"""Floor""", value=f"```{projectFloorPrice} {projectChain}     ```""", inline=True)
    embed.add_field(name=f"""Supply""", value=f"```{projectSupply}       ```", inline=True)
    embed.add_field(name=f"""Owners""", value=f"```{projectOwners}       ```", inline=True)

    time.sleep(0.1)
    response = scraper.get(f"https://api-mainnet.magiceden.dev/v2/ord/btc/activities?kind=buying_broadcasted&collectionSymbol={symbol}&limit=20", headers=headers).text
    data = json.loads(response)

    # 판매 데이터를 포맷팅합니다.
    formatted_sales = fetch_and_format_sales(data['activities'])

    # 포맷된 판매 데이터를 이용해 테이블을 만듭니다.
    sales_list = create_table(formatted_sales)

    embed.add_field(name="Activity Info", value=sales_list, inline=False)  # 판매 목록 추가

    embed.add_field(name=f"""Links""", value=f"{projectLinks}", inline=True)

    embed.set_footer(text="Powered by 으노아부지#2642")

    await ctx.reply(embed=embed, mention_author=True)

async def me_sol(ctx, symbol):
    api_key = operating_system.getenv("MAGICEDEN_API_KEY")
    scraper = cloudscraper.create_scraper(delay=10, browser={
        'browser': 'chrome',
        'platform': 'android',
        'desktop': False,
    })
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    response = scraper.get(f"https://api-mainnet.magiceden.dev/collections/{symbol}", headers=headers).text
    # print(response)
    data = json.loads(response)
    # print(data)

    try:
        if data['msg'] == "Invalid collection name.":
            embed = Embed(title="Not Found", description=f"Collection with slug `{symbol}` not found.", color=0xff0000)
            embed.set_footer(text="Powered by 으노아부지#2642")
            await ctx.reply(embed=embed, mention_author=True)
            return
    except:
        pass

    projectName = data["name"]
    projectImg = data['image']
    projectChain = 'SOL'
    projectTwitter = data['twitter']
    projectDiscord = data['discord']
    projectWebsite = data['website']
    projectLinks = f"[MegicEden](https://magiceden.io/ko/marketplace/{symbol})"
    if projectWebsite:
        projectLinks += f" | [Website]({projectWebsite})"
    if projectDiscord:
        projectLinks += f" | [Discord]({projectDiscord})"
    if projectTwitter:
        projectLinks += f" | [Twitter]({projectTwitter})"

    time.sleep(0.1)
    response = scraper.get(f"https://api-mainnet.magiceden.dev/v2/collections/{symbol}/stats", headers=headers).text
    # print(response)
    data = json.loads(response)

    projectFloorPrice = float(data['floorPrice']) / 1000000000

    time.sleep(0.1)
    response = scraper.get(f"https://api-mainnet.magiceden.dev/v2/collections/{symbol}/holder_stats", headers=headers).text
    # print(response)
    data = json.loads(response)

    try:
        projectSupply = data['totalSupply']
        projectOwners = data['uniqueHolders']
    except:
        projectSupply = "-"
        projectOwners = "-"

    embed = Embed(title=f"{projectName}", color=0xbc2467, url=f"https://magiceden.io/ko/marketplace/{symbol}")
    embed.set_thumbnail(url=f"{projectImg}")
    embed.add_field(name=f"""Floor""", value=f"```{projectFloorPrice} {projectChain}     ```""", inline=True)
    embed.add_field(name=f"""Supply""", value=f"```{projectSupply}       ```", inline=True)
    embed.add_field(name=f"""Owners""", value=f"```{projectOwners}       ```", inline=True)
    embed.add_field(name=f"""Links""", value=f"{projectLinks}", inline=True)
    embed.set_footer(text="Powered by 으노아부지#2642")

    await ctx.reply(embed=embed, mention_author=True)

async def me_matic(ctx, symbol):
    api_key = operating_system.getenv("MAGICEDEN_API_KEY")
    scraper = cloudscraper.create_scraper(delay=10, browser={
        'browser': 'chrome',
        'platform': 'android',
        'desktop': False,
    })
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    response = scraper.get(f"https://polygon-api.magiceden.io/v2/xc/collections/polygon/{symbol}", headers=headers).text
    data = json.loads(response)
    # print(data)

    try:
        if data['detail'] == "Collection not found":
            embed = Embed(title="Not Found", description=f"Collection with slug `{symbol}` not found.", color=0xff0000)
            embed.set_footer(text="Powered by 으노아부지#2642")
            await ctx.reply(embed=embed, mention_author=True)
            return
    except:
        pass

    projectName = data["name"]
    projectImg = data['media']
    projectChain = 'MATIC'
    projectTwitter = data['twitterLink']
    projectDiscord = data['discordLink']
    projectWebsite = data['websiteLink']
    projectLinks = f"[MegicEden](https://magiceden.io/ko/collections/polygon/{symbol})"
    if projectWebsite:
        projectLinks += f" | [Website]({projectWebsite})"
    if projectDiscord:
        projectLinks += f" | [Discord]({projectDiscord})"
    if projectTwitter:
        projectLinks += f" | [Twitter]({projectTwitter})"

    time.sleep(0.1)
    response = scraper.get(f"https://polygon-api.magiceden.io/v2/xc/collections/polygon/{symbol}/stats", headers=headers).text
    data = json.loads(response)

    projectFloorPrice = float(data['floorPrice']) / 1000000000000000000
    projectSupply = data['totalSupply']
    projectOwners = data['ownerCount']

    embed = Embed(title=f"{projectName}", color=0xbc2467, url=f"https://magiceden.io/ko/collections/polygon/{symbol}")
    embed.set_thumbnail(url=f"{projectImg}")
    embed.add_field(name=f"""Floor""", value=f"```{projectFloorPrice} {projectChain}     ```""", inline=True)
    embed.add_field(name=f"""Supply""", value=f"```{projectSupply}       ```", inline=True)
    embed.add_field(name=f"""Owners""", value=f"```{projectOwners}       ```", inline=True)
    embed.add_field(name=f"""Links""", value=f"{projectLinks}", inline=True)
    embed.set_footer(text="Powered by 으노아부지#2642")

    await ctx.reply(embed=embed, mention_author=True)

@bot.command()
async def 메(ctx, keyword):
    await me(ctx, keyword)

@bot.command()
async def me(ctx, keyword):
    result = Queries.select_keyword(db, keyword)
    print(result['blockchain'], result['symbol'])

    if result['blockchain'] == "BTC":
        await me_btc(ctx, result['symbol'])
    elif result['blockchain'] == "SOL":
        await me_sol(ctx, result['symbol'])
    elif result['blockchain'] == "MATIC":
        await me_matic(ctx, result['symbol'])

@bot.command()
async def 옾(ctx, keyword, count:int = 0):
    await os(ctx, keyword, count)

@bot.command()
async def os(ctx, keyword, count:int = 0):
    time.sleep(2)

    result = Queries.select_keyword(db, keyword)
    symbol = result['symbol']

    api_key = operating_system.getenv("OPENSEA_API_KEY")
    scraper = cloudscraper.create_scraper(delay=10, browser={
        'browser': 'chrome',
        'platform': 'android',
        'desktop': False,
    })
    headers = {"X-API-KEY": api_key}
    response = requests.get(f"https://api.opensea.io/api/v1/collection/{symbol}", headers=headers)
    results = json.loads(response.text)
    # print(results)

    try:
        if not results['success']:
            embed = Embed(title="Not Found", description=f"Collection with slug `{keyword}` not found.", color=0xff0000)
            embed.set_footer(text="Powered by 으노아부지#2642")
            await ctx.reply(embed=embed, mention_author=True)
            return
    except:
        pass

    try:
        if results['detail'] == "Request was throttled. Expected available in 1 second.":
            print(f"retry {count + 1}")
            await 옾(ctx, keyword, count + 1)
            return
    except:
        pass

    data = results['collection']

    projectName = data["name"]
    projectImg = data['image_url']
    projectTwitter = f"https://twitter.com/{data['twitter_username']}"
    projectDiscord = data['discord_url']
    projectWebsite = data['external_url']
    projectChain = result['blockchain']
    projectLinks = f"[OpenSea](https://opensea.io/collection/{symbol})"
    if projectWebsite:
        projectLinks += f" | [Website]({projectWebsite})"
    if projectDiscord:
        projectLinks += f" | [Discord]({projectDiscord})"
    if projectTwitter:
        projectLinks += f" | [Twitter]({projectTwitter})"

    projectFloorPrice = round(float(data['stats']['floor_price']),3)
    projectSupply = int(data['stats']['total_supply'])
    projectOwners = int(data['stats']['num_owners'])

    sales_list = "```\n"
    sales_list += "{:<12s}{:<13s}{:<8s}{:<9s}\n".format("Activity", "Volume", "Sales", "Average")
    sales_list += "-"*44 + "\n"  # 24 characters + 10 characters + 10 characters
    sales_list += "{:<12s}{:<13s}{:<8s}{:<9s}\n".format(
        "Last Hour",
        f"{round(float(data['stats']['one_hour_volume']),3)}",
        f"{int(data['stats']['one_hour_sales'])}",
        f"{round(float(data['stats']['one_hour_average_price']),3)} {projectChain}",
    )
    sales_list += "{:<12s}{:<13s}{:<8s}{:<9s}\n".format(
        "Last Day",
        f"{round(float(data['stats']['one_day_volume']),3)}",
        f"{int(data['stats']['one_day_sales'])}",
        f"{round(float(data['stats']['one_day_average_price']),3)} {projectChain}",
    )
    sales_list += "{:<12s}{:<13s}{:<8s}{:<9s}\n".format(
        "Last Week",
        f"{round(float(data['stats']['seven_day_volume']),3)}",
        f"{int(data['stats']['seven_day_sales'])}",
        f"{round(float(data['stats']['seven_day_average_price']),3)} {projectChain}",
    )
    sales_list += "{:<12s}{:<13s}{:<8s}{:<9s}\n".format(
        "All Time",
        f"{round(float(data['stats']['total_volume']),3)}",
        f"{int(data['stats']['total_sales'])}",
        f"{round(float(data['stats']['average_price']),3)} {projectChain}",
    )
    sales_list += "```"

    embed = Embed(title=f"{projectName}", color=0x2081E2, url=f"https://opensea.io/collection/{symbol}")
    embed.set_thumbnail(url=f"{projectImg}")
    embed.add_field(name=f"""Floor""", value=f"```{projectFloorPrice} {projectChain}     ```""", inline=True)
    embed.add_field(name=f"""Supply""", value=f"```{projectSupply}       ```", inline=True)
    embed.add_field(name=f"""Owners""", value=f"```{projectOwners}       ```", inline=True)

    embed.add_field(name="Activity Info", value=sales_list, inline=False)

    embed.add_field(name=f"""Links""", value=f"{projectLinks}", inline=True)
    embed.set_footer(text="Powered by 으노아부지#2642")

    await ctx.reply(embed=embed, mention_author=True)

@bot.command()
async def msave(ctx, blockchain, keyword, symbol):
    reg_user = f"{ctx.message.author.name}#{ctx.message.author.discriminator}"
    user_id = ctx.author.id

    Queries.update_keyword(db, blockchain, keyword, symbol, reg_user, user_id)

    embed = Embed(title="Saved", description=f"✅ Keyword `{keyword}` has been saved.\n\n✅ `{keyword}` 키워드가 저장되었습니다.", color=0x37E37B)
    embed.set_footer(text="Powered by 으노아부지#2642")
    await ctx.reply(embed=embed, mention_author=True)

timezone_mapping = {tz: tz for tz in all_timezones}
# Common abbreviations
timezone_mapping.update({
    'UTC': 'UTC',
    'EST': 'US/Eastern',
    'EDT': 'US/Eastern',
    'CST': 'US/Central',
    'CDT': 'US/Central',
    'MST': 'US/Mountain',
    'MDT': 'US/Mountain',
    'PST': 'US/Pacific',
    'PDT': 'US/Pacific',
    'HST': 'US/Hawaii',
    'AKST': 'US/Alaska',
    'AKDT': 'US/Alaska',
    'AEST': 'Australia/Eastern',
    'AEDT': 'Australia/Eastern',
    'ACST': 'Australia/Central',
    'ACDT': 'Australia/Central',
    'AWST': 'Australia/West',
    'KST': 'Asia/Seoul',
    'JST': 'Asia/Tokyo',
    'CET': 'Europe/Paris',
    'CEST': 'Europe/Paris',
    'EET': 'Europe/Bucharest',
    'EEST': 'Europe/Bucharest',
    'WET': 'Europe/Western',
    'WEST': 'Europe/Western',
    # Add more if needed
})

@bot.command()
async def mtime(ctx, date_str, time_str, from_tz_param, to_tz_str_param):
    from_tz_str = timezone_mapping.get(from_tz_param.upper())
    to_tz_str = timezone_mapping.get(to_tz_str_param.upper())

    if not from_tz_str or not to_tz_str:
        embed = Embed(title="Error", description=f"❌ Invalid timezone provided.\n\n❌ 시간대가 올바르지 않습니다.", color=0xff0000)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)
        return

    from_tz = pytz.timezone(from_tz_str)
    to_tz = pytz.timezone(to_tz_str)

    datetime_str = date_str + ' ' + time_str

    try:
        from datetime import datetime
        datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
    except ValueError:
        embed = Embed(title="Error", description="❌ Invalid datetime format. Please use `YYYY-MM-DD HH:MM`\n\n❌ 날짜형식이 올바르지 않습니다. `YYYY-MM-DD HH:MM` 형식으로 입력해주세요.", color=0xff0000)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)
        return

    datetime_obj = from_tz.localize(datetime_obj)
    datetime_in_to_tz = datetime_obj.astimezone(to_tz)

    embed = Embed(title="Date Conversion", description=f"```{datetime_str}({from_tz_param.upper()})\n\n🔄\n\n{datetime_in_to_tz.strftime('%Y-%m-%d %H:%M')}({to_tz_str_param.upper()})```", color=0xFEE501)
    embed.set_footer(text="Powered by 으노아부지#2642")
    await ctx.reply(embed=embed, mention_author=True)

@bot.command()
async def 해외주식(ctx, stock_symbol: str):
    user = f"{ctx.message.author.name}#{ctx.message.author.discriminator}"

    if not(user == "일론마스크#1576" or user == "으노아부지#2642"):
        embed = Embed(title="NO NO NO!", description="❌ Only for 일론마스크#1576\n\n❌ 오직 일론 형님만 조회 가능합니다!", color=0xff0000)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)
        return

    import matplotlib.pyplot as plt
    import mplfinance as mpf
    import pandas as pd
    from datetime import datetime
    from io import BytesIO
    from matplotlib.dates import DateFormatter

    stock_key = operating_system.getenv("STOCK_KEY")
    BASE_URL = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": stock_symbol,
        "apikey": stock_key  # replace with your own API key
    }

    response = requests.get(BASE_URL, params=params)
    data = response.json()

    if 'Time Series (Daily)' not in data:
        embed = Embed(title="Warning", description="ℹ️ Could not fetch the stock data. Please check the stock symbol. This function can be used up to 5 times every 5 minutes.\n\nℹ️ 주식 데이터를 가져올 수 없습니다. 주식 심볼을 확인해주세요. 이 기능은 5분마다 최대 5회까지 사용 가능합니다.", color=0xFFFFFF)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)
        return

    # Convert the time series data into a pandas DataFrame
    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index', dtype=float)
    df.index = pd.to_datetime(df.index)  # convert index to datetime
    df = df.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '6. volume': 'Volume'})  # rename columns
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # rearrange columns

    # Create the plot with the desired style and save it as an image file
    mc = mpf.make_marketcolors(up='g', down='r', volume='b', inherit=True)
    s  = mpf.make_mpf_style(base_mpf_style='kenan', marketcolors=mc, rc={'xtick.major.pad': 10, 'ytick.major.pad': 5})
    fig, axes = mpf.plot(df, style=s, type='candle', volume=True, title=f"{stock_symbol} Stock Chart", returnfig=True, show_nontrading=True)
    axes[0].yaxis.tick_right()
    axes[0].yaxis.set_label_position("right")
    axes[0].xaxis_date()
    axes[0].xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))  # New line to format date
    fig.tight_layout()
    fig.savefig('stock_chart.png')
    plt.close(fig)

    await ctx.reply(file=discord.File('stock_chart.png'), mention_author=True)

@bot.command()
async def coin(ctx, coin_symbol: str, period: str = "1day"):
    await 코인(ctx, coin_symbol, period)

@bot.command()
async def 코인(ctx, base_coin: str, period: str = "1day"):
    import os
    import discord
    from discord.ext import commands
    from binance.client import Client
    import pandas as pd
    import matplotlib.pyplot as plt
    import mplfinance as mpf
    from matplotlib.dates import DateFormatter
    from datetime import datetime, timedelta
    import re
    from matplotlib.ticker import FuncFormatter
    import pytz

    base_coin = base_coin.upper()
    quote_coin = 'USDT'

    symbol = base_coin + quote_coin

    if not re.match('^[A-Z0-9-_.]{1,20}$', symbol):
        embed = Embed(title="Warning", description=f"❌ '{symbol}' is not a valid coin symbol. \n\n❌ '{symbol}'은(는) 유효한 코인 심볼이 아닙니다.", color=0xFFFFFF)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)
        return

    binance_api_key = operating_system.getenv("BINANCE_API_KEY")
    binance_secret_key = operating_system.getenv("BINANCE_SECRET_KEY")
    binance_client = Client(binance_api_key, binance_secret_key)

    if period == "5min" or period == "1day":
        interval = Client.KLINE_INTERVAL_5MINUTE
    else:
        interval = Client.KLINE_INTERVAL_1DAY

    limit = 1000

    try:
        candles = binance_client.get_klines(symbol=symbol, interval=interval, limit=limit)
    except:
        embed = Embed(title="Warning", description="❌ Invalid symbol. Please check the symbol and try again.\n\n❌ 잘못된 기호입니다. 기호를 확인하고 다시 시도하십시오.", color=0xFFFFFF)
        embed.set_footer(text="Powered by 으노아부지#2642")
        await ctx.reply(embed=embed, mention_author=True)
        return

    df = pd.DataFrame(candles, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)

    df.index = df.index.to_pydatetime()
    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Seoul')

    end_date = df.index.max()
    if period is not None:
        if period == "3year":
            start_date = end_date - timedelta(days=3*365)
            period_str = "3-Year"
        elif period == "1year":
            start_date = end_date - timedelta(days=365)
            period_str = "1-Year"
        elif period == "1mon":
            start_date = end_date - timedelta(days=30)
            period_str = "1-Month"
        elif period == "3mon":
            start_date = end_date - timedelta(days=90)
            period_str = "3-Month"
        elif period == "1week":
            start_date = end_date - timedelta(days=7)
            period_str = "1-Week"
        elif period == "1day":
            start_date = end_date - timedelta(days=1)
            period_str = "1-Day (5min interval)"
        elif period == "5min":
            start_date = end_date - timedelta(minutes=120)
            period_str = "2-Hour (5min interval)"
        else:
            embed = Embed(title="Warning", description="ℹ️ Please enter a valid period: '3year', '1year', '3mon', '1mon', '1week', '1day', '5min' or leave it blank for full data.\n\nℹ️ '3year', '1year', '3mon', '1mon', '1week', '1day', '5min' 형식의 기간을 입력하거나 전체 데이터를 입력하려면 공백으로 두십시오.", color=0xFFFFFF)
            embed.set_footer(text="Powered by 으노아부지#2642")
            await ctx.reply(embed=embed, mention_author=True)
            return
    else:
        start_date = end_date - timedelta(days=90)
        period_str = "3-Monthly"

    df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    df.index = df.index.to_pydatetime()

    mc = mpf.make_marketcolors(up='g', down='r', volume='b', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc)
    fig, axes = mpf.plot(df, type='candle', style=s, volume=True, returnfig=True)

    fig.suptitle(f"{base_coin} Coin Chart", fontsize=20)

    axes[0].yaxis.tick_right()
    axes[0].yaxis.set_label_position("right")
    axes[0].xaxis_date()
    axes[0].set_ylabel('PRICE (USDT)')
    fig.tight_layout()

    fig.savefig('./static/coin_chart.png')
    plt.close(fig)

    # response = requests.get('https://api.coingecko.com/api/v3/coins/list')
    # coins = response.json()
    #
    # coin_name = next((coin['name'] for coin in coins if coin['symbol'].upper() == base_coin), base_coin)
    coin_name = f"{base_coin}/{quote_coin}"

    # Get the latest ticker information
    ticker = binance_client.get_ticker(symbol=symbol)

    # Extract the necessary information
    last_price = float(ticker['lastPrice'])
    change_24h = float(ticker['priceChange'])
    change_24h_percent = float(ticker['priceChangePercent'])
    change_prefix = '+' if change_24h > 0 else ''
    high_24h = float(ticker['highPrice'])
    low_24h = float(ticker['lowPrice'])
    volume_24h_volume = float(ticker['volume'])
    volume_24h_usdt = float(ticker['quoteVolume'])

    now_in_seconds = time.time()
    now_in_milliseconds = int(now_in_seconds * 1000)

    # Now you can use these values in your code or embed message
    embed = discord.Embed(title=f"{coin_name}", description=f"{coin_name} {period_str} Chart Based on Binance", color=0xEFB90A)
    embed.add_field(name="24h Change", value=f"```diff\n{change_prefix}{change_24h:,.2f} ({change_prefix}{change_24h_percent}%)```")
    embed.add_field(name="24h High", value=f"```{high_24h:,.2f}```")
    embed.add_field(name="24h Low", value=f"```{low_24h:,.2f}```")
    embed.add_field(name=f"24h Volume ({base_coin})", value=f"```{volume_24h_volume:,.2f}```")
    embed.add_field(name="24h Volume (USDT)", value=f"```{volume_24h_usdt:,.2f}```")
    embed.set_image(url=f"{operating_system.getenv('SEARCHFI_BOT_DOMAIN')}/static/coin_chart.png?v={now_in_milliseconds}")  # Set the image in the embed using the image URL
    embed.set_footer(text="Powered by 으노아부지#2642")
    await ctx.reply(embed=embed, mention_author=True)

@bot.command()
@commands.has_any_role('SF.Team', 'SF.Super', 'SF.Guardian', 'SF.dev')
async def addrole(ctx, sheet_name, role_name):
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials

    # 결과를 저장할 문자열을 초기화합니다.
    result_str = ""

    try:
        # 구글 시트 접근 설정
        scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/spreadsheets',
                 "https://www.googleapis.com/auth/drive.file", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name('searchfi.json', scope)
        client = gspread.authorize(creds)

        # 시트 열기
        sheet = client.open(sheet_name).sheet1
        user_list = sheet.get_all_records()

        guild = ctx.guild
        role = discord.utils.get(guild.roles, name=role_name)

        total_count = len(user_list)
        processed_count = 0

        for user_info in user_list:
            if 'discord_uid' in user_info:
                try:
                    uid = int(user_info['discord_uid'])
                except ValueError:
                    result_str += f"UID {user_info['discord_uid']}은(는) 유효한 숫자 형식이 아닙니다.\n"
                    continue

                member = guild.get_member(uid)

                if member is not None:
                    result_str += f"{member.name}#{member.discriminator} (UID: {member.id}) 님에게 {role_name} 롤을 부여했습니다.\n"
                    await member.add_roles(role)
                else:
                    result_str += f"UID {uid}의 사용자는 서버에 없습니다.\n"

            processed_count += 1

            # 500명마다 진행 상태를 업데이트합니다. 마지막 사용자도 처리합니다.
            if processed_count % 500 == 0 or processed_count == total_count:
                await ctx.send(f"진행률: {processed_count}/{total_count} ({(processed_count / total_count) * 100:.2f}%)")

        # 결과를 txt 파일로 저장합니다.
        with open('result.txt', 'w') as f:
            f.write(result_str)

        await ctx.send(file=discord.File('result.txt'))

    except Exception as e:
        print(e)
        await ctx.send(f"오류가 발생했습니다: {str(e)}")

    await ctx.send("사용자 확인을 완료했습니다.")

@bot.command()
@commands.has_any_role('SF.Team', 'SF.Guardian', 'SF.dev')
async def removerole(ctx, role_name):
    try:
        # 결과를 저장할 문자열을 초기화합니다.
        result_str = ""

        guild = ctx.guild  # 현재 채팅창의 길드를 가져옵니다.
        role = discord.utils.get(guild.roles, name=role_name)  # 특정 역할을 가져옵니다.

        if role is None:
            await ctx.send(f"{role_name} 역할은 서버에 없습니다.")
            return

        member_count = len(guild.members)
        processed_count = 0

        # 길드의 모든 멤버를 반복하면서 역할이 있는 멤버를 찾습니다.
        for member in guild.members:
            if role in member.roles:
                await member.remove_roles(role)  # 역할을 제거합니다.
                result_str += f"{member.name}#{member.discriminator} 님에게서 {role_name} 역할을 제거했습니다.\n"

            processed_count += 1

            # 5000명마다 진행 상태를 업데이트합니다. 마지막 멤버도 처리합니다.
            if processed_count % 5000 == 0 or processed_count == member_count:
                await ctx.send(f"진행률: {processed_count}/{member_count} ({(processed_count / member_count) * 100:.2f}%)")

        # 결과를 txt 파일로 저장합니다.
        with open('remove_result.txt', 'w') as f:
            f.write(result_str)

        # 파일을 메시지로 첨부하여 보냅니다.
        await ctx.send(file=discord.File('remove_result.txt'))

    except Exception as e:
        # 에러가 발생하면 그 내용을 출력하고, 에러 메시지를 반환합니다.
        print(e)
        await ctx.send(f"오류가 발생했습니다: {str(e)}")

    # 완료 메시지를 보냅니다.
    await ctx.send(f"{role_name} 역할 제거를 완료했습니다.")

@bot.command()
async def 나무(ctx):
    embed = Embed(title="SearchFi 나무위키", description="https://namu.wiki/w/SearchFi", color=0xFFFFFF)
    await ctx.reply(embed=embed, mention_author=True)

openai.organization = "org-xZ19FcsARsvTdq3flptdn56l"
openai.api_key = operating_system.getenv("OPENAI_SECRET_KEY")

@bot.command()
async def ai(ctx, count = "0", *prompts):
    await draw(ctx, count, *prompts)

@bot.command()
async def ai2(ctx):
    import uuid
    from PIL import Image

    if len(ctx.message.attachments) == 0:
        await ctx.reply("No image provided. Please attach an image.")
        return

    random_color = random.randint(0, 0xFFFFFF)

    embed = Embed(title="SearchFi AI Image Edit Bot", color=random_color)
    embed.set_footer(text="Editing images...")
    await ctx.send(embed=embed)

    # Download the image from the attachment
    attachment = ctx.message.attachments[0]
    temp_uuid = uuid.uuid4()  # Generate a random UUID for the temporary image file
    image_path = f"./{temp_uuid}.png"  # Use the UUID as the file name to prevent duplication
    await attachment.save(image_path)

    # Open the image file and convert it to 'RGBA'
    image = Image.open(image_path).convert('RGBA')
    image.save(image_path)

    # Use the image to create a new image
    try:
        with open(image_path, "rb") as image_file:
            response = openai.Image.create_variation(
                image=image_file.read(),
                n=1,
                size="1024x1024"
            )

        image_url = response['data'][0]['url']

        embed = Embed(title="Image Edit", color=random_color)
        embed.set_image(url=image_url)
        await ctx.reply(embed=embed, mention_author=True)

    finally:
        # Remove the temporary image file after the new image has been created
        if operating_system.path.exists(image_path):
            operating_system.remove(image_path)

def imageToString(img):
    import io
    import base64

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    my_encoded_img = base64.encodebytes(img_byte_arr.getvalue()).decode('ascii')
    return my_encoded_img

@bot.command()
async def draw(ctx, count = "0", *prompts):
    import urllib
    from PIL import Image

    random_color = random.randint(0, 0xFFFFFF)

    try:
        count = int(count)
    except:
        error_embed = Embed(title="Error", description="Enter 1 to 4 images to create.\n\n생성할 이미지 개수를 1~4까지 입력하세요.", color=0xFF0000)
        await ctx.reply(embed=error_embed, mention_author=True)
        return

    if count == 0 or count > 4:
        error_embed = Embed(title="Error", description="Enter 1 to 4 images to create.\n\n생성할 이미지 개수를 1~4까지 입력하세요.", color=0xFF0000)
        await ctx.reply(embed=error_embed, mention_author=True)
        return

    if len(prompts) == 0:
        error_embed = Embed(title="Error", description="No prompt provided. Please provide a prompt.\n\n프롬프트가 입력되지 않습니다. 프롬프트를 입력하십시오.", color=0xFF0000)
        await ctx.reply(embed=error_embed, mention_author=True)
        return

    embed = Embed(title="SearchFi AI Image Gen Bot", color=random_color)
    embed.set_footer(text="Generating images...")
    await ctx.send(embed=embed)

    prompt_text = " ".join(prompts)
    model = "gpt-3.5-turbo"

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant who is good at translating."
        },
        {
            "role": "user",
            "content": f"```{prompt_text}```\n\nPlease translate the above sentence directly into English.\n\nIf the sentence is English, please print it out as it is."
        }
    ]

    # ChatGPT API 호출하기
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )
    answer = response['choices'][0]['message']['content']
    print(answer)

    try:
        # 프롬프트에 사용할 제시어
        prompt = answer
        negative_prompt = ""
        seeds = []
        seed = random.randint(0, 4294967291)

        for index in range(count):
            seeds.append(seed + index)

        # [내 애플리케이션] > [앱 키] 에서 확인한 REST API 키 값 입력
        REST_API_KEY = operating_system.getenv("KARLO_API_KEY")

        r = requests.post(
            'https://api.kakaobrain.com/v2/inference/karlo/t2i',
            json = {
                'prompt': prompt,
                'width': 512,
                'height': 512,
                'samples': count,
                'image_quality': 70,
                'guidance_scale': 12.5,
                'num_inference_steps': 20,
                'seed': seeds
            },
            headers = {
                'Authorization': f'KakaoAK {REST_API_KEY}',
                'Content-Type': 'application/json'
            }
        )
        # 응답 JSON 형식으로 변환
        response = json.loads(r.content)

        img_arr = []

        for i in range(count):
            img = Image.open(urllib.request.urlopen(response.get("images")[i].get("image")))
            img_base64 = imageToString(img)
            img_arr.append(img_base64)

        r = requests.post(
            'https://api.kakaobrain.com/v2/inference/karlo/upscale',
            json = {
                'images': img_arr,
                'scale': 2,
                'image_quality': 100
            },
            headers = {
                'Authorization': f'KakaoAK {REST_API_KEY}',
                'Content-Type': 'application/json'
            }
        )
        # 응답 JSON 형식으로 변환
        response = json.loads(r.content)
        # print(response)

        # 응답의 첫 번째 이미지 생성 결과 출력하기
        image_urls = [img for img in response.get("images")]
        # image_urls = [img["image"] for img in response.get("images")]
    except Exception as e:
        print(str(e))
        error_embed = Embed(title="Error", description="An unexpected error occurred.\n\n예기치 않은 오류가 발생했습니다.", color=0xFF0000)
        await ctx.reply(embed=error_embed, mention_author=True)
        return

    index = 0
    for image_url in image_urls:
        index += 1
        embed = Embed(title=f"Image {index}", color=random_color)
        embed.set_image(url=image_url)
        embed.set_footer(text=f"Image {index} generation complete")
        await ctx.send(embed=embed)

    embed = Embed(title="All Image generation complete", color=random_color)
    await ctx.reply(embed=embed, mention_author=True)

@bot.command()
async def 챗(ctx, *prompts):
    await gpt(ctx, *prompts)

@bot.command()
async def gpt(ctx, *prompts):
    user_id = ctx.message.author.id

    if len(prompts) == 0:
        error_embed = Embed(title="Error", description="No prompt provided. Please provide a prompt.\n\n프롬프트가 입력되지 않습니다. 프롬프트를 입력하십시오.", color=0xFF0000)
        await ctx.reply(embed=error_embed, mention_author=True)
        return

    random_color = random.randint(0, 0xFFFFFF)

    embed = Embed(title="SearchFi AI Chat Bot", color=random_color)
    embed.set_footer(text="Waiting for an answer...")
    await ctx.send(embed=embed)

    prompt_text = " ".join(prompts)

    # Load previous context for the current user
    previous_context = Queries.select_message(db, user_id)

    # If the user has sent messages before
    if previous_context:
        # Get the timestamp of the last message
        last_message_time = previous_context[-1]['timestamp']

        # Check if the user is sending a query within 5 seconds
        if datetime.datetime.now() - last_message_time < datetime.timedelta(seconds=10):
            error_embed = Embed(title="Error", description="You are sending queries too fast. Please wait a few seconds.\n\n질문을 너무 빠르게 보내고 있습니다. 몇 초 기다려 주세요.", color=0xFF0000)
            await ctx.reply(embed=error_embed, mention_author=True)
            return

    try:
        messages_with_timestamps = previous_context
        messages_for_openai = [{"role": m["role"], "content": m["content"]} for m in messages_with_timestamps]

        messages = [
                       { "role": "system", "content": "You are a helpful assistant in SearchFi Community." },
                   ] \
                   + [
                       {"role": "user", "content": "서치파이는 NFT DAO 커뮤니티입니다.\n\n프로젝트 탐색 및 연구를 기반으로 생태계를 확장하는 것이 목표입니다.\n\n디스코드 내에서 서비스를 운영하고 있으며 한국어, 영어, 일본어, 중국어 채널이 따로 있을 만큼 해외 이용자 수가 많습니다.\n\n팀원은 12명으로 CEO는 이정진이며, 그의 트위터는 @eth_apple 입니다."}
                   ] \
                   + [
                       {"role": "user", "content": "SearchFi is an NFT DAO community.\n\nThe goal is to expand the ecosystem based on project exploration and research.\n\nWe operate the service within Discord and have a large number of overseas users, with separate Korean, English, Japanese, and Chinese channels.\n\nThere are 12 team members, CEO Lee Jung-jin, and his Twitter account is @eth_apple."}
                   ] \
                   + messages_for_openai \
                   + [
                       { "role": "user", "content": f"{prompt_text}\n\nAnswers up to 600 characters."},
                   ]

        min = 3
        max = len(messages)
        if max > 0:
            while min < max:
                # print(min, max)
                if len(str(messages[0:2] + messages[min:max])) < 4097:
                    messages = messages[0:2] + messages[min:max]
                    break
                min += 1

        # print(messages)
        # print(len(str(messages)))

        result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
    except Exception as e:
        print(e)
        error_embed = Embed(title="Error", description="Failed to get a response from AI.\n\nAI로부터 응답을 받지 못했습니다.", color=0xFF0000)
        await ctx.reply(embed=error_embed, mention_author=True)
        return

    if 'choices' in result and len(result['choices']) > 0:
        assistant_response = result['choices'][0]['message']['content']
        embed = Embed(title="SearchFi AI Answer", description=assistant_response, color=random_color)
        await ctx.reply(embed=embed, mention_author=True)

        # Save user's message to the DB
        Queries.insert_message(db, user_id, "user", prompt_text)

        # Save AI's message to the DB
        Queries.insert_message(db, user_id, "assistant", assistant_response)
    else:
        error_embed = Embed(title="Error", description="Failed to get a response from AI.\n\nAI로부터 응답을 받지 못했습니다.", color=0xFF0000)
        await ctx.reply(embed=error_embed, mention_author=True)

@bot.command()
async def mstats(ctx):
    results = Queries.select_stats(db)

    num_pages = (len(results) + 9) // 10

    pages = []

    for page in range(num_pages):
        description = "```\n📅 : Project REG Count (2 Point)\n"
        description += "✅ : Project CHECKER Count (1.5 Point)\n"
        description += "📢 : Project Call Count (1.5 Point)\n"
        description += "👍 : Project UP Count (0.1 Point)\n"
        description += "👎 : Project DOWN Count (0.1 Point)\n\n```"

        embed = Embed(title=f"Top {page * 10 + 1} ~ {page * 10 + 10} Rank\n", description=f"{description}", color=0x00ff00)

        field_value = "```\n"

        for i in range(10):
            index = page * 10 + i
            if index >= len(results):
                break

            item = results[index]
            user = bot.get_user(int(item['user_id']))
            field_value += "{:>4s}{:<6s}{:<6s}{:<6s}{:<6s}{:<6s}{:<20s}\n".format(
                f"{item['ranking']}. ",
                f"📅 {item['REG']}",
                f"✅ {item['CHECKER']}",
                f"📢 {item['SEARCHFI_CALL']}",
                f"👍 {item['UP']}",
                f"👎 {item['DOWN']}",
                f"@{user}",
            )

        field_value += "```"
        embed.add_field(name="", value=field_value, inline=False)
        embed.set_footer(text=f"by SearchFI Bot")

        cal = Page(content=f"**🏆 Project REG / CHECKER / CALL / UP / DOWN Ranking 🏆**", embed=embed)
        pages.append(cal)

    paginator = Paginator(bot)
    await paginator.send(ctx.channel, pages, type=NavigationType.Buttons)

@bot.command()
async def 타로(ctx):
    await tarot(ctx)
def get_card_frame(index):
    from PIL import Image, ImageSequence

    filepath = "tarot-cards-slide-show.gif"
    img = Image.open(filepath)
    if img.is_animated:
        frames = [frame.copy() for frame in ImageSequence.Iterator(img)]
        return frames[index]
    return None
@bot.command()
async def tarot(ctx):
    import datetime

    user_id = ctx.message.author.id
    regUser = f"{ctx.message.author.name}#{ctx.message.author.discriminator}"
    current_date = datetime.date.today()

    now_in_seconds = time.time()
    now_in_milliseconds = int(now_in_seconds * 1000)

    result = Queries.select_tarots(db, user_id)

    if result and current_date <= result['draw_date']:
        keyword = Queries.select_keyword(db, f"tarot{result['card_index']}")

        # If the user has drawn today, just send the previous draw
        filename = f"{result['card_index']}.jpg"

        embed = discord.Embed(title=f"{regUser} Today`s Tarot", description=f"{keyword['symbol']}", color=random.randint(0, 0xFFFFFF))
        embed.set_image(url=f"{operating_system.getenv('SEARCHFI_BOT_DOMAIN')}/static/{filename}?v={now_in_milliseconds}")  # Set the image in the embed using the image URL
        await ctx.reply(embed=embed, mention_author=True)
    else:
        # Else, make a new draw
        random_color = random.randint(0, 0xFFFFFF)
        frame_index = random.randint(0,77)
        filename = f"{frame_index}.jpg"

        keyword = Queries.select_keyword(db, f"tarot{frame_index}")

        embed = discord.Embed(title=f"{regUser} Today`s Tarot", description=f"{keyword['symbol']}", color=random_color)
        embed.set_image(url=f"{operating_system.getenv('SEARCHFI_BOT_DOMAIN')}/static/{filename}?v={now_in_milliseconds}")  # Set the image in the embed using the image URL

        Queries.insert_tarots(db, user_id, current_date, frame_index)

        await ctx.reply(embed=embed, mention_author=True)

@bot.command()
async def mp(ctx, symbol: str, amount: float):
    ex_api_key = operating_system.getenv("EXCHANGERATE_API_KEY")
    binance_api_url = "https://api.binance.com/api/v3/ticker/price"
    exchange_rate_api_url = f"https://v6.exchangerate-api.com/v6/{ex_api_key}/latest/USD"

    # Get coin price in USD from Binance API
    response = requests.get(binance_api_url, params={"symbol": symbol.upper() + "USDT"})
    if response.status_code != 200:
        await ctx.send("Invalid coin symbol.")
        return
    coin_price_in_usd = float(response.json()['price'])

    # Get exchange rates
    response = requests.get(exchange_rate_api_url)
    if response.status_code != 200:
        await ctx.send("Error getting exchange rates.")
        return
    exchange_rates = response.json()['conversion_rates']

    # Convert amount to different currencies
    usd_amount = coin_price_in_usd * amount
    result = {
        "USD": usd_amount,
        "KRW": usd_amount * exchange_rates['KRW'],
        "CNY": usd_amount * exchange_rates['CNY'],
        "JPY": usd_amount * exchange_rates['JPY']
    }

    embed = discord.Embed(title=f"{amount} {symbol.upper()} is equal to:", color=0xEFB90A)

    embed.add_field(name="🇺🇸 USA", value="```{:,.2f} USD```".format(result['USD']), inline=False)
    embed.add_field(name="🇰🇷 SOUTH KOREA", value="```{:,.2f} KRW```".format(result['KRW']), inline=False)
    embed.add_field(name="🇨🇳 CHINA", value="```{:,.2f} CNY```".format(result['CNY']), inline=False)
    embed.add_field(name="🇯🇵 JAPAN", value="```{:,.2f} JPY```".format(result['JPY']), inline=False)

    await ctx.send(embed=embed)


bot.run(bot_token)

