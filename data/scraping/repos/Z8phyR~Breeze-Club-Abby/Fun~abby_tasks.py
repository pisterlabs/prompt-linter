"""
Set Tasks For Users. Every morning at the user's {active_time} time, send a message to the user
to remind them of their tasks for the day. This generates an openai response based on the user's
tasks for the day. This is called "Abby's Morning Coffee" and is a daily task for admins/modnators.
"""
from utils.log_config import setup_logging, logging
from utils.mongo_db import connect_to_mongodb
import discord
import openai
import os
from discord.ext import commands, tasks
from dotenv import load_dotenv
import datetime
import random
import dateutil.parser as parser
import asyncio


setup_logging()
logger = logging.getLogger(__name__)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# OpenAi Call
def openai_call(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
                "content": "You are Abby, virtual bunny assistant for Breeze Club, designed to help the user with their tasks. Your response should appear as such (with a happy bunny persona and flair):"},
            {"role": "system","content": """
            👋 Greetings {user_mention} 🐰! You have an upcoming task to complete today. Your task is to: 
             `{task_description}`
            ⌚ You have until **__{task_time}__**[format time to be human-readable] to complete this task.
            💡 Here are some useful tips to help you complete your task: 
                - [tip 1]
                - [tip 2]
                - [tip 3]
             
             ❓ Please ask any questions you may have about your task by typing `hey abby` and then your question. 🐇

             Have a great day! 🐰 *hops away*
             """},    
            {"role": "assistant", "content": f"{prompt}"}
        ],
        max_tokens=500,
        temperature=0.5
    )
    return response["choices"][0]["message"]["content"]



class AbbyTasks(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.active_task = False
        self.task_channel = 1103490012500201632 # Abby's Chat
        self.start_tasks()
 
    def cog_unload(self):
        # This will cancel the task when the cog is unloaded
        self.morning_tasks.cancel()
        self.check_task.cancel()

    def start_tasks(self):
        if not self.morning_tasks.is_running():
            self.morning_tasks.start()
        if not self.check_task.is_running():
            self.check_task.start()

    def Abby_Tasks_db(self):
        client = connect_to_mongodb()
        AbbyTasks = client['Abby_Tasks']
        db = AbbyTasks['Tasks']
        return db
    
    async def send_message(self, channel, message):
        if len(message) <= 2000:
            await channel.send(message)
        else:
            chunks = [message[i: i + 1999] for i in range(0, len(message), 1999)]
            for chunk in chunks:
                await channel.send(chunk)

    @commands.Cog.listener()
    async def on_ready(self):
        logger.info(f"[🐰] Abby Tasks is ready -> ON_READY")



    @tasks.loop(hours=1) # Check every hour
    async def check_task(self):
        logger.info(f"[🐰] Checking tasks (Hourly)")
        channel = self.bot.get_channel(self.task_channel)
        # Get the current date and time
        current_datetime = datetime.datetime.now()

        # Access the database
        collection = self.Abby_Tasks_db()

        # Loop through tasks for all users
        for task in collection.find():
            user_id = task['userID']
            task_description = task['taskDescription']
            task_time = task['taskTime']

            time_until_task = task_time - current_datetime

            # Send a reminder if the task is due within the next hour
            if time_until_task <= datetime.timedelta(hours=1) and time_until_task >= datetime.timedelta():
                user_mention = f"<@{user_id}>"
                prompt = f"Upcoming task.\n\nUser: {user_mention}\nTask: {task_description}\nTime: {task_time}\n\nAbby:"
                response = openai_call(prompt)
                await self.send_message(channel, response)
            # Delete the task if it's past the due date
            elif time_until_task < datetime.timedelta():
                user_mention = f"<@{user_id}>"
                logger.info(f"[⌚] Deleting task '{task_description}' for {user_id} as it's past the due date.")
                collection.delete_one({'userID': user_id, 'taskDescription': task_description, 'taskTime': task_time})
                logger.info(f"[⌚] Task '{task_description}' has been deleted!")


    @tasks.loop(hours=24) # Check every 24 hours
    async def morning_tasks(self):
        channel = self.bot.get_channel(self.task_channel)
        logger.info(f"[🐰] Sending morning tasks")
        # Get the current date
        current_date = datetime.datetime.now().date()
        
        # Access the database
        collection = self.Abby_Tasks_db()
        if collection.count_documents({}) == 0:
            logger.info(f"[🐰] No tasks to send")
            return
        
        # Loop through users and send messages about their tasks
        for user in collection.find():
            user_id = user['userID']
            user_tasks = [task for task in collection.find({'userID': user_id}) if task['taskTime'].date() == current_date]
        # Check if the user has any tasks or if theres any value associalted with the user
        if user_tasks is None or len(user_tasks) == 0:
            logger.info(f"[🐰] No tasks for {user_id}")
            pass
        if user_tasks:
            user_mention = f"<@{user_id}>"
            tasks_info = "\n".join([f"Task {task['taskID']}: {task['taskDescription']} at {task['taskTime'].strftime('%H:%M')}" for task in user_tasks])
            prompt = f"Morning Tasks:\n\nUser: {user_mention}\nTasks: {tasks_info}\n\nAbby:"
            response = openai_call(prompt)
            await channel.send(response)
    
    @morning_tasks.before_loop
    async def before_morning_tasks(self):
        await self.bot.wait_until_ready()
        now = datetime.datetime.now()
        
        # Calculate the time for the next 7 AM
        tomorrow = now + datetime.timedelta(days=1)
        tomorrow = tomorrow.replace(hour=7, minute=0, second=0)
        
        # If it's already past 7 AM today, set the next 7 AM to tomorrow
        if now.hour >= 7:
            tomorrow = now + datetime.timedelta(days=1)
            tomorrow = tomorrow.replace(hour=7, minute=0, second=0)
        
        # Calculate the time difference between now and the next 7 AM
        time_until_7_am = (tomorrow - now).total_seconds()
        
        # Sleep until the next 7 AM
        logger.info(f"[🐰] Sleeping until {tomorrow}")
        await asyncio.sleep(time_until_7_am)

    # Create a "Tasks" Command Group
    @commands.group(name='task', invoke_without_command=True)
    async def task(self, ctx):
        await ctx.send("Please specify a subcommand. Use `!task help` for more information.")

    @task.command(name='add')
    @commands.has_permissions(administrator=True)
    async def add_task(self, ctx):
        # Check if the user is authorized to use this command
        def check(m):
            return m.author == ctx.author and m.channel == ctx.channel

        # insert the task into database
        Task_database = self.Abby_Tasks_db()

        await ctx.send('Who is the task for?')
        msg = await self.bot.wait_for('message', check=check)
        #If no mention, assume the task is for the user that sent the message
        # if message content is not a mention, assume the task is for the user that sent the message
        if not msg.mentions:
            user_mention = ctx.author.id
        else:
            user_mention = msg.content  # assuming mention in message content
        user_id = user_mention.strip("<@!>")
        await ctx.send(f'The task is for {user_mention}')

        # Find all existing task IDs for the user
        existing_task_ids = [task['taskID'] for task in Task_database.find({'userID': user_id})]
        
        # Find the lowest available task ID
        lowest_available_task_id = min(set(range(1, max(existing_task_ids) + 2)) - set(existing_task_ids))

        await ctx.send('What is the task?')
        msg = await self.bot.wait_for('message', check=check)
        task_description = msg.content


        await ctx.send('When should the task be scheduled?')
        msg = await self.bot.wait_for('message', check=check)
        task_time_str = msg.content = msg.content 
        try:
            task_time = parser.parse(task_time_str) # Convert to datetime object
        except Exception as e:
            logger.error(e)
            await ctx.send('Invalid date format. Please use a valid date and time format.')
            return

        # Fetch ID from user_mention
        user_id = user_mention.strip("<@!>")

        # Confirm the task details before inserting into database
        await ctx.send(f'Please confirm the task details:\nUser: {user_mention}\nTask: {task_description}\nTime: {task_time}\nIs this correct? (yes/no)')
        msg = await self.bot.wait_for('message', check=check)
        if msg.content.lower() == 'yes' or msg.content.lower() == 'y':
            pass
        else:
            return await ctx.send('Task canceled!')

        task = {
            'userID': user_id,
            'taskDescription': task_description,
            'taskTime': task_time,
            'taskID': lowest_available_task_id
        }


        result = Task_database.insert_one(task)
        if result.acknowledged:
            await ctx.send('Task has been successfully added.')
        else:
            await ctx.send('Error while adding task.')


    @task.command(name="list", aliases=["listtask"])
    async def list_tasks(self, ctx, *args):
        # Access the database
        collection = self.Abby_Tasks_db()      
        if len(args) == 0:
            user_id = str(ctx.author.id)
            user_tasks = collection.find({'userID': user_id})

            tasks_info = "\n".join([f"Task {task['taskID']}: {task['taskDescription']} at {task['taskTime']}" for task in user_tasks])
            if not tasks_info:
                await ctx.send("You have no tasks for today!")
            else:
                await ctx.send(f"Your tasks for today are:\n{tasks_info}")
        else:
            mentioned_user = ctx.message.mentions[0].name
            mentioned_user_id = str(ctx.message.mentions[0].id)
            user_tasks = collection.find({'userID': mentioned_user_id})

            tasks_info = "\n".join([f"Task {task['taskID']}: {task['taskDescription']} at {task['taskTime']}" for task in user_tasks])
            if not tasks_info:
                await ctx.send("This user has no tasks for today!")
            else:
                await ctx.send(f"{mentioned_user}'s tasks for today are:\n{tasks_info}")

    # Remove a task for the user tagged if no user is tagged, remove a task for the user that sent the message 
    @task.command(name="remove", aliases=["rem","delete","del"])
    async def remove_task(self, ctx, *args):
        if len(args) == 0:
            await ctx.send("Please provide the task ID and optionally tag the user.")
            return
        
        task_id = args[0]

        # If a user is tagged, use the tagged user's ID
        if len(ctx.message.mentions) > 0:
            user_id = str(ctx.message.mentions[0].id)
        else:
            user_id = str(ctx.author.id)
        
        # Access the database
        collection = self.Abby_Tasks_db()

        user_id = str(ctx.author.id)   

        # Find and remove the task based on user ID and task ID
        result = collection.delete_one({'userID': user_id, 'taskID': int(task_id)})

        if result.deleted_count == 1:
            await ctx.send("Task successfully removed.")
        else:
            await ctx.send("Error while removing task.")
    
    @task.command(name="help")
    async def help(self, ctx):
        # Create embed
        embed = discord.Embed(title="Abby's Task Manager", description="Abby's Task Manager for Discord Breeze Club", color=0x00ff00)
        embed.add_field(name="Commands", value="task add: Add a task\n task list: List all tasks for the user tagged\n task remove: Remove a task for the user tagged", inline=False)
        embed.add_field(name="Adding a task", value="Abby will ask for the user the task is for, the task description, and the time the task should be scheduled\nThe task will be scheduled for the specified time - if the time has already passed, the task will be added to the next day\nThe task will be added to the database and will be sent to the user at the specified time", inline=False)
        embed.add_field(name="How it works", value="Abby will check the database every hour to see if there are any tasks that need to be sent\nIf there are any tasks that need to be sent, Abby will notify the user an hour before the task is scheduled\nThis will be a GPT generated message\nIf the task is past due, Abby will notify the user that the task is past due and delete the task from the database", inline=False)
        await ctx.send(embed=embed)
      


async def setup(bot):
    await bot.add_cog(AbbyTasks(bot))

