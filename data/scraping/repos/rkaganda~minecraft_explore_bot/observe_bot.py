from javascript import require, On, Once, AsyncTask, once, off
import math
import logging
import json

import bot_functions
import bot_tasks
import config
import db
from db import BotDB
import openai

# logger init
logger = logging.getLogger('bot')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename=config.settings['bot_log_path'], encoding='utf-8', mode='a')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

# load js
mineflayer = require('mineflayer')
Vec3 = require('vec3')
pathfinder = require('mineflayer-pathfinder')

# init bot
BOT_USERNAME = config.settings['bot_name']
bot = mineflayer.createBot({
    'host': config.settings['server_ip'],
    'port': config.settings['server_port'],
    'username': BOT_USERNAME,
    'hideErrors': False})

# add modules to bot
bot.loadPlugin(pathfinder.pathfinder)
mcData = require('minecraft-data')(bot.version)
movements = pathfinder.Movements(bot, mcData)

# init db
db.init_db()

# spawn bot
once(bot, 'spawn')
bot.chat(f'{BOT_USERNAME} spawned')
logger.info(f'{BOT_USERNAME} spawned')


# bot state
last_location = bot.entity.position
current_bot_tasks = list()


@On(bot, 'chat')
def handle_msg(_bot, sender, message, *args):
    try:
        if sender and (sender != BOT_USERNAME):
            _bot.chat(f'heard - {message}')
            logger.info(f'heard - {message}')
            if 'come' in message:
                player = bot.players[sender]
                target = player.entity
                if not target:
                    bot.chat("can't see target")
                    return
                pos = target.position
                current_bot_tasks.extend([{
                    "function": bot_functions.go_to_location,
                    "arguments": {"location": pos}
                }])
                bot_functions.go_to_location(bot, pos, 1)
            elif message == 'stop':
                off(_bot, 'chat', handle_msg)
            elif message == 'do_task':
                do_task()
            else:
                _bot.chat("processing task...")
                handle_user_request(_bot, message)
    except Exception as e:
        logger.exception("bot:chat")
        raise e


@On(bot, 'move')
def handle_move(*args):
    try:
        def euclidean_distance_3d(point1, point2):
            return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2 + (point2.z - point1.z) ** 2)

        global last_location

        move_threshold = 2
        bot_location = bot.entity.position
        if bot_location is not None:
            distance_traveled = round(abs(euclidean_distance_3d(bot_location, last_location)))
            if distance_traveled > move_threshold:
                bot_functions.observe_local_blocks(bot)
            last_location = bot.entity.position
    except Exception as e:
        logger.exception("bot:move")
        raise e


@On(bot, 'goal_reached')
def handle_goal_reached(*args):
    bot.chat(f"goal reached.")
    try:
        if len(current_bot_tasks) > 0:
            current_task = current_bot_tasks[0]  # get the current task/function
            logger.debug(f"handle_goal_reached : current_task={current_task['function'].__name__}")

            logger.debug(f"current_task['function'].__name__ = {current_task['function'].__name__}")
            logger.debug(f"bot_functions.go_to_location.__name__ = {bot_functions.go_to_location.__name__}")

            # if the current task is go_to_location
            if current_task['function'].__name__ == bot_functions.go_to_location.__name__:
                current_bot_tasks.pop(0)  # goal was reached so remove got_to_location from list
                logger.debug(f"pop len(current_bot_tasks)={len(current_bot_tasks)}")
            else:
                logger.debug("mismatch")
            do_task()

    except Exception as e:
        logger.exception("bot:goal_reached")
        raise e


@On(bot, 'diggingCompleted')
def handle_digging_completed(*args):
    logger.debug(f"handle_digging_completed start")
    bot_functions.observe_local_blocks(bot)  # update state
    try:
        if len(current_bot_tasks) > 0:
            current_task = current_bot_tasks[0]  # get the current task/function
            logger.debug(f"handle_digging_completed : current_task={current_task['function'].__name__}")
            bot.chat("digging completed.")

            # if the current task is dig_block_by_location
            if current_task['function'].__name__ == bot_functions.dig_block_by_location.__name__:
                current_bot_tasks.pop(0)  # dig_block_by_location done, remove from task list
                logger.debug(f"pop len(current_bot_tasks)={len(current_bot_tasks)}")
                do_task()  # call do task

    except Exception as e:
        logger.exception("bot:handle_digging_completed")
        raise e


@On(bot, 'diggingAborted')
def handle_digging_aborted(block):
    logger.debug(f"handle_digging_aborted start")
    try:
        if len(current_bot_tasks) > 0:
            current_task = current_bot_tasks[0]  # get the current task/function
            logger.debug(f"handle_digging_aborted : current_task={current_task}")

            # if the current task is dig_block_by_location
            if current_task['function'] == bot_functions.dig_block_by_location:
                do_task()  # call do task without removing current task to reattempt dig_block_by_location

    except Exception as e:
        logger.exception("bot:handle_digging_aborted")
        raise e


@On(bot, 'playerCollect')
def handle_player_collect(_bot, collector, collected):
    if collector.id == bot.entity.id:
        bot.chat(f"collected item {collected.name}")
    # logger.debug(f"handle_player_collect collector={collector} collected={collected}")


@On(bot, 'entitySpawn')
def handle_entity_spawn(_bot, entity):
    pass
    # logger.debug(f"handle_entity_spawn args={entity}")


@On(bot, 'itemDrop')
def handle_item_drop(_bot, entity):
    if entity.position.distanceTo(_bot.entity.position) < 2:
        logger.debug("scheduled item pickup")
        current_bot_tasks.append({
            "function": bot_functions.go_to_location,
            "arguments": {
                "location": Vec3(entity.position.x, entity.position.y, entity.position.z),
                "distance_from": 0
            }})
        if len(current_bot_tasks) == 1:  # our task is the only one
            do_task()


def do_task():
    if len(current_bot_tasks) > 0:  # if there is another task
        next_task = current_bot_tasks[0]  # get next task
        logger.debug(f"do_task : next_task['function']={next_task['function']}")
        logger.debug(f"do_task : next_task['arguments'].keys()={next_task['arguments'].keys()}")
        next_task['arguments']['bot'] = bot  # add bot to arguments
        next_task['function'](**next_task['arguments'])  # call next task function


def handle_user_request(_bot, message):
    bot_db = BotDB()

    response, response_type = openai.query_functions_functions(bot_db, message)
    if response_type == 'function_call':
        if 'arguments' in response:
            function_args = response['arguments']
            function_args = json.loads(function_args)
        else:
            function_args = {}
        logger.debug(f"handle_user_request: function={response['name']} args={function_args}")
        if 'unused' in function_args:  # remove unused param
            del function_args['unused']

        function_args['bot'] = _bot
        function_args['bot_tasks'] = current_bot_tasks
        function_call = response['name']
        method = getattr(bot_tasks, function_call, None)  # load the function
        method(**function_args)  # call the function with the arguments provided


