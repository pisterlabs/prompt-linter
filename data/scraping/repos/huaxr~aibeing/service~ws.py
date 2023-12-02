# -*- coding: utf-8 -*-
# @Team: AIBeing
# @Author: huaxinrui@tal.com
import queue
import sys
import threading
import traceback
import uuid
import asyncio

import openai
import websockets
from websockets.exceptions import WebSocketException

from core.log import logger
from core.conf import config
from interact.handler import handler
from interact.llm.tasks.base import AIBeingBaseTask
from interact.llm.gen import regen_task
from interact.llm.hook import AIBeingHook, AIBeingHookAsync
from interact.llm.sessions import sessions
from interact.schema.chat import response
from interact.schema.protocal import protocol

class WSServer(object):
    def __init__(self, port:int):
        self.port = port
        # linux only not darwin
        self.use_async = config.llm_async
        self.handler = handler.StreamHandler()

    async def asyncall_websocket_handler(self, websocket, path):
        session_id = str(uuid.uuid4())
        while 1:
            try:
                message = await websocket.recv()
                if len(message) == 0:
                    await websocket.send(response(protocol=protocol.exception, debug="should not empty").toStr())
                    continue
                js, returnDirectly  = await self.handler.async_on_message(message)
                if returnDirectly:
                    data = js.get("content")
                    assert isinstance(data, response), "returnDirectly must be response"
                    await websocket.send(data.toStr())
                    continue

                session = js.get("session_id", session_id)
                task = sessions.get(session)
                if not task:
                    task = AIBeingBaseTask()
                    logger.info("create new session: {}".format(session))

                task = regen_task(task, js)
                sessions.put(session, task)
                template_id = js.get("template_id", -1)
                aiSay = await task.async_generate(js, hook=AIBeingHookAsync(websocket, template_id))
                await websocket.send(aiSay)

            except Exception as e:
                if isinstance(e, WebSocketException):
                    await websocket.close()
                    logger.info("session closed {}".format(session_id))
                    break
                elif isinstance(e, openai.OpenAIError):
                    excepts = "openai exception! %s" % (str(e))
                elif isinstance(e, asyncio.CancelledError):
                    excepts = "async future task exception! %s" % (str(e))
                else:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    error_stack = traceback.format_tb(exc_traceback)
                    errors = []
                    for line in error_stack:
                        errors.append(line)
                    logger.error("\n".join(errors))
                    excepts = f"internal exception! {exc_type.__name__}: {exc_value}"
                logger.error(excepts)
                await websocket.send(response(protocol=protocol.exception, debug=excepts).toStr())

    def streaming_token(self, queue: queue.Queue, websocket):
        while True:
            item = queue.get()
            if item is None:
                break
            item = item.toStr() if isinstance(item, response) else str(item)
            asyncio.run(websocket.send(item))
    async def websocket_handler(self, websocket, path):
        session_id = str(uuid.uuid4())
        token_queue = queue.Queue()
        streaming_token_thread = threading.Thread(target=self.streaming_token, args=(token_queue, websocket))
        streaming_token_thread.start()

        while 1:
            try:
                message = await websocket.recv()
                if len(message) == 0:
                    await websocket.send(response(protocol=protocol.exception, debug="should not empty").toStr())
                    continue
                js, returnDirectly = self.handler.on_message(message)

                if returnDirectly:
                    data = js.get("content")
                    assert isinstance(data, response), "returnDirectly must be response"
                    await websocket.send(data.toStr())
                    continue
                template_id = js.get("template_id", -1)

                session = js.get("session_id", session_id)
                task = sessions.get(session)
                if not task:
                    task = AIBeingBaseTask("unknown")
                task = regen_task(task, js)
                sessions.put(session, task)
                aiSay = task.generate(js, hook=AIBeingHook(token_queue, template_id))
                await websocket.send(aiSay)
            except Exception as e:
                if isinstance(e, WebSocketException):
                    await websocket.close()
                    logger.info("session closed %s" %session_id)
                    break
                else:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    error_stack = traceback.format_tb(exc_traceback)
                    errors = []
                    for line in error_stack:
                        errors.append(line)
                    logger.error("\n".join(errors))
                    excepts = f"internal exception! {exc_type.__name__}: {exc_value}"
                logger.error(excepts)
                await websocket.send(response(protocol=protocol.exception, debug=excepts).toStr())
    def exception_handler(self, loop, context):
        exception = context.get("exception")
        logger.error(f"【*_*】exception_handler: {exception}")

    def init_handler(self):
        if self.use_async:
            loop = asyncio.get_event_loop()
            loop.set_exception_handler(self.exception_handler)
            return self.asyncall_websocket_handler
        else:
            return self.websocket_handler

    def server(self):
        return websockets.serve(
            self.init_handler(),
            '0.0.0.0',
            self.port,
            ping_interval=None,
        )


def startapp(port):
    srv = WSServer(int(port) if port is not None else config.ws_port)
    start_server = srv.server()
    logger.info("websocket listening on: %d" %srv.port)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
