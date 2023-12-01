import asyncio
import logging
import sys
from threading import Thread
from glob import glob

import elevenlabs
import openai
from decouple import config

from input import app
from sessions.session import Session
from utils.logging import log

openai.api_key = config('OPENAI_API_KEY')
elevenlabs.set_api_key(config('ELEVENLABS_API_KEY'))


def service():
    asyncio.run(session.process())


def api():
    sys.modules['flask.cli'].show_server_banner = lambda *x: None
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    app.run(port=8008)


if __name__ == '__main__':
    with Session() as session:
        saves = glob('saves/*.sierra')
        if saves:
            if input('Load from previous session? (y/N): ').lower().startswith('y'):
                log.info(''.join([f'{i}: {save}\n' for i, save in enumerate(saves)]))
                index = int(input('Load save: '))
                session.load(saves[index])

        service_thread = Thread(target=service)
        service_thread.start()
        api_thread = Thread(target=api)
        api_thread.start()
