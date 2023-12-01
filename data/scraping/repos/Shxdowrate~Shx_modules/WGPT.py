# —————————————————————————————————————————————————————————————————————————————————
#   █▀ █ █ ▀▄▀ █▀▄ █▀█ █ █ █ █▀█ ▄▀█ ▀█▀ █▀▀  Channel: https://t.me/mescr_m
#   ▄█ █▀█ █ █ █▄▀ █▄█ ▀▄▀▄▀ █▀▄ █▀█  █  ██▄  Not licensed
# —————————————————————————————————————————————————————————————————————————————————                         
#   █▀▀ ▀▄▀ █▀▀ █   █ █ █▀ █ █ █ █▀▀   █▀▄▀█ █▀█ █▀▄ █ █ █   █▀▀
#   ██▄ █ █ █▄▄ █▄▄ █▄█ ▄█ █ ▀▄▀ ██▄   █ ▀ █ █▄█ █▄▀ █▄█ █▄▄ ██▄
# —————————————————————————————————————————————————————————————————————————————————
# idea: 
# meta developer: @mescr_m
# requires: openai
# thanks: 
# —————————————————————————————————————————————————————————————————————————————————

import openai
from .. import loader, utils

__version__ = (1, 0, 0)

class WGPT(loader.Module):
    """Модуль для получения информации от искуственного интелекта\nРаботает без комманд, попробуйте: // привет\nDeveloper: @mescr_m"""
    strings = {
        "name": "WGPT",
        'api_key': 'API ключ для работы модуля.'
    }
    def __init__(self):
        self.config = loader.ModuleConfig(
            loader.ConfigValue(
                "api_key",None,
                lambda: self.strings['api_key'],
                validator=loader.validators.Hidden(),
            ),
        )

    @loader.watcher(out = True)
    async def watcher(self, message):
        messages = message.raw_text
        prefix = utils.escape_html(self.get_prefix())
        messaged = messages.split(' ')
        ttt = messaged[0]
        prefixx = self.db.get(self.name, 'prefix')
        mono = self.db.get(self.name, 'mono')
        
        if ttt == prefixx:
            if self.config['api_key'] == None:
                await message.reply(f'У вас не установлен API ключ. Проверьте конфиг <code>{prefix}config WGPT</code>.')
                return
            if mono == True:
                m1 = '<code>'
                m2 = '</code>'
            else:
                m1 = ''
                m2 = ''
            messagef = message.raw_text
            openai.api_key = self.config["api_key"]
            model_engine = "text-davinci-003"
            completion = openai.Completion.create(
                engine=model_engine,
                prompt=messagef,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
            )
            response = completion.choices[0].text
            await message.reply(f"<b>GPT: </b>{m1}{response}{m2}")

    async def client_ready(self, client, db):
        self.client = client
        self.db = db
        aa = self.db.get(self.name, 'prefix')
        if aa == None:
            self.db.set(self.name, 'prefix', '//')
        bb = self.db.get(self.name, 'mono')
        if bb == None:
            self.db.set(self.name, 'mono', 'False')

    @loader.command()
    async def gptsettings(self, message):
        ''' - настройки модуля\nprefix - изменить префикс\nmono - выделять ли ответ GPT моноширным шрифтом?'''
        aa = self.db.get(self.name, 'prefix')
        bb = self.db.get(self.name, 'mono')
        prefix = utils.escape_html(self.get_prefix())
        args = utils.get_args_raw(message)
        if args == 'prefix':
            await utils.answer(message, f'<emoji document_id=5784993237412351403>✅</emoji> Ваш префикс "<code>{aa}</code>".')
            return
        if 'prefix ' in args:
            args = args.split()
            if args[0] == 'prefix':
                self.db.set(self.name, 'prefix', args[1])
                await utils.answer(message, f'<emoji document_id=5784993237412351403>✅</emoji> Ваш префикс заменен на <code>{args[1]}</code>')
                return
        if args == 'mono':
            if bb == True:
                self.db.set(self.name, 'mono', False)
                bbs = '<emoji document_id=5784993237412351403>✅</emoji> Ответ больше не выделяется моноширным шрифтом.'
                
            else:
                bbs = '<emoji document_id=5784993237412351403>✅</emoji> Ответ теперь выделяется моноширным шрифтом.'
                self.db.set(self.name, 'mono', True)
            await utils.answer(message, bbs)
            return
        elif not args:
            await utils.answer(message, f'<emoji document_id=5877477244938489129>🚫</emoji> <b>Error |</b> <code>{prefix}gptsettings</code>\nВы не указали параметр.')
            return
        else:
            await utils.answer(message, f'Параметра <code>{args}</code> не существует.')
            return
        
        
        
