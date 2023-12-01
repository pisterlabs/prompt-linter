from openai import OpenAI;  client = OpenAI()
from constants import *


def voice2text(filename):
    file = open(filename, "rb")
    response = client.audio.transcriptions.create(
        model="whisper-1", 
        file= file
    )
    file.close()
    return response.text


def text2voice(text):
    response = client.audio.speech.create(
        model="tts-1",
        voice="echo",
        input=text
    )
    response.stream_to_file(DIR_FILE_PATH / "temp.mp3")
    return open('temp.mp3','rb')


def predict(chat, stream=True):
    prompt = chat.get_bound( bound=4000 )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        messages=prompt,
        stream = stream
    )

    if stream:
        return response
    return response.choices[0].message.content


def exist(username, chats):
    return True if username in chats.keys() else False


def get_temp_file(user):
    path = PATH_HISTORY / user / '~temp.json'
    if not os.path.exists(path):
        return
    with open(path, 'r') as f:
        endpoint = json.load(f)
    return endpoint


def create_chat(msg, chats):
    username = msg.from_user.username
    if not exist(username, chats):
        chats[username] = Chat( msg=msg, history=ROLES[DEFAULT_ROLE] )
        chats[username].create_temp_file()
        return True
    return False


def get_user_by_chat_id(chat_id, chats):
    for user, chat in chats.items():
        if chat.chat_id == chat_id:
            return user
    return None

def _print_(*msg):
    print(*msg, end='\n\n')
    

#--------------------------------------------------------------------------------------------------------------
# clases útiles
from telebot import types
import numpy as np, pandas as pd, os, re, json
import tiktoken;  enc = tiktoken.encoding_for_model("gpt-3.5-turbo")


class Chat:
    def __init__(self, msg: types.Message=None, chat=None, from_dict :dict=None, history: list=[]):
        if msg:	
            self.user = msg.from_user.username
            self.chat_id = msg.chat.id
            self.first_name = msg.from_user.first_name
            
            self.history = history
            self.tokens = self.get_tokens_from_chat(history)
            self.length = len(history)
            
            self.file_in_use = ''
            self.role = self.get_role()
            self.pinned_msg_id = None
            print(f'Se creó un nuevo chat para el usuario `{msg.from_user.username}`')
            _print_(f'role: `{self.role}`')

        elif chat:
            self.user = chat.user
            self.chat_id = chat.chat_id
            self.first_name = chat.first_name
			
            self.history = chat.history
            self.tokens = chat.tokens
            self.length = chat.length
			
            self.file_in_use = chat.file_in_use
            self.role = chat.role
            self.pinned_msg_id = chat.pinned_msg_id

        elif from_dict:
            self.user = from_dict['user']
            self.chat_id = from_dict['chat_id']
            self.first_name = from_dict['first_name']
            
            self.history = from_dict['history']
            self.tokens = from_dict['tokens']
            self.length = from_dict['length']
            
            self.file_in_use = from_dict['file_in_use']
            self.role = from_dict['role']
            self.pinned_msg_id = from_dict['pinned_msg_id']

    def add(self, role, content):
        self.history.append({"role": role, "content": content})
        self.tokens += self.get_tokens(content)
        self.length += 1
        return self

    def copy(self):
        return Chat(chat=self)

    def __str__(self):
        return '\n'.join([f"{m['role']}: {m['content']}" for m in self.history])

    def get_bound(self, bound=4000):
        tokens=[]
        for message in self.history:
            tokens.append(self.get_tokens(message["content"]))
        
        temp = np.array(tokens)[::-1].cumsum()
        return np.array(self.history, dtype=object)[temp[::-1]<bound].tolist()

    def get_tokens(self, text):
        return len(enc.encode(text))
    
    def get_tokens_from_chat(self, chat):
        tokens = 0
        for message in chat:
            tokens += self.get_tokens(message["content"])
        return tokens
    
    def get_role(self):
        if not self.history[0]['content']:
            return 'General assistant 🤖'

        for key, value in zip(ROLES.keys(), ROLES.values()):
            if key == 'General assistant 🤖':
                continue
            if value[0]['content'] == self.history[0]['content']:
                return key
    
    def set_role(self, role, bot):
        if role == self.role:
            return
        self.save_content()
        self.history = ROLES[role]
        self.tokens = self.get_tokens_from_chat(self.history)
        self.length = len(self.history)
        self.role = role
        self.file_in_use = ''
        print(f'Se fijó el rol {self.role} en el chat del usuario {self.user}!')
        self.send_hello(bot)

    def delete_content(self):
        self.history = self.history[:2]
        self.tokens = self.get_tokens_from_chat(self.history)
        self.length = len(self.history)
        print(f'Se eliminó el historial del chat del usuario {self.user}!')

    def save_content(self):
        try:
            if self.length > 2:
                print(f'Guardando el chat del usuario {self.user}!')
                if self.file_in_use: # verificamos si se está usando un archivo
                    path = PATH_HISTORY / self.user
                    if not os.path.exists(path): # verificamos si existe una carpeta para el usuario
                        os.mkdir(path)
                    path = path / self.file_in_use
                    if os.path.exists(path): # verificamos si existe el archivo
                        pd.DataFrame(self.history).T.to_json(path)
                        print('Se guardó el archivo',end='\n\n')
                        return
                    self.file_in_use = '' # si el archivo no existe, se limpia la variable file_in_use

                # si no se está usando un archivo o el archivo en uso ya no existe, se guarda en un nuevo archivo
                prompt = "Elije un título para la conversación basado en el contexto actual, que NO exceda las 5 palabras."
                chat = self.copy()
                chat.history = chat.history[2:] + [{"role": "user", "content": prompt}]
                response = predict(chat, stream=False).strip('.')

                regex = re.compile('[^a-zA-ZáéíóúÁÉÍÓÚñÑ0-9_\-.\s]+')
                response = regex.sub('', response)
                
                path = PATH_HISTORY / self.user
                if not os.path.exists(path): # verificamos si existe una carpeta para el usuario
                    os.mkdir(path)

                path = path / f'{response.lower()}.json'
                pd.DataFrame(self.history).T.to_json(path)
                self.file_in_use = f'{response.lower()}.json'
                print('Se guardó el archivo',end='\n\n')
        except Exception as e:
            _print_('Error in `save_content()`:', e)
        
    def get_contents_files(self, extension='.json'):
        print('Se entró en la función get_contents_files():')
        try:
            path = PATH_HISTORY / self.user
            files_name = [file for file in os.listdir(path) if file.endswith(extension) and file != '~temp.json']
            contents = [file.capitalize().replace(extension,'') for file in files_name]

            print(f'1- Se obtuvieron los nombres de los archivos {extension}!',end='\n\n')
            return files_name, contents
        except Exception as e:
            _print_('Error in `get_contents_files()`:', e)
            return [], []
        
    def get_content_from_file(self, file_name: str) -> list:
        print('Se entró en la función get_content_from_file():')
        try:
            path = PATH_HISTORY / self.user / file_name
            content = pd.read_json(path).to_dict()

            print(f'1- Se obtuvo el registro a partir del archivo "{file_name}"!',end='\n\n')
            return [v for v in content.values()]
        except Exception as e:
            _print_('Error in `get_content_from_file()`:', e)
            return []
        
    def set_content_from_file(self, file_name: str, bot):
        content = self.get_content_from_file(file_name)

        if not content:
            bot.send_message(chat_id=self.chat_id, text='No se pudo cargar el historial de este chat 🤷‍♂️')
            return

        self.save_content()
        self.history = content
        self.tokens = self.get_tokens_from_chat(self.history)
        self.length = len(self.history)
        self.role = self.get_role()
        self.file_in_use = file_name
        self.send_hello(bot)

    def create_temp_file(self):
        endpoint = {}
        endpoint['chat'] = self.__dict__

        if not os.path.isdir( PATH_HISTORY / self.user ):
            os.mkdir( PATH_HISTORY / self.user )

        with open(PATH_HISTORY / self.user / '~temp.json', 'w') as f:
            json.dump(endpoint, f)

    def delete_temp_file(user):
        if not os.path.isdir( PATH_HISTORY / user ):
            return
        os.remove( PATH_HISTORY / user / '~temp.json' )

    def send_hello(self, bot):
        print(f'Enviando saludo al usuario `{self.user}`')
        msg = bot.send_message(chat_id=self.chat_id, text=f'Hola {self.first_name} 👋. Mi rol actual es: {self.role} !')
        
        if self.pinned_msg_id is not None:
            try:
                bot.unpin_chat_message(chat_id=self.chat_id, message_id=self.pinned_msg_id)
            except Exception as e:
                print(e)
                pass
        self.pinned_msg_id = msg.message_id
        
        bot.pin_chat_message(chat_id=self.chat_id, message_id=self.pinned_msg_id)
        _print_('Se fijó el mensaje de saludo')

    def response_to_text(self, msg: types.Message, bot):
        self.add("user", msg.text)
        m = bot.send_message(chat_id=msg.chat.id, text='Generating...')
        bot.send_chat_action(msg.chat.id, 'typing')
        message = predict(self, stream=False)  
        self.add("assistant", message)
        bot.edit_message_text(chat_id=msg.chat.id, message_id=m.message_id, text=message)
        self.create_temp_file()

    def transcript_to_text(self, msg: types.Message, bot):
        m = bot.send_message(chat_id=msg.chat.id, text='Transcribing...')
        bot.send_chat_action(msg.chat.id, 'typing')
        voice_bytes = bot.download_file( bot.get_file(msg.voice.file_id).file_path )
        with open('output.ogg', 'wb') as f:
            f.write(voice_bytes)
        message = voice2text('output.ogg'); os.remove('output.ogg')
        bot.edit_message_text(chat_id=msg.chat.id, message_id=m.message_id, text=message)

    def response_to_voice(self, msg: types.Voice, bot):

        m = bot.reply_to(msg, 'Downloading...')
        voice_bytes = bot.download_file( bot.get_file(msg.voice.file_id).file_path )

        bot.edit_message_text(chat_id=msg.chat.id, message_id=m.message_id, text='Saving...')
        with open('output.ogg', 'wb') as f:
            f.write(voice_bytes)

        bot.edit_message_text(chat_id=msg.chat.id, message_id=m.message_id, text='Transcribing...')
        input_text = voice2text('output.ogg'); os.remove('output.ogg')

        self.add("user", input_text)

        bot.edit_message_text(chat_id=msg.chat.id, message_id=m.message_id, text='Generating...')
        bot.send_chat_action(msg.chat.id, 'typing')
        output_text = predict(self, stream=False)

        self.add("assistant", output_text)

        output_text = '🗣️​ **You**\n\n' + input_text + '\n\n' + '🤖 **GPT**\n\n' + output_text

        # output_voice = text2voice(output_text)
        # bot.send_voice(chat_id=msg.chat.id, voice=output_voice); output_voice.close(); os.remove('temp.mp3')
        bot.edit_message_text(chat_id=msg.chat.id, message_id=m.message_id, text=output_text)
        self.create_temp_file()

    def new_chat(self, bot):
        self.save_content()
        self.delete_content()
        self.file_in_use = ''
        self.send_hello(bot)
