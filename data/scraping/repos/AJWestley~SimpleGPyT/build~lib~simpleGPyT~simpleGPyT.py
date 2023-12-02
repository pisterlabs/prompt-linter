'''A simple object-oriented interface for OpenAI's chat completion API.'''

import openai

class Conversation:
    '''
    The Conversation object-class.

    Attributes
    ----------
    model: str
        The OpenAI chat completion model to use, defaults to "gpt-3.5-turbo"
    user: str
        The name of the user, defaults to "user"
    assistant: str
        The name of the assistant, defaults to "assistant"
    temperature: float
        The randomness of responses, with lower numbers being more direct and higher numbers being 
        more random (defaults to 1)
    max_tokens: int
        The maximum number of tokens allowed within a response, defaults to 100
    messages: list
        A list of all the chat's messages
    settings: list
        A list of all the chat's settings
    
    Methods
    -------
    say(message: str)
        Sends a message to the assistant and returns the assistant's response
    
    last_response()
        Gets the last response from the assistant
    
    clear_messages()
        Clears the chat, but keeps assistant settings intact
    
    pop_setting(index: int = -1)
        Removes a setting, defaults to the last setting
    
    get_settings()
        Returns a list of the assistant's settings
    
    add_setting(setting: str)
        Adds a setting to the assistant
    
    clear_settings()
        Clears the assistant's settings, but keeps the chat intact
    
    get_temperature()
        Returns the chat's temperature setting
    
    set_temperature(temperature: float)
        Changes the chat's temperature setting
    
    get_max_tokens()
        Returns the max number of tokens allowed in a chat response
    
    set_max_tokens(max_tokens: int)
        Changes the max number of tokens allowed in a chat response
    
    reset()
        Clears the chat and all settings
    '''

    def __init__(
        self,
        api_key: str,
        user_name: str = "user",
        assistant_name: str = "assistant",
        model: str = "gpt-3.5-turbo",
        settings: list[str] = None,
        temperature: float = 1,
        max_tokens: int = 100
        ) -> None:

        openai.api_key = api_key

        self.__model = model
        self.__temperature = None
        self.__max_tokens = None
        self.__messages = []
        self.__user = user_name
        self.__assistant = assistant_name
        self.__settings = []

        self.set_temperature(temperature)
        self.set_max_tokens(max_tokens)

        if settings is None:
            settings = []
        for setting in settings:
            self.add_setting(setting)


    # Chat

    def say(self, message: str) -> str:
        '''Sends a message to the assistant and gets its response

        Parameters
        ----------
        message : str
            The message to send the assistant
        
        Returns
        -------
        str
            The assistant's response
        '''

        self.__messages.append({"role": "user", "content": message})
        name_settings = [
            {
                "role": "system", 
                "content": f"Your name is {self.__assistant}"
            },
            {
                "role": "system",
                "content": f"The user's name is {self.__user}, \
                but only refer to them by name if asked to"
            }
            ]
        messages = name_settings + self.__settings + self.__messages
        response = openai.ChatCompletion.create(
            model=self.__model,
            messages=messages,
            temperature=self.__temperature,
            max_tokens=self.__max_tokens
        )
        self.__messages.append(response.choices[0]["message"])
        return self.last_response()

    def last_response(self) -> str:
        '''Gets the most recent response from the assistant
        
        Returns
        -------
        str
            The assistant's most recent response
        '''
        return self.__messages[-1]["content"]

    def get_messages(self) -> list:
        '''Gets a list of all messages'''
        return {message["role"]: message["content"] for message in self.__messages}

    def clear_messages(self) -> None:
        '''Clears all messages from the chat'''
        self.__messages.clear()


    # Settings

    def pop_setting(self, index: int = -1) -> str:
        '''Removes a setting, if there is one to remove
        
        Parameters
        ----------
        index : int, optional
            The index of the setting to remove (defaults to 1)
        '''
        return None if len(self.__settings) < 1 else self.__settings.pop(index)

    def get_settings(self) -> list:
        '''Gets a list of the assistant's settings
        
        Returns
        -------
        list
            The assistant's settings
        '''
        return [s["content"] for s in self.__settings]

    def add_setting(self, setting: str) -> None:
        '''Adds a setting to the assistant
        
        Parameters
        ----------
        setting : str
            The setting to be added
        '''
        self.__settings.append({"role": "system", "content": setting})

    def clear_settings(self) -> None:
        '''Erases all the assistant's settings'''
        self.__settings.clear()


    # Getters and Setters

    def get_temperature(self) -> float:
        '''Gets the assistant's temperature setting'''
        return self.__temperature

    def set_temperature(self, temperature: float) -> None:
        '''Sets the assistant's temperature setting
        Only accepts values between 0 and 2
        '''
        if temperature > 2 or temperature < 0:
            raise ValueError(f"temperature = {temperature}. temperature must be between 0 and 2")
        self.__temperature = temperature

    def get_max_tokens(self) -> int:
        '''Gets the maximum number of tokens allowed in a response'''
        return self.__max_tokens

    def set_max_tokens(self, max_tokens: int) -> None:
        '''Sets the maximum number of tokens allowed in a reponse
        Only accepts positive values
        '''
        if max_tokens <= 0:
            raise ValueError(f"max_tokens = {max_tokens}. max_tokens must be >== 0")
        self.__max_tokens = max_tokens


    # Auxilliary

    def reset(self) -> None:
        '''Clears the chat and erases all settings'''
        self.clear_settings()
        self.clear_messages()

    def __str__(self) -> str:
        messages = [
            f"{self.__user if message['role'] == 'user' else self.__assistant}:\n{message['content']}"
            for message in self.__messages
        ]
        return "\n\n".join(messages)

    def __len__(self) -> int:
        return len(self.__messages)
