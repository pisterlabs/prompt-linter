import openai 
from openai import OpenAIError
import logging

class RegularAgent:
    def __init__(
            self,
            prompt: str = "",
            conversation: dict = [],
            system_message: str = "",
            model: str = "gpt-4-0613",
            temperature: float = 0.6,
            top_p: float = 1.0,
            stream: bool = False,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0):
        
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.prompt = prompt
        self.conversation = conversation
        self.system_message = system_message
        self.stream = stream
    
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    @property
    def top_p(self):
        return self._top_p

    @top_p.setter
    def top_p(self, value):
        self._top_p = value

    @property
    def frequency_penalty(self):
        return self._frequency_penalty

    @frequency_penalty.setter
    def frequency_penalty(self, value):
        self._frequency_penalty = value

    @property
    def presence_penalty(self):
        return self._presence_penalty

    @presence_penalty.setter
    def presence_penalty(self, value):
        self._presence_penalty = value
        
    @property
    def system_message(self):
        return self._system_message
    
    @system_message.setter
    def system_message(self, value):
        self._system_message = value

    def call(
            self,
            prompt: str = "",
            conversation: dict = [],
            system_message: str = "",
            model: str = "gpt-4-0613",
            temperature: float = 0.6,
            top_p: float = 1.0,
            stream: bool = True,
            frequency_penalty: float = 0.0,
            presence_penalty: float = 0.0):
            
        try:
            response = openai.ChatCompletion.create(
                model=model,
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                messages=[
                    {"role": "system", "content": system_message},
                ] + conversation + [
                    {"role": "user", "content": prompt}
                ],
                stream=stream,
            )
            
            #print('streaming response')
            #collected_chunks = []
            #collected_messages = []
            #iterate through the stream of events
            #for chunk in response:
            #    collected_chunks.append(chunk)  # save the event response                
            #    chunk_message = chunk['choices'][0]['delta']  # extract the message
            #    collected_messages.append(chunk_message)  # save the message
            #    if (chunk.choices[0].finish_reason != 'stop'):
            #        print(chunk.choices[0].delta.content)

            #print the time delay and text receivedquest")
            #full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
            #return_object = {
            #    "role": "assistant",
            #    "content": full_reply_content,
            #}
            #print(f"Full conversation received: {full_reply_content}")
        except OpenAIError as error:
            logging.error(f"OpenAI API call failed: {str(error)}")
            return "OpenAI API call failed due to an internal server error.", conversation
        except openai.error.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
            return "Failed to connect to OpenAI.", conversation
        except openai.error.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
            return "Requests exceed OpenAI rate limit.", conversation
        return response