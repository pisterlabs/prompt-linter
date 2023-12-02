## built-in libaries
import typing
import base64
import re
import time
import typing
import asyncio
import os

## third party modules
from openai import AsyncOpenAI
from openai import AuthenticationError, InternalServerError, RateLimitError, APIError, APIConnectionError, APITimeoutError

import backoff
import tiktoken
import spacy

## custom modules
from handlers.json_handler import JsonHandler
from handlers.katakana_handler import KatakanaHandler

from modules.file_ensurer import FileEnsurer
from modules.logger import Logger
from modules.toolkit import Toolkit

##-------------------start-of-SystemTranslationMessage--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class SystemTranslationMessage(typing.TypedDict):

    """

    SystemTranslationMessage is a type that is used to interact with the OpenAI API and translates the text by batch, specifically for the system message.

    """

    role: typing.Literal['system']
    content: str

##-------------------start-of-ModelTranslationMessage--------------------------------------------------------------------------------------------------------------------------------------------------------------------------


class ModelTranslationMessage(typing.TypedDict):

    """

    ModelTranslationMessage is a type that is used to interact with the OpenAI API and translates the text by batch, specifically for the model/user message.

    """

    role: typing.Literal['user']
    content: str

##-------------------start-of-MaxBatchDurationExceeded--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class MaxBatchDurationExceeded(Exception):

    """

    MaxBatchDurationExceeded is an exception that is raised when the max batch duration is exceeded.

    """

    pass


##-------------------start-of-Kijiku--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class Kijiku:

    """
    
    Kijiku is a secondary class that is used to interact with the OpenAI API and translates the text by batch.
    
    """
    
    ## the text to translate
    text_to_translate =  []

    ## the translated text
    translated_text = []

    ## the text for j-e checking
    je_check_text = []

    ## the text for errors that occur during translation
    error_text = []

    ## the messages that will be sent to the api, contains a system message and a model message, system message is the instructions,
    ## model message is the text that will be translated  
    translation_batches = []

    ## number of malformed batches that have occurred
    malformed_batches = 0

    ## async client session
    client = AsyncOpenAI(max_retries=0, api_key="Dummy")

    _semaphore = asyncio.Semaphore(30)

    ##--------------------------------------------------------------------------------------------------------------------------

    translation_print_result = ""

    ##--------------------------------------------------------------------------------------------------------------------------

    MODEL = ""
    translation_instructions = ""
    message_mode = 0 
    prompt_size = 0
    sentence_fragmenter_mode = 0
    je_check_mode = 0
    num_of_malform_retries = 0
    max_batch_duration = 0
    num_concurrent_batches = 0

##-------------------start-of-translate()--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def translate() -> None:

        """

        Translate the text in the file at the path given.

        """

        ## set this here cause the try-except could throw before we get past the settings configuration
        time_start = time.time()

        try:
        
            await Kijiku.initialize()

            JsonHandler.validate_json()

            await Kijiku.check_settings()

            ## set actual start time to the end of the settings configuration
            time_start = time.time()

            await Kijiku.commence_translation()

        except Exception as e:
            
            Kijiku.translation_print_result += "An error has occurred, outputting results so far..."

            FileEnsurer.handle_critical_exception(e)

        finally:

            time_end = time.time() 

            Kijiku.assemble_results(time_start, time_end)

##-------------------start-of-initialize()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def initialize() -> None:

        """

        Sets the open api key.
    
        """

        await Kijiku.setup_api_key()

        ## try to load the kijiku rules
        try: 

            JsonHandler.load_kijiku_rules()

        ## if the kijiku rules don't exist, create them
        except: 
            
            JsonHandler.reset_kijiku_rules_to_default()

            JsonHandler.load_kijiku_rules()
            
        Toolkit.clear_console()

##-------------------start-of-setup_api_key()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def setup_api_key() -> None:

        """
        
        Sets up the api key.

        """

        ## get saved api key if exists
        try:
            with open(FileEnsurer.openai_api_key_path, 'r', encoding='utf-8') as file: 
                api_key = base64.b64decode((file.read()).encode('utf-8')).decode('utf-8')

            Kijiku.client.api_key = api_key

            ## make dummy request to check if api key is valid
            await Kijiku.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content":"This is a test."}],
                max_tokens=1
            )
        
            Logger.log_action("Used saved api key in " + FileEnsurer.openai_api_key_path, output=True)
            Logger.log_barrier()

            time.sleep(2)

        ## else try to get api key manually
        except:

            Toolkit.clear_console()
                
            api_key = input("DO NOT DELETE YOUR COPY OF THE API KEY\n\nPlease enter the openapi key you have : ")

            ## if valid save the api key
            try: 

                Kijiku.client.api_key = api_key

                ## make dummy request to check if api key is valid
                await Kijiku.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"user","content":"This is a test."}],
                    max_tokens=1
                )

                FileEnsurer.standard_overwrite_file(FileEnsurer.openai_api_key_path, base64.b64encode(api_key.encode('utf-8')).decode('utf-8'), omit=True)
                
            ## if invalid key exit
            except AuthenticationError: 
                    
                Toolkit.clear_console()
                        
                Logger.log_action("Authorization error with setting up openai, please double check your api key as it appears to be incorrect.", output=True)

                Toolkit.pause_console()
                        
                exit()

            ## other error, alert user and raise it
            except Exception as e: 

                Toolkit.clear_console()
                        
                Logger.log_action("Unknown error with setting up openai, The error is as follows " + str(e)  + "\nThe exception will now be raised.", output=True)

                Toolkit.pause_console()

                raise e

##-------------------start-of-check-settings()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def check_settings() -> None:

        """

        Prompts the user to confirm the settings in the kijiku rules file.

        """

        print("Are these settings okay? (1 for yes or 2 for no) : \n\n")

        ## Reset settings to default if the json is botched
        if("open ai settings" not in JsonHandler.current_kijiku_rules):
            JsonHandler.reset_kijiku_rules_to_default()

        for key, value in JsonHandler.current_kijiku_rules["open ai settings"].items():
            print(key + " : " + str(value))

        if(input("\n") == "1"):
            pass
        else:
            JsonHandler.change_kijiku_settings()

        Toolkit.clear_console()

        print("Do you want to change your api key? (1 for yes or 2 for no) : ")

        if(input("\n") == "1"):
            os.remove(FileEnsurer.openai_api_key_path)
            await Kijiku.setup_api_key()

        Toolkit.clear_console()


##-------------------start-of-commence_translation()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def commence_translation() -> None:

        """

        Uses all the other functions to translate the text provided by Kudasai.

        """
        
        Logger.log_barrier()
        Logger.log_action("Kijiku Activated, Settings are as follows : ")
        Logger.log_barrier()

        for key,value in JsonHandler.current_kijiku_rules["open ai settings"].items():
            Logger.log_action(key + " : " + str(value))

        Kijiku.MODEL = JsonHandler.current_kijiku_rules["open ai settings"]["model"]
        Kijiku.translation_instructions = JsonHandler.current_kijiku_rules["open ai settings"]["system_message"]
        Kijiku.message_mode = int(JsonHandler.current_kijiku_rules["open ai settings"]["message_mode"])
        Kijiku.prompt_size = int(JsonHandler.current_kijiku_rules["open ai settings"]["num_lines"])
        Kijiku.sentence_fragmenter_mode = int(JsonHandler.current_kijiku_rules["open ai settings"]["sentence_fragmenter_mode"])
        Kijiku.je_check_mode = int(JsonHandler.current_kijiku_rules["open ai settings"]["je_check_mode"])
        Kijiku.num_of_malform_retries = int(JsonHandler.current_kijiku_rules["open ai settings"]["num_malformed_batch_retries"])
        Kijiku.max_batch_duration = float(JsonHandler.current_kijiku_rules["open ai settings"]["batch_retry_timeout"])
        Kijiku.num_concurrent_batches = int(JsonHandler.current_kijiku_rules["open ai settings"]["num_concurrent_batches"])

        Kijiku._semaphore = asyncio.Semaphore(Kijiku.num_concurrent_batches)

        Toolkit.clear_console()

        Logger.log_barrier()
        Logger.log_action("Starting Prompt Building")
        Logger.log_barrier()

        Kijiku.build_translation_batches()

        ## get cost estimate and confirm
        num_tokens, min_cost, Kijiku.MODEL = Kijiku.estimate_cost(Kijiku.MODEL)

        print("\nNote that the cost estimate is not always accurate, and may be higher than the actual cost. However cost calculation now includes output tokens.\n")

        Logger.log_barrier()
        Logger.log_action("Calculating cost")
        Logger.log_barrier()
        
        Logger.log_action("Estimated number of tokens : " + str(num_tokens), output=True, omit_timestamp=True)
        Logger.log_action("Estimated minimum cost : " + str(min_cost) + " USD", output=True, omit_timestamp=True)
        Logger.log_barrier()

        if(input("\nContinue? (1 for yes or 2 for no) : ") == "1"):
            Logger.log_action("User confirmed translation.")

        else:
            Logger.log_action("User cancelled translation.")
            exit()

        Toolkit.clear_console()

        Logger.log_barrier()
        
        Logger.log_action("Starting Translation...", output=True)
        Logger.log_barrier()

        ## requests to run asynchronously
        async_requests = []
        length = len(Kijiku.translation_batches)

        for i in range(0, length, 2):
            async_requests.append(Kijiku.handle_translation(i, length, Kijiku.translation_batches[i], Kijiku.translation_batches[i+1]))

        ## Use asyncio.gather to run tasks concurrently/asynchronously and wait for all of them to complete
        results = await asyncio.gather(*async_requests)

        Logger.log_barrier()
        Logger.log_action("Translation Complete!", output=True)

        Logger.log_barrier()
        Logger.log_action("Starting Redistribution...", output=True)

        Logger.log_barrier()

        ## Sort results based on the index to maintain order
        sorted_results = sorted(results, key=lambda x: x[0])

        ## Redistribute the sorted results
        for index, translated_prompt, translated_message in sorted_results:
            Kijiku.redistribute(translated_prompt, translated_message)

        ## try to pair the text for j-e checking if the mode is 2
        if(Kijiku.je_check_mode == 2):
            Kijiku.je_check_text = Kijiku.fix_je()

        Toolkit.clear_console()

        Logger.log_action("Done!", output=True)
        Logger.log_barrier()

##-------------------start-of-generate_prompt()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def generate_prompt(index:int) -> tuple[typing.List[str],int]:

        """

        Generates prompts for the messages meant for the ai.

        Parameters:
        index (int) an int representing where we currently are in the text file.

        Returns:
        prompt (list - string) a list of japanese lines that will be assembled into messages.
        index (int) an updated int representing where we currently are in the text file.

        """

        prompt = []

        while(index < len(Kijiku.text_to_translate)):
            sentence = Kijiku.text_to_translate[index]

            if(len(prompt) < Kijiku.prompt_size):

                if(any(char in sentence for char in ["▼", "△", "◇"])):
                    prompt.append(sentence + '\n')
                    Logger.log_action("Sentence : " + sentence + ", Sentence is a pov change... leaving intact.")

                elif("part" in sentence.lower() or all(char in ["１","２","３","４","５","６","７","８","９", " "] for char in sentence) and not all(char in [" "] for char in sentence)):
                    prompt.append(sentence + '\n') 
                    Logger.log_action("Sentence : " + sentence + ", Sentence is part marker... leaving intact.")

                elif(bool(re.match(r'^[\W_\s\n-]+$', sentence)) and not KatakanaHandler.is_punctuation(sentence)):
                    Logger.log_action("Sentence : " + sentence + ", Sentence is punctuation... skipping.")
            
                elif(bool(re.match(r'^[A-Za-z0-9\s\.,\'\?!]+\n*$', sentence) and "part" not in sentence.lower())):
                    Logger.log_action("Sentence is empty... skipping translation.")

                else:
                    prompt.append(sentence + "\n")
    
            else:
                return prompt, index
            
            index += 1

        return prompt, index
    
##-------------------start-of-build_translation_batches()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def build_translation_batches() -> None:

        """

        Builds translations batches dict for ai prompt.
        
        """

        i = 0

        while i < len(Kijiku.text_to_translate):
            prompt, i = Kijiku.generate_prompt(i)

            prompt = ''.join(prompt)

            ## message mode one structures the first message as a system message and the second message as a model message
            if(Kijiku.message_mode == 1):
                system_msg = SystemTranslationMessage(role="system", content=Kijiku.translation_instructions)

            ## while message mode two structures the first message as a model message and the second message as a system message, typically used for non-gpt-4 models if at all
            else:
                system_msg = ModelTranslationMessage(role="user", content=Kijiku.translation_instructions)

            Kijiku.translation_batches.append(system_msg)

            model_msg = ModelTranslationMessage(role="user", content=prompt)

            Kijiku.translation_batches.append(model_msg)

        Logger.log_barrier()
        Logger.log_action("Built Messages : ")
        Logger.log_barrier()

        i = 0

        for message in Kijiku.translation_batches:

            i+=1

            if(i % 2 == 0):

                Logger.log_action(str(message))
        
            else:

                Logger.log_action(str(message))
                Logger.log_barrier()

##-------------------start-of-estimate_cost()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def estimate_cost(model:str, price_case:int | None = None) -> typing.Tuple[int, float, str]:

        """

        Attempts to estimate cost.

        Parameters:
        model (string) : the model used to translate the text.
        price_case (int) : the price case used to calculate the cost.

        Returns:
        num_tokens (int) : the number of tokens used.
        min_cost (float) : the minimum cost of translation.
        model (string) : the model used to translate the text.

        """
        
        allowed_models = [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-3.5-turbo-0301",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-3.5-turbo-1106",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4-1106-preview"
        ]

        assert model in allowed_models, f"""Kudasai does not support : {model}. See https://github.com/OpenAI/OpenAI-python/blob/main/chatml.md for information on how messages are converted to tokens."""

        ## default models are first, then the rest are sorted by price case
        if(price_case is None):

            if(model == "gpt-3.5-turbo"):
                print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-1106 as it is the most recent version of gpt-3.5-turbo.")
                return Kijiku.estimate_cost("gpt-3.5-turbo-1106", price_case=2)
            
            elif(model == "gpt-4"):
                print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-1106-preview as it is the most recent version of gpt-4.")
                return Kijiku.estimate_cost("gpt-4-1106-preview", price_case=4)
            
            elif(model == "gpt-3.5-turbo-0613"):
                print("Warning: gpt-3.5-turbo-0613 is considered depreciated by OpenAI as of November 6, 2023 and could be shutdown as early as June 13, 2024. Consider switching to gpt-3.5-turbo-1106.")
                return Kijiku.estimate_cost(model, price_case=1)

            elif(model == "gpt-3.5-turbo-0301"):
                print("Warning: gpt-3.5-turbo-0301 is considered depreciated by OpenAI as of June 13, 2023 and could be shutdown as early as June 13, 2024. Consider switching to gpt-3.5-turbo-1106 unless you are specifically trying to break the filter.")
                return Kijiku.estimate_cost(model, price_case=1)
            
            elif(model == "gpt-3.5-turbo-1106"):
                return Kijiku.estimate_cost(model, price_case=2)
            
            elif(model == "gpt-3.5-turbo-16k-0613"):
                print("Warning: gpt-3.5-turbo-16k-0613 is considered depreciated by OpenAI as of November 6, 2023 and could be shutdown as early as June 13, 2024. Consider switching to gpt-3.5-turbo-1106.")
                return Kijiku.estimate_cost(model, price_case=3)
            
            elif(model == "gpt-4-1106-preview"):
                return Kijiku.estimate_cost(model, price_case=4)
            
            elif(model == "gpt-4-0314"):
                print("Warning: gpt-4-0314 is considered depreciated by OpenAI as of June 13, 2023 and could be shutdown as early as June 13, 2024. Consider switching to gpt-4-0613.")
                return Kijiku.estimate_cost(model, price_case=5)
            
            elif(model == "gpt-4-0613"):
                return Kijiku.estimate_cost(model, price_case=5)
            
            elif(model == "gpt-4-32k-0314"):
                print("Warning: gpt-4-32k-0314 is considered depreciated by OpenAI as of June 13, 2023 and could be shutdown as early as June 13, 2024. Consider switching to gpt-4-32k-0613.")
                return Kijiku.estimate_cost(model, price_case=6)
            
            elif(model == "gpt-4-32k-0613"):
                return Kijiku.estimate_cost(model, price_case=6)
            
        else:
            encoding = tiktoken.encoding_for_model(model)

            cost_per_thousand_input_tokens = 0
            cost_per_thousand_output_tokens = 0

            ## gpt-3.5-turbo-0301
            ## gpt-3.5-turbo-0613
            if(price_case == 1):
                cost_per_thousand_input_tokens = 0.0015
                cost_per_thousand_output_tokens = 0.0020

            ## gpt-3.5-turbo-1106
            elif(price_case == 2):
                cost_per_thousand_input_tokens = 0.0010
                cost_per_thousand_output_tokens = 0.0020

            ## gpt-3.5-turbo-16k-0613
            elif(price_case == 3):
                cost_per_thousand_input_tokens = 0.0030
                cost_per_thousand_output_tokens = 0.0040

            ## gpt-4-1106-preview
            elif(price_case == 4):
                cost_per_thousand_input_tokens = 0.01
                cost_per_thousand_output_tokens = 0.03

            ## gpt-4-0314
            ## gpt-4-0613
            elif(price_case == 5):
                cost_per_thousand_input_tokens = 0.03
                cost_per_thousand_output_tokens = 0.06

            ## gpt-4-32k-0314
            ## gpt-4-32k-0613
            elif(price_case == 6):
                cost_per_thousand_input_tokens = 0.06
                cost_per_thousand_output_tokens = 0.012

            ## break down the text into a string than into tokens
            text = ''.join(Kijiku.text_to_translate)

            num_tokens = len(encoding.encode(text))

            min_cost_for_input = round((float(num_tokens) / 1000.00) * cost_per_thousand_input_tokens, 5)
            min_cost_for_output = round((float(num_tokens) / 1000.00) * cost_per_thousand_output_tokens, 5)

            min_cost = round(min_cost_for_input + min_cost_for_output, 5)

            return num_tokens, min_cost, model
        
        raise Exception("An unknown error occurred while calculating the minimum cost of translation.")
    
##-------------------start-of-translate_message()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    ## backoff wrapper for retrying on errors
    @staticmethod
    @backoff.on_exception(backoff.expo, max_time=lambda: Kijiku.get_max_batch_duration(), exception=(AuthenticationError, InternalServerError, RateLimitError, APIError, APIConnectionError, APITimeoutError), on_backoff=lambda details: Kijiku.log_retry(details), on_giveup=lambda details: Kijiku.log_failure(details), raise_on_giveup=False)
    async def translate_message(translation_instructions:SystemTranslationMessage | ModelTranslationMessage, translation_prompt:ModelTranslationMessage) -> str:

        """

        Translates system and user message.

        Parameters:
        translation_instructions (dict) : the system message also known as the instructions.
        translation_prompt (dict) : the user message also known as the prompt.

        Returns:
        output (string) a string that gpt gives to us also known as the translation.\n

        """

        ## max_tokens and logit bias are currently excluded due to a lack of need, and the fact that i am lazy

        response = await Kijiku.client.chat.completions.create(
            model=Kijiku.MODEL,
            messages=[
                translation_instructions,
                translation_prompt,
            ],

            temperature = float(JsonHandler.current_kijiku_rules["open ai settings"]["temp"]),
            top_p = float(JsonHandler.current_kijiku_rules["open ai settings"]["top_p"]),
            n = int(JsonHandler.current_kijiku_rules["open ai settings"]["n"]),
            stream = JsonHandler.current_kijiku_rules["open ai settings"]["stream"],
            stop = JsonHandler.current_kijiku_rules["open ai settings"]["stop"],
            presence_penalty = float(JsonHandler.current_kijiku_rules["open ai settings"]["presence_penalty"]),
            frequency_penalty = float(JsonHandler.current_kijiku_rules["open ai settings"]["frequency_penalty"]),            

        )

        ## if anyone knows how to type hint this please let me know
        output = response.choices[0].message.content
        
        return output
    
##-------------------start-of-get_max_batch_duration()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_max_batch_duration() -> float:

        """
        
        Returns the max batch duration.

        Returns:
        max_batch_duration (float) : the max batch duration.

        """

        return Kijiku.max_batch_duration
    
##-------------------start-of-handle_translation()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def handle_translation(index:int, length:int, translation_instructions:SystemTranslationMessage | ModelTranslationMessage, translation_prompt:ModelTranslationMessage) -> tuple[int, ModelTranslationMessage, str]:

        """

        Handles the translation for a given system and user message.

        Parameters:
        index (int) : the index of the message in the original list.
        translation_instructions (dict) : the system message also known as the instructions.
        translation_prompt (dict) : the user message also known as the prompt.

        Returns:\n
        index (int) : the index of the message in the original list.
        translation_prompt (dict) : the user message also known as the prompt.
        translated_message (str) : the translated message.

        """

        async with Kijiku._semaphore:
            num_tries = 0

            while True:
            
                message_number = (index // 2) + 1
                Logger.log_action(f"Trying translation for batch {message_number} of {length//2}...", output=True)


                try:
                    translated_message = await Kijiku.translate_message(translation_instructions, translation_prompt)

                ## will only occur if the max_batch_duration is exceeded, so we just return the untranslated text
                except MaxBatchDurationExceeded:
                    translated_message = translation_prompt["content"]
                    Kijiku.error_text += Logger.log_action(f"Batch {message_number} of {length//2} was not translated due to exceeding the max request duration, returning the untranslated text...", output=True, is_error=True)
                    break

                ## do not even bother if not a gpt 4 model, because gpt-3 seems unable to format properly
                if("gpt-4" not in Kijiku.MODEL):
                    break

                if(await Kijiku.check_if_translation_is_good(translated_message, translation_prompt) or num_tries >= Kijiku.num_of_malform_retries):
                    break

                else:
                    num_tries += 1
                    Kijiku.error_text += Logger.log_action(f"Batch {message_number} of {length//2} was malformed, retrying...", output=True, is_error=True)
                    Kijiku.malformed_batches += 1

            Logger.log_action(f"Translation for batch {message_number} of {length//2} successful!", output=True)

            return index, translation_prompt, translated_message
    
##-------------------start-of-check_if_translation_is_good()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    async def check_if_translation_is_good(translated_message:str, translation_prompt:ModelTranslationMessage) -> bool:

        """
        
        Checks if the translation is good, i.e. the number of lines in the prompt and the number of lines in the translated message are the same.

        Parameters:
        translated_message (string) : the translated message.
        translation_prompt (dict) : the user message also known as the prompt.

        Returns:
        is_valid (bool) : whether or not the translation is valid.

        """

        prompt = translation_prompt["content"]
        is_valid = False

        jap = [line for line in prompt.split('\n') if line.strip()]  ## Remove blank lines
        eng = [line for line in translated_message.split('\n') if line.strip()]  ## Remove blank lines

        if(len(jap) == len(eng)):
            is_valid = True
    
        return is_valid
    
##-------------------start-of-log_retry()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def log_retry(details):

        """

        Logs the retry message.

        Parameters:
        details (dict) : the details of the retry.

        """

        retry_msg = f"Retrying translation after {details['wait']} seconds after {details['tries']} tries {details['target']} due to {details['exception']}."

        Logger.log_barrier()
        Kijiku.error_text += Logger.log_action(retry_msg, is_error=True)
        Logger.log_barrier()

##-------------------start-of-log_failure()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def log_failure(details):

        """
        
        Logs the translation batch failure message.

        Parameters:
        details (dict) : the details of the failure.

        """

        error_msg = f"Exceeded duration, returning untranslated text after {details['tries']} tries {details['target']}."

        Logger.log_barrier()
        Kijiku.error_text += Logger.log_action(error_msg, is_error=True)
        Logger.log_barrier()

        raise MaxBatchDurationExceeded(error_msg)

##-------------------start-of-redistribute()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def redistribute(translation_prompt:dict, translated_message:str) -> None:

        """

        Puts translated text back into the text file.

        Parameters:
        Kijiku (object - Kijiku) : the Kijiku object.
        translated_message (string) : the translated message.

        """

        if(Kijiku.je_check_mode == 1):
            Kijiku.je_check_text.append("\n-------------------------\n"+ str(translation_prompt["content"]) + "\n\n")
            Kijiku.je_check_text.append(translated_message + '\n')
        elif(Kijiku.je_check_mode == 2):
            Kijiku.je_check_text.append(str(translation_prompt["content"]))
            Kijiku.je_check_text.append(translated_message)

        ## mode 1 is the default mode, uses regex and other nonsense to split sentences
        if(Kijiku.sentence_fragmenter_mode == 1): 

            sentences = re.findall(r"(.*?(?:(?:\"|\'|-|~|!|\?|%|\(|\)|\.\.\.|\.|---|\[|\])))(?:\s|$)", translated_message)

            patched_sentences = []
            build_string = None

            for sentence in sentences:
                if(sentence.startswith("\"") and not sentence.endswith("\"") and build_string is None):
                    build_string = sentence
                    continue
                elif(not sentence.startswith("\"") and sentence.endswith("\"") and build_string is not None):
                    build_string += f" {sentence}"
                    patched_sentences.append(build_string)
                    build_string = None
                    continue
                elif(build_string is not None):
                    build_string += f" {sentence}"
                    continue

                Kijiku.translated_text.append(sentence + '\n')

            for i in range(len(Kijiku.translated_text)):
                if Kijiku.translated_text[i] in patched_sentences:
                    index = patched_sentences.index(Kijiku.translated_text[i])
                    Kijiku.translated_text[i] = patched_sentences[index]

        ## mode 2 uses spacy to split sentences
        elif(Kijiku.sentence_fragmenter_mode == 2): 

            nlp = spacy.load("en_core_web_lg")

            doc = nlp(translated_message)
            sentences = [sent.text for sent in doc.sents]

            for sentence in sentences:
                Kijiku.translated_text.append(sentence + '\n')

        ## mode 3 just assumes gpt formatted it properly
        elif(Kijiku.sentence_fragmenter_mode == 3): 
            
            Kijiku.translated_text.append(translated_message + '\n\n')
        
##-------------------start-of-fix_je()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def fix_je() -> typing.List[str]:

        """

        Fixes the J->E text to be more j-e check friendly.

        Note that fix_je() is not always accurate, and may use standard j-e formatting instead of the corrected formatting.

        Returns:
        final_list (list - str) : the fixed J->E text.

        """
        
        i = 1
        final_list = []

        while i < len(Kijiku.je_check_text):
            jap = Kijiku.je_check_text[i-1].split('\n')
            eng = Kijiku.je_check_text[i].split('\n')

            jap = [line for line in jap if line.strip()]  ## Remove blank lines
            eng = [line for line in eng if line.strip()]  ## Remove blank lines    

            final_list.append("-------------------------\n")

            if(len(jap) == len(eng)):

                for jap_line,eng_line in zip(jap,eng):
                    if(jap_line and eng_line): ## check if jap_line and eng_line aren't blank
                        final_list.append(jap_line + '\n\n')
                        final_list.append(eng_line + '\n\n')

                        final_list.append("--------------------------------------------------\n")
     

            else:

                final_list.append(Kijiku.je_check_text[i-1] + '\n\n')
                final_list.append(Kijiku.je_check_text[i] + '\n\n')

                final_list.append("--------------------------------------------------\n")

            i+=2

        return final_list

##-------------------start-of-assemble_results()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    @staticmethod
    def assemble_results(time_start:float, time_end:float) -> None:

        """

        Outputs results to a string.

        """

        Kijiku.translation_print_result += "Time Elapsed : " + Toolkit.get_elapsed_time(time_start, time_end)
        Kijiku.translation_print_result += "\nNumber of malformed batches : " + str(Kijiku.malformed_batches)

        Kijiku.translation_print_result += "\n\nDebug text have been written to : " + FileEnsurer.debug_log_path
        Kijiku.translation_print_result += "\nJ->E text have been written to : " + FileEnsurer.je_check_path
        Kijiku.translation_print_result += "\nTranslated text has been written to : " + FileEnsurer.translated_text_path
        Kijiku.translation_print_result += "\nErrors have been written to : " + FileEnsurer.error_log_path + "\n"

##-------------------start-of-write_kijiku_results()---------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    @staticmethod
    def write_kijiku_results() -> None:

        """
        
        This function is called to write the results of the Kijiku translation module to the output directory.

        """

        ## ensures the output directory exists, cause it could get moved or fucked with.
        FileEnsurer.standard_create_directory(FileEnsurer.output_dir)

        with open(FileEnsurer.error_log_path, 'a+', encoding='utf-8') as file:
            file.writelines(Kijiku.error_text)

        with open(FileEnsurer.je_check_path, 'w', encoding='utf-8') as file:
            file.writelines(Kijiku.je_check_text)

        with open(FileEnsurer.translated_text_path, 'w', encoding='utf-8') as file:
            file.writelines(Kijiku.translated_text)

        ## pushes the tl debug log to the file without clearing the file
        Logger.push_batch()
