import logging
import asyncio
import os
from langchain.agents import tool
from dotenv import load_dotenv

from vocode.streaming.models.message import BaseMessage
from vocode.streaming.models.synthesizer import ElevenLabsSynthesizerConfig
from vocode.streaming.models.transcriber import (
    DeepgramTranscriberConfig,
    PunctuationEndpointingConfig,
)

from vocode.streaming.models.telephony import TwilioConfig


load_dotenv()

from call_transcript_utils import delete_transcript, get_transcript

from vocode.streaming.telephony.conversation.outbound_call import OutboundCall
from vocode.streaming.telephony.config_manager.redis_config_manager import (
    RedisConfigManager,
)
from vocode.streaming.models.agent import ChatGPTAgentConfig
import time

LOOP = asyncio.new_event_loop()
asyncio.set_event_loop(LOOP)

my_information = "Use this information whenever needed User information " + open("info.txt").read() + "  . Your task "

@tool("call phone number")
def call_phone_number(input: str) -> str:
    """calls a phone number as a bot and returns a transcript of the conversation. Verifies the phone number from the user before calling.
    make sure you call `get all contacts` first to get a list of phone numbers to call.
    the input to this tool is a pipe separated list of a phone number, a prompt (including history), and the first thing the bot should say
    The prompt should instruct the bot with what to do on the call and be in the 3rd person,
    like 'the assistant is performing this task' instead of 'perform this task'.

    e.g. phone_number|prompt|initial_message

    should only use this tool once it has found and verified adequate phone number to call.
    """
    phone_number, prompt, initial_message = input.split("|",2)
    print(phone_number, prompt, initial_message)
    call = OutboundCall(
        base_url=os.environ["BASE_URL"],
        to_phone=phone_number,
        from_phone=os.environ["TWILIO_PHONE"],
        config_manager=RedisConfigManager(),
        agent_config=ChatGPTAgentConfig(
            initial_message=BaseMessage(text=initial_message),
            prompt_preamble=my_information + prompt,
            generate_responses=True,
            allow_agent_to_be_cut_off=False
        ),
        twilio_config=TwilioConfig(
                account_sid=os.environ["TWILIO_ACCOUNT_SID"],
                auth_token=os.environ["TWILIO_AUTH_TOKEN"],
                record=True
            ),
        synthesizer_config=ElevenLabsSynthesizerConfig.from_telephone_output_device(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID"),
        ),
        transcriber_config=DeepgramTranscriberConfig.from_telephone_input_device(
            endpointing_config=PunctuationEndpointingConfig()
        ),
        logger=logging.Logger("OutboundCall"),
    )
    LOOP.run_until_complete(call.start())
    return "Call Started"
