import contextlib
import logging
import os

import nltk
import openai

from payphone_ai.prompts import Prompt

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def start_ai_conversation(prompt: Prompt):
    transcript = f"""\
    The following is a phone conversation.
    {prompt.text}
    """
    users = ["Caller", "Receiver"]
    start_sequence = f"\n{users[0]}: "
    restart_sequence = f"\n{users[1]}: "
    try:

        def _run_ai_conversation_turn(input_text):
            nonlocal transcript
            transcript += restart_sequence + input_text + start_sequence
            response = openai.Completion.create(
                engine="text-davinci-002",
                prompt=transcript,
                temperature=0.9,
                max_tokens=300,
                top_p=1,
                frequency_penalty=0.9,
                presence_penalty=0.6,
                stop=users,
                stream=True,
                api_key=os.environ["OPENAI_API_KEY"],
            )
            for chunk in response:
                output_text_chunk = chunk["choices"][0]["text"]
                transcript += output_text_chunk
                yield output_text_chunk

        def _iterate_ai_conversation_turn(input_text):
            buffer = ""
            for output_text in _run_ai_conversation_turn(input_text):
                buffer += output_text
                sentences = nltk.tokenize.sent_tokenize(buffer)
                if not sentences:
                    continue
                buffer = sentences.pop()
                for sentence in sentences:
                    yield sentence.strip()
            yield buffer.strip()

        yield _iterate_ai_conversation_turn
    finally:
        pass


async def main(prompt: Prompt, human_text_receiver, ai_text_sender):
    async with human_text_receiver, ai_text_sender:
        with start_ai_conversation(prompt) as ai_conversation:
            async for human_text in human_text_receiver:
                logger.info("Human: %s", human_text)
                for ai_text in ai_conversation(human_text):
                    logger.info("AI: %s", ai_text)
                    await ai_text_sender.send(ai_text)
