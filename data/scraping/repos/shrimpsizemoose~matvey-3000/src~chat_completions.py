import os
import textwrap
from dataclasses import dataclass

import anthropic
import openai


openai_client = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
anthro_client = anthropic.AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))


@dataclass(frozen=True)
class TextResponse:
    success: bool
    text: str

    @classmethod
    async def generate(cls, config, chat_id, messages):
        provider = config.provider_for_chat_id(chat_id)
        if provider == config.PROVIDER_OPENAI:
            return await cls._generate_openai(
                openai_client,
                config.model_for_chat_id(chat_id),
                messages,
            )
        elif provider == config.PROVIDER_ANTHROPIC:
            return await cls._generate_anthropic(
                anthro_client,
                config.model_for_chat_id(chat_id),
                messages,
            )
        else:
            return cls(
                success=False,
                text=f'Unsupported provider: {config.provider}'
            )

    @classmethod
    async def _generate_openai(cls, client, model, messages):
        payload = [
            {'role': role, 'content': text}
            for role, text in messages
        ]
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=payload,
            )
        except openai.RateLimitError as e:
            return cls(
                success=False,
                text=f'Кажется я подустал и воткнулся в рейт-лимит. Давай сделаем перерыв ненадолго.\n\n{e}',  # noqa
            )
        except openai.BadRequestError as e:
            return cls(
                success=False,
                text=f'Beep-bop, кажется я не умею отвечать на такие вопросы:\n\n{e}',  # noqa
             )
        except TimeoutError as e:
            return cls(
                success=False,
                text=f'Кажется у меня сбоит сеть. Ты попробуй позже, а я пока схожу чаю выпью.\n\n{e}',  # noqa
            )
        else:
            return cls(
                success=True,
                text=response.choices[0].message.content,
            )

    @classmethod
    async def _generate_anthropic(cls, client, model, messages):
        user_tag = anthropic.HUMAN_PROMPT
        bot_tag = anthropic.AI_PROMPT
        system = [text for role, text in messages if role == 'system'][0]

        # claude 2.1 might need different format for prompting since it has system prompts
        # full_prompt = f'{system} {user_tag}{human}{bot_tag}'

        # claude instant 1.2 though... Needs a different thing
        prompt = textwrap.dedent(
            """
                Here are a few back and forth messages between user and a bot.
                User messages are in tag <user>, bot messages are in tag <bot>
            """.strip()
        )
        prompt = [f'{user_tag}{prompt}']
        for role, text in messages:
            if role == 'system':
                continue
            prompt.append(f'\n<{role}>{text}</{role}>')
        prompt.append(f'\n{system}')
        prompt.append('\nTake content of last unpaired "user" and use this as completion prompt.')
        prompt.append(f'\nRespond ONLY with text and no tags.{bot_tag}')
        prompt = ''.join(prompt)

        #print(prompt)
        try:
            response = await client.completions.create(
                model=model,
                max_tokens_to_sample=1024,  # no clue about this value
                prompt=prompt,
            )
        except openai.RateLimitError as e:
            return cls(
                success=False,
                text=f'Кажется я подустал и воткнулся в рейт-лимит. Давай сделаем перерыв ненадолго.\n\n{e}',  # noqa
            )
        except openai.BadRequestError as e:
            return cls(
                success=False,
                text=f'Beep-bop, кажется я не умею отвечать на такие вопросы:\n\n{e}',  # noqa
             )
        except TimeoutError as e:
            return cls(
                success=False,
                text=f'Кажется у меня сбоит сеть. Ты попробуй позже, а я пока схожу чаю выпью.\n\n{e}',  # noqa
            )
        else:
            #print(response.completion)
            completion = response.completion.replace("<", "[").replace(">","]")
            return cls(
                success=True,
                text=completion,
            )


@dataclass(frozen=True)
class ImageResponse:
    success: bool
    image_url: str

    @classmethod
    async def generate(cls, prompt):
        # no other providers yet so meh
        openai_client = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        return await cls._generate_openai(openai_client, prompt)

    @classmethod
    async def _generate_openai(cls, client, prompt):
        img_gen_reply = await client.images.generate(
            prompt=prompt,
            n=1,
            size='512x512',
        )
        return cls(success=True, image_url=img_gen_reply.data[0].url)


