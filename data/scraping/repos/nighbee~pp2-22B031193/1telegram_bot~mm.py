import openai
import torch
from aiogram import Bot, Dispatcher, executor, types
from transformers import AutoModelForCausalLM, AutoTokenizer

botToken = '5965750764:AAGzp7lxFO0Se2s_uBYmYL6LiOL_iRx-Jg4'
openAi = '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'
model_name = "microsoft/DialoGPT-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

bot = Bot(token=botToken)
dp = Dispatcher(bot)

async def welcome(message: types.Message):
    await message.reply('Hi, how can I help?')

async def comprehend(prompt: str, model, tokenizer) -> str:
    # Encode the prompt and generate the response
    prompt = 'User: ' + prompt + '\nAI:'
    prompt_input = tokenizer.encode(prompt, return_tensors='pt')
    generated_output = model.generate(prompt_input, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(generated_output[:, prompt_input.shape[-1]:][0], skip_special_tokens=True)

    return response

async def echo(message: types.Message):
    prompt = message.text
    response = await comprehend(prompt, model, tokenizer)

    await message.answer(response)

if __name__ == '__main__':
    dp.register_message_handler(welcome, commands=['start'])
    dp.register_message_handler(echo)
    executor.start_polling(dp, skip_updates=True)
