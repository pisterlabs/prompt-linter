from openai import OpenAI
from openai import Timeout

import helper
import logging
from telebot import types
# from datetime import datetime

client = OpenAI()

advisor_commands = {
    'Advice' : 'Advice',
    'Analyse': 'Analyse',
    'Tips' : 'Tips'
}

advice_categories = {
    "Save money on food" : "Save money on food",
    "Save money on rent" : "Save money on rent",
    "Money saving tips" : "Money saving tips",
    "Expenses saving strategies" : "Expenses saving strategies",
    "Improve income sources" : "Improve income sources",
    "Guide on Investments" : "Guide on Investments",
    "Use credit cards better" : "Use credit cards better",
    "Save money on Tax" : "Save money on Tax",
}

def run(message, bot):
    helper.read_json()
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
    markup.row_width = 2
    for c in advisor_commands:
        markup.add(c)

    msg = bot.reply_to(message, "Advisor bot", reply_markup=markup)
    bot.register_next_step_handler(msg, post_advisor_type, bot)

def post_advisor_type(message, bot):
    post_advisor_type = message.text
    if post_advisor_type == 'Advice':
        post_advisor_category(message, bot)
    elif post_advisor_type == 'Analyse':
        analyse_bot(message, bot)
    elif post_advisor_type == 'Tips':
        tips_message(message, bot)
    else:
      helper.throw_exception('Command not found', message, bot, logging)

def post_advisor_category(message, bot):

    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True)
    markup.row_width = 2

    for c in advice_categories:
      markup.add(c)

    msg = bot.reply_to(message, "Advisor bot", reply_markup=markup)
    bot.register_next_step_handler(msg, post_advisor_prompting, bot)

def post_advisor_prompting(message, bot):
    print("Category selected")

    chat_id = message.chat.id
    messages = [
      {"role": "user", "content": "You are a specialized finance advisor bot. Give me answers in a paragraph or two at max."},
    ]

    post_advisor_category = message.text

    if post_advisor_category == 'Save money on food':
        messages.append({"role": "user", "content": "Give me advice on how I can save money on food"})
    elif post_advisor_category == 'Save money on rent':
        messages.append({"role": "user", "content": "Give me advice on how I can save money on rent"})

    elif post_advisor_category == 'Money saving tips':
        messages.append({"role": "user", "content": "Give me advice on money saving tips"})

    elif post_advisor_category == 'Expenses saving strategies':
        messages.append({"role": "user", "content": "Give me advice on strategies to save on expenses"})

    elif post_advisor_category == 'Improve income sources':
        messages.append({"role": "user", "content": "Give me advice on how to improve income sourcse"})

    elif post_advisor_category == 'Guide on Investments':
        messages.append({"role": "user", "content": "Give me an introduction on investments"})

    elif post_advisor_category == 'Use credit cards better':        
        messages.append({"role": "user", "content": "Give me advice on how I can improve on using credit cards"})

    elif post_advisor_category == 'Save money on Tax':
        messages.append({"role": "user", "content": "Give me advice on how to save money on taxes"})

    else:
      helper.throw_exception('Command not found', message, bot, logging)
      return
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
      )
    msg = response.choices[0].message.content

    bot.send_message(chat_id, msg)

def analyse_bot(message, bot):
    
    messages = [
      {"role": "user", "content": "You are a specialized finance advisor bot. You must the budget items I provide and do an analysis on them"},
      {"role": "user", "content": "Keep the analysis short"},
      {"role": "user", "content": "Highlight any potential possibilites where I can save money or where I am wasting money"},
      {"role": "user", "content": "My budget is as below"},
    ]
    chat_id = message.chat.id
    data = helper.getUserData(chat_id)

    print(data)

    expenses = data['expense']
    expenses_str = ''
    if expenses:
      for exp in expenses:
        temp = exp.split(',')
        formatted = f"Expense: Date: {temp[0]} Category: {temp[1]} Amount: {temp[2]}"
        expenses_str = f"{expenses_str}\n{formatted}"

    incomes = data['income']
    incomes_str = ''
    if incomes:
      for exp in incomes:
        temp = exp.split(',')
        formatted = f"Income: Date: {temp[0]} Name: {temp[1]} Amount: {temp[2]}"
        incomes_str = f"{incomes_str}\n{formatted}"

    # recurrent = data['recurrent']

    balance = data['budget']['budget']
    saving = data['budget']['saving']

    balance_str = f"Balance in bank: {balance}"
    saving_str = f"Balance in savings account: {saving}"

    if(expenses_str):
        messages.append({"role": "user", "content": f"{expenses_str}"})
    if(incomes_str):
        messages.append({"role": "user", "content": f"{incomes_str}"})
    if(balance_str):
        messages.append({"role": "user", "content": f"{balance_str}"})
    if(saving_str):
        messages.append({"role": "user", "content": f"{saving_str}"})

    bot.send_message(chat_id, "Starting analysis")
    print('called analyse')
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
      )
    msg = response.choices[0].message.content
    print('analyse done')

    bot.send_message(chat_id, msg)


def tips_message(message, bot):
    chat_id = message.chat.id
    messages = [
      {"role": "user", "content": "You are a specialized finance advisor bot. Keep the answers short."},
      {"role": "user", "content": "Give me exactly one random tip to save on my budget"},
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
      )
    msg = response.choices[0].message.content

    bot.send_message(chat_id, msg)

