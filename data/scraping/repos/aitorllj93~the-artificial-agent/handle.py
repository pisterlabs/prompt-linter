
from core.adapters import openai, telegram
from chat.commands.enumerate import prompt

from notes.get_notes import get_today_daily_note


async def handle(
    params: dict,
    command: dict,
    update: telegram.Update,
    context: telegram.ContextTypes.DEFAULT_TYPE
):
    note = await get_today_daily_note()

    tasks = note.list_tasks_from_section(
        "Assistant", params['completed'] or False)

    tasksStr = ''

    for index, task in enumerate(tasks):
        tasksStr += f'{index + 1}. {task.text}\n'

    prompt_text = prompt(
        tasksStr,
        command['personality']
    )

    message = await openai.generate_text_from_prompt(prompt_text)

    await telegram.send_text_message(message, update, context)
