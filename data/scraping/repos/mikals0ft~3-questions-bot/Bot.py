import asyncio
import os.path
import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import interactions
import openai
from interactions import ActionRow, Button, SelectMenu, SelectOption

# Set up OpenAI
MODEL = 'gpt-3-5-turbo-discord'
openai.api_key = os.getenv('OPENAI_API_KEY')

bot = interactions.Client(token=os.getenv('BOT_TOKEN'))

# @bot.command(
#     name='openai_test', description='Calls OpenAi to generate questions',
# )
# async def openai_test(ctx: interactions.CommandContext):
#     await ctx.defer()
#     prompt = {
#         'role': 'user',
#         'content': 'Generate 10 question for friends in a Discord server who already know each other well. The questions are funny and quirky. The questions should be answerable in 1-3 open-ended words. They are outputted in numbered list.',
#     }
#     openai_response = openai.ChatCompletion.create(model=MODEL, temperature=1.0, messages=[prompt])
#     openai_result = openai_response.choices[0].message.content
#     await ctx.send(openai_result)


########################## GuessWhoAnswered ###########################
#### GLOBAL VARIABLES ####
question_bank_file = open(os.path.dirname(__file__) + '/guess_who_questions.txt')
question_bank = [question.rstrip('\n') for question in question_bank_file.readlines()]
guess_who_answered_votes = []
users_who_voted_already = []
guess_who_answered_participants = 0
guess_who_answered_questions = []
guess_who_answered_answers = {}
guess_who_answered_friends: Set[interactions.Member] = set()
# Map user to their score
guess_who_answered_scores = {}

@bot.command(
    name='guesswho_addplayer',
    description='Add user to Guess Who game',
    options=[
        interactions.Option(
            name='member',
            description='users to add to the game',
            type=interactions.OptionType.MENTIONABLE,
            required=True,
        ),
    ],
)
async def guess_who_addplayer(ctx: interactions.CommandContext, member: interactions.Member):
    guess_who_answered_friends.add(member)
    await ctx.send(f'Added <@{member.user.id}> to the game')


@bot.command(
    name='guesswho_start',
    description='Vote on a question to get to know your friends better, then guess who answered what!',
)
async def guess_who_answered(ctx: interactions.CommandContext):
    friends_str = ', '.join([f'<@{u.user.id}>' for u in list(guess_who_answered_friends)])

    current_user = ctx.author.mention
    guess_who_answered_participants = len(guess_who_answered_friends)

    global guess_who_answered_answers
    guess_who_answered_answers = {}
    global guess_who_answered_votes
    guess_who_answered_votes = [0, 0, 0]
    global users_who_voted_already
    users_who_voted_already = []

    global guess_who_answered_questions
    guess_who_answered_questions = random.sample(question_bank, 3)
    game_instructions = f'Hey {friends_str}! Are you ready to play Guess Who?\n\nHow well do YOU know your friends? :thinking:\n\nFirst, vote on a question to answer:'

    s1 = SelectMenu(
        custom_id='voting_menu',
        placeholder='Select a question to vote for',
        options=[SelectOption(label=question, value=i) for i, question in enumerate(guess_who_answered_questions)],
    )
    await ctx.send(game_instructions, components=ActionRow.new(s1))

    # Wait until (all participants voted or there's only one participant) or until it's been 30 seconds.
    prefix = 'Time is up! \n\n'
    message = await ctx.send('You have 30 seconds left to vote for a question!')
    for i in range(1, 31):
        await asyncio.sleep(1)
        await message.edit(content=f'You have {30 - i} seconds left to vote for each question!')
        if guess_who_answered_participants == len(users_who_voted_already):
            prefix = 'All players voted! \n\n'
            break

    index = guess_who_answered_votes.index(max(guess_who_answered_votes))
    chosen_question = guess_who_answered_questions[index]

    b1 = Button(style=1, custom_id='b1', label='press me')
    await ctx.send(
        prefix
        + '**The chosen question was:** '
        + chosen_question
        + '\n\nPress this button to answer:',
        components=ActionRow.new(b1),
    )

    # Wait while all participants have answered or it has been 60 seconds.
    prefix = 'Time is up! \n\n'
    message = await ctx.send('You have 60 seconds left to answer!')
    for j in range(1, 61):
        await asyncio.sleep(1)
        await message.edit(content=f'You have {60 - j} seconds left to answer!')
        if guess_who_answered_participants == len(guess_who_answered_answers):
            prefix = 'All players answered! \n\n'
            break

    users = list(guess_who_answered_answers.keys())
    answers = list(guess_who_answered_answers.values())
    random.shuffle(users)
    random.shuffle(answers)

    await ctx.send(
        prefix
        + 'Now, can you guess who gave each answer? :mag: \n\n'
        + '**The answers:**\n- '
        + '\n- '.join(answers)
        + '\n\n**People who answered:**\n- '
        + '\n- '.join(users)
    )

    for answer in answers:
        options = []
        for user in users:
            if answer == guess_who_answered_answers[user]:
                options.append(SelectOption(label=user, value='correct'))
            else:
                options.append(SelectOption(label=user, value='incorrect'))

        s1 = SelectMenu(custom_id=f'guessing_menu', placeholder='Select a player', options=options)
        await ctx.send(f'Who do you think answered "{answer}"?', components=ActionRow.new(s1))

@bot.component('guessing_menu')
async def guessing_menu_response(ctx, response):
    global guess_who_answered_scores
    if ctx.author.mention not in guess_who_answered_scores:
        guess_who_answered_scores[ctx.author.mention] = [0, 0]
    if response[0] == 'correct':
        guess_who_answered_scores[ctx.author.mention][0] += 1
    else:
        guess_who_answered_scores[ctx.author.mention][1] += 1
    if guess_who_answered_scores[ctx.author.mention][0] + guess_who_answered_scores[ctx.author.mention][1] == len(
        guess_who_answered_answers
    ):
        message = '{} got {}/{} correct!'.format(
            ctx.author.mention, guess_who_answered_scores[ctx.author.mention][0], len(guess_who_answered_answers)
        )
        await ctx.send(message)
    else:
        await ctx.send('Guess recorded!', ephemeral=True)


@bot.component('b1')
async def b1_response(ctx):
    index = guess_who_answered_votes.index(max(guess_who_answered_votes))
    chosen_question = guess_who_answered_questions[index]
    modal = interactions.Modal(
        title="It's time to give your answer",
        custom_id='answer_modal',
        components=[
            interactions.TextInput(
                style=interactions.TextStyleType.PARAGRAPH,
                label='The chosen question was:',
                placeholder=chosen_question,
                custom_id='text_input_response',
                min_length=1,
                max_length=100,
            ),
        ],
    )
    await ctx.popup(modal)


@bot.component('voting_menu')
async def voting_menu_response(ctx, response):
    if ctx.author.id in users_who_voted_already:
        await ctx.send('Uhh, you already voted :no_mouth:')
        return

    users_who_voted_already.append(ctx.author.id)
    chosen_idx = int(response[0])
    guess_who_answered_votes[chosen_idx] += 1

    line1 = '**You voted for:** {} \n'.format(guess_who_answered_questions[chosen_idx])
    await ctx.send(line1, ephemeral=True)


@bot.modal('answer_modal')
async def modal_response(ctx, response: str):
    index = guess_who_answered_votes.index(max(guess_who_answered_votes))
    chosen_question = guess_who_answered_questions[index]
    line1 = '**The chosen question was:** {}\n'.format(chosen_question)
    line2 = '**And you answered:** {}\n\n'.format(response)
    line3 = 'Waiting for others to answer...\n'
    global guess_who_answered_answers
    guess_who_answered_answers[ctx.author.name] = response
    await ctx.send(line1 + line2 + line3, ephemeral=True)


########################## VoteWho ###########################
#### GLOBAL VARIABLES ####
vwqf = open(os.path.dirname(__file__) + '/most_likely.txt')
vote_who_questions = [question.rstrip('\n') for question in vwqf.readlines()]
vote_who_members: Set[interactions.Member] = set()
vote_who_scores: Dict[str, int] = defaultdict(lambda: 0)
vote_who_mappings: Dict[interactions.Snowflake, str] = {}
vote_who_answers: Dict[Tuple[interactions.Snowflake, str], int] = defaultdict(lambda: 0)
# Message Id to a list of users who have voted for this message.
vote_who_voters: Dict[interactions.Snowflake, Set[str]] = {}


@bot.command(
    name='mostlikely_addplayer',
    description='Add user to the Most Likely game',
    options=[
        interactions.Option(
            name='member', description='users to add to the game', type=interactions.OptionType.USER, required=True,
        ),
    ],
)
async def votewhoadd(ctx: interactions.CommandContext, member: interactions.Member):
    vote_who_members.add(member)
    await ctx.send(f'Added <@{member.user.id}> to the game')


@bot.component('vote_who')
async def vote_who_response(ctx, response):
    if ctx.message.id in vote_who_voters and ctx.author.id in vote_who_voters[ctx.message.id]:
        await ctx.send(f'You already voted for this question!', ephemeral=True)
        return

    if ctx.message.id not in vote_who_voters:
        vote_who_voters[ctx.message.id] = set()
    vote_who_voters[ctx.message.id].add(ctx.author.id)
    user = response[0]
    vote_who_scores[user] += 1
    vote_who_answers[(ctx.message.id, user)] += 1
    question = vote_who_mappings[ctx.message.id]
    await ctx.send(f'For question "{question}" you voted for {user}.', ephemeral=True)


@bot.command(
    name='mostlikely_start', description="'Most Likely' is a game where you vote which friend matches a question best!",
)
async def votewhoplay(ctx: interactions.CommandContext):
    global vote_who_members
    n = len(vote_who_members)
    if n < 1:
        await ctx.send('To play Most Likely need to add players using /mostlikely_addplayer')
        return
    if n > 10:
        await ctx.send('You cannot play with more than 10 people, removing extra players')
        vote_who_members = random.sample(vote_who_members, 10)
        return
    else:
        questions = random.sample(vote_who_questions, n)
        formatted_list_of_members = ', '.join([f'<@{member.user.id}>' for member in vote_who_members])
        formatted_questions = formatted_string = '\n'.join(
            ['{}. {}'.format(index + 1, string.strip()) for index, string in enumerate(questions)]
        )

        ux_copy_lines = []
        ux_copy_lines.append(
            "Hey there {}! You've been invited to play Most Likely!\n\n".format(formatted_list_of_members)
        )
        ux_copy_lines.append(
            '**Here are your {} questions. Vote for which friend matches each question best! :speech_left: :question:**\n'.format(
                n
            )
        )
        ux_copy_lines.append('{}'.format(formatted_questions))
        ux_copy_lines.append('\n\nYou have 30 seconds to vote for each question! :clock4: \n\n')
        ux_copy = ''.join(ux_copy_lines)
        await ctx.send(ux_copy)
        for question in questions:
            message = await ctx.send(
                question,
                components=interactions.SelectMenu(
                    custom_id='vote_who',
                    placeholder='Select a player',
                    options=[
                        interactions.SelectOption(label=member.user.username, value=member.user.username)
                        for member in vote_who_members
                    ],
                ),
            )
            vote_who_mappings[message.id] = question
        message = await ctx.send('You have 30 seconds left to vote for each question!')
        for i in range(1, 31):
            await asyncio.sleep(1)
            await message.edit(content=f'You have {30 - i} seconds left to vote for each question!')

        vote_who_dict: Dict[interactions.Snowflake, List[str]] = defaultdict(lambda: [])
        vote_who_answers_sorted = dict(sorted(vote_who_answers.items(), key=lambda x: x[1], reverse=True))
        for (q, u), c in vote_who_answers_sorted.items():
            vote_who_dict[q].append(f'{u} got {c} vote(s)')
        vote_who_list: List[Tuple[interactions.Snowflake, str]] = [
            (question, '\n'.join(scores)) for question, scores in vote_who_dict.items()
        ]

        if vote_who_list == []:
            await ctx.send(
                'The time is up and no one voted :frowning: \n\n Want to play another round? Just run: /mostlikely_start\n'
            )
            return

        vote_who_answers_formatted = '\n\n'.join([f'**{vote_who_mappings[q]}**\n{a}' for q, a in vote_who_list])
        await ctx.send(
            f'The votes are in! :tada: \n\n{vote_who_answers_formatted}\n\n Want to play another round? Just run: /mostlikely_start\n'
        )
        vote_who_scores_formatted = '\n'.join([f'{u}: {s}' for u, s in vote_who_scores.items()])
        # await ctx.send(f'Here \n {vote_who_scores_formatted}')

        vote_who_mappings.clear()
        vote_who_answers.clear()
        vote_who_voters.clear()


@bot.command(
    name='mostlikely_endgame', description='End the MostLikely game and report the scores',
)
async def votewhoendgame(ctx: interactions.CommandContext):
    vote_who_scores_formatted = '\n'.join([f'{u}: {s}' for u, s in vote_who_scores.items()])
    await ctx.send(f'Game over! Here were the scores for each player \n {vote_who_scores_formatted}')

    vote_who_members.clear()
    vote_who_scores.clear()
    vote_who_mappings.clear()
    vote_who_answers.clear()


########################## QOTD ###########################
#### GLOBAL VARIABLES ####


import asyncio
from contextlib import suppress


class Periodic:
    def __init__(self, func):
        self.func = func
        self.is_started = False
        self._task = None

    async def start(self, ctx, time):
        if not self.is_started:
            self.is_started = True
            # Start task to call func periodically:
            self._task = asyncio.ensure_future(self._run(ctx, time))

    async def stop(self):
        if self.is_started:
            self.is_started = False
            # Stop task and await it stopped:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def _run(self, ctx, time):
        while True:
            await self.func(ctx)
            await asyncio.sleep(time * 60)


async def schedule_question(ctx: interactions.CommandContext):
    question = random.sample(question_bank, 1)[0]
    message = await ctx.send(question)
    await message.create_thread(question)


schedule = Periodic(func=schedule_question)


@bot.command(
    name='qotd_start',
    description='Enable the bot to ask a question and create a thread for server members to discuss',
    options=[
        interactions.Option(
            name='mins',
            description='frequency in which the bot creates the question thread',
            type=interactions.OptionType.INTEGER,
            required=True,
        ),
    ],
)
async def qotd_start(ctx: interactions.CommandContext, mins: int):
    await ctx.send(f'I will create a question thread in this channel every {mins} minutes')
    await schedule.start(ctx, mins)


@bot.command(name='qotd_end', description='Stop bot from sending questions')
async def qotd_end(ctx: interactions.CommandContext):
    await ctx.send(f'I will no longer create question threads in this channel')
    await schedule.stop()


bot.start()
