import openai
from .teammate import prompt_for_teammate
from .teammate import prompt_for_teammate_after_response


def teammates_guess(clue_word, number_of_guesses, cards):
    prompts = []
    responses = []
    init_prompt = prompt_for_teammate(cards, clue_word, number_of_guesses)
    answers = {'blue': f' Correct answer.\nYou have {number_of_guesses} guesses. Please guess another word:',
               'red': 'Wrong! It is another team. You have no more guesses left.',
               'neutral': 'Wrong! It is neutral. You have no more guesses left.',
               'bomb': 'It is the bomb. GAME OVER!',
               'invalid word': 'This word is not in the list'}

    inverted_cards = {}
    for team_name, cards_for_team in cards.items():
        for card_text in cards_for_team:
            inverted_cards[card_text.lower()] = team_name

    while number_of_guesses > 0:
        response = openai.Completion.create(
            model= "davinci:ft-university-of-cambridge:codenames-2023-01-16-11-18-30",
            prompt=init_prompt,
            temperature=0.7,
            max_tokens=6,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        number_of_guesses -= 1
        answers['blue'] = f' Correct answer.\nYou have {number_of_guesses} guesses. Please guess another word:'
        last_response = response['choices'][0]['text']

        last_response = last_response.replace('\n', '')
        last_response = last_response.strip().split(' ')[0]
        responses.append(last_response)

        prompts.append(init_prompt)
        init_prompt = init_prompt + '\n\n' + last_response

        outcome = inverted_cards.get(last_response.lower(), 'invalid word')

        init_prompt = init_prompt + '\n\n' + \
            prompt_for_teammate_after_response(answers[outcome])
        # prompts.append(prompt_for_teammate_after_response)
        if outcome != 'blue':
            break
    return outcome == 'blue', prompts, responses
