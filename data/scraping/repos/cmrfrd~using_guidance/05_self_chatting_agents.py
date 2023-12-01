from guidance import Program, llms
from pydantic import BaseModel

role_simulator = Program(
    """
{{#system~}}
You are a helpful assistant
{{~/system}}

{{#user~}}
## Goal

You are an AI game bot for the game "GuessThatWord". You will assume one of two roles: the guesser or the answerer.

## Rules

### For the Guesser:

1. The guesser must phrase the questions in a way that can be answered with "yes" or "no" by the answerer.
2. The guesser should ask specific and strategic questions to narrow down the possibilities.
3. The guesser should listen carefully to the answerer's responses and use the information to make informed guesses.
4. The guesser should avoid repeating questions already asked to maximize efficiency.
5. If the guesser correctly guesses the answer, they win the game.

### For the Answerer:

1. The answerer should keep their chosen answer secret and not reveal it until the guesser makes a correct guess.
2. The answerer must answer the guesser's questions truthfully with a simple "yes" or "no."
3. The answerer should provide concise responses without giving away too much information.
4. The answerer should pay attention to the guesser's questions and avoid unintentionally revealing clues.
5. The answerer should not change the chosen answer once the game begins.
{{~/user}}

{{#assistant~}}
{{#if (equal role "Guesser")}}
You are the guesser. Ask yes-or-no questions to determine the secret word. Start by 
asking macro general questions. Once you get enough information about the secret word,
only then should you guess.

If you get 3 "no" responses in a row, ask more general questions again to 
get more information.

Output the question you want to ask the answerer and nothing else.
{{else}}
You are the answerer and the secret word is "{{secret_word}}".
{{/if}} 
You will assume this role and perform it to the best of your ability.
{{~/assistant}}

{{~! Then the conversation unrolls }}
{{#user~}}
Comment: Remember, respond as the {{role}}.
Conversation so far:
{{#each conversation}}{{this.role}}: {{this.message}}
{{/each}}
{{~/user}}

{{#if (equal role "Guesser")}}
{{#assistant~}}
{{gen 'question' temperature=0 max_tokens=60}}
{{~/assistant}}
{{else}}
{{#assistant~}}
{{#select 'response'}}Yes{{or}}No{{or}}Sort of{{/select}}
{{~/assistant}}
{{/if}}
"""
)


class ChatMessage(BaseModel):
    role: str
    message: str


def has_won(secret_word: str, convo: list[ChatMessage]) -> bool:
    """Check if the guesser has won."""
    return any(
        [secret_word.lower() in m.message for m in filter(lambda m: m.role == "Guesser", convo)]
    )


secret_word = "chicken"

guesser = role_simulator(
    role="Guesser",
    secret_word=secret_word,
    llm=llms.OpenAI("gpt-4"),
    await_missing=True,
    caching=False,
)
answerer = role_simulator(
    role="Answerer",
    secret_word=secret_word,
    llm=llms.OpenAI("gpt-4"),
    await_missing=True,
    caching=False,
)

conversation: list[ChatMessage] = []
print(f"Secret word: {secret_word}")
for iteration in range(100):
    guesser_response = guesser(conversation=list(map(lambda c: c.dict(), conversation)))
    conversation.append(ChatMessage(role="Guesser", message=guesser_response["question"]))
    print(f"Guesser: {guesser_response['question']}")

    answerer_response = answerer(conversation=list(map(lambda c: c.dict(), conversation)))
    conversation.append(ChatMessage(role="Answerer", message=answerer_response["response"]))
    print(f"Answerer: {answerer_response['response']}")

    if has_won(secret_word, conversation):
        print("Winner!")
        break
else:
    print("oh no! you lost!")
