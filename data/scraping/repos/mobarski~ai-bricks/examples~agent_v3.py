import sys; sys.path[0:0] = ['.','..'] # for local testing

from ai_bricks.api import openai
from ai_bricks.api import anthropic
from ai_bricks import agent

aa = agent.actions
aa.wikipedia_search_many.__name__ = 'wikipedia-summary'
aa.wikipedia_get_data.__name__ = 'wikipedia-data'
actions = [
    aa.wikipedia_search_many,
    aa.wikipedia_get_data,
    aa.python_eval,
]

#q = "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?"
#q = "Which planet has the moon with the largest diameter - Jupiter or Saturn?"
#q = "Which planet has the biggest moon - Jupiter or Saturn?"
q = "Which planet has the heaviest moon - Jupiter or Saturn?"
#q = "Which planet has the moon with the largest surface gravity force - Jupiter or Saturn?"
#q = "Which planet has the moon with the largest surface gravity force - Jupiter or Saturn? Give the answer in m/s^2."
#q = "Which planet has the moon with the largest surface gravity force - Jupiter or Saturn? Give the answer in gs."
#q = "Which planet has bigger moon - Jupiter or Saturn?"

#model = openai.model('gpt-3.5-turbo', temperature=0.0) # key from OPENAI_API_KEY env variable
#model = anthropic.model('claude-v1.2', temperature=0.0) # key from ANTHROPIC_API_KEY env variable
model = anthropic.model('claude-instant-v1', temperature=0.0) # key from ANTHROPIC_API_KEY env variable
a = agent.get('react')(model=model, actions=actions)
answer = a.run(q, n_turns=10)

#print(model.complete('2+2='))
