import openai
import os

openai.api_key = os.environ.get('GPT3_API_KEY')

openai.Engine.list()


# Completion of poki poems

prompt = "I Love The Zoo : roses are red,  violets are blue.   i love the zoo.   do you? The scary forest. : the forest is really haunted.  i believe it to be so.  but then we are going camping. A Hike At School: i took a hike at school today  and this is what i saw     bouncing balls      girls chatting against the walls     kids climbing on monkey bars     i even saw some teachers' cars     the wind was blowing my hair in my face     i saw a mud puddle,  but just a trace all of these things i noticed just now on my little hike. Computer :  "

response = openai.Completion.create(
  engine="davinci",
  prompt=prompt,
  max_tokens=100
)

print(response['choices'][0]['text'])

# Translation

prompt = "English: How are you ? French:Quel est votre nom? English: What is your name ? French: Wie lautet dein Name ? English: Where are you from ? French:"

response = openai.Completion.create(
  engine="davinci",
  prompt=prompt,
  max_tokens=100
)

# Summarization

prompt = "By design, a blockchain is resistant to modification of its data. This is because once recorded, the data in any given block cannot be altered retroactively without alteration of all subsequent blocks. For use as a distributed ledger, a blockchain is typically managed by a peer-to-peer network collectively adhering to a protocol for inter-node communication and validating new blocks. Although blockchain records are not unalterable, blockchains may be considered secure by design and exemplify a distributed computing system with high Byzantine fault tolerance. The blockchain has been described as an open, distributed ledger that can record transactions between two parties efficiently and in a verifiable and permanent way. Summarizing this we can say "

response = openai.Completion.create(
  engine="davinci",
  prompt=prompt,
  max_tokens=100
)

# Poetry

prompt = "There once was a wonderful star \
Who thought she would go very far \
Until she fell down \
And looked like a clown \
She knew she would never go far."



response = openai.Completion.create(
  engine="davinci",
  prompt=prompt,
  max_tokens=200
)

# Question Answer

prompt = "Me:Hi! Can you help me with good restaurants I can visit ? AI: Sure, what cuisine would you like ? \
Me: I prefer continental AI:"


response = openai.Completion.create(
  engine="davinci",
  prompt=prompt,
  max_tokens=100
)

# Story writing

prompt = "Once there was a monkey and a duck who were good"


response = openai.Completion.create(
  engine="davinci",
  prompt=prompt,
  max_tokens=200
)
