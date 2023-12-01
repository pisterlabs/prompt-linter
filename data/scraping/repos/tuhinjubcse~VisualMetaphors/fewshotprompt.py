import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

start_sequence = "\n Objects to be included:"

response = openai.Completion.create(
  model="text-davinci-002",
  prompt="Your task will be to explain a metaphor with rich visual details along with the provided objects to be included and implicit meaning. Make sure to include the implicit meaning and the objects to be included in the elaboration. \n\n\n1. Metaphor: My lawyer is a shark. \nObjects to be included: Lawyer, Shark\nImplicit Meaning: fierce\nVisual Elaboration: A shark in a suit with fierce eyes and a suitcase and a mouth open with pointy teeth.\n\n2. Metaphor: I've reached my boiling point.\nObjects to be included: Person, Boiling Pot\nImplicit Meaning: anger\nVisual Elaboration: A boiling pot of water with a person's head popping out of the top, steam coming out of their ears, and an angry expression on their face. \n\n3. Metaphor: Joe: that's because you're like a snail surfing on molasses. \nObjects to be included: Person like a snail, Snail on molasses\nImplicit Meaning: slow\nVisual Elaboration: A person with a snail shell on their back slowly sliding down a hill of molasses.\n\n4. Absence is the dark room in which lovers develop negatives\nObjects to be included: Darkroom, Negative Film Strip with a red heart, Person Implicit\nMeaning: ominous and lonely\nVisual Elaboration: An ominous dark room with film strip negatives hanging and a red heart in the center with a person in the corner looking sad and lonely\n\n5. Metaphor: My heart is a rose thorn\n Objects to be included: Heart, Thorn\nImplicit Meaning: prickly\nVisual Elaboration: A heart with a prickly thorn coming out of the center and barbs going outwards.\n\n6: Metaphor: Death is a Fantasy",
  temperature=0.7,
  max_tokens=256,
  top_p=1,
  frequency_penalty=0.5,
  presence_penalty=0.5
)

print(response)
