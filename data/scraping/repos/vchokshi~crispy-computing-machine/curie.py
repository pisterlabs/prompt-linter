import os
import openai

import pprint

pp = pprint.PrettyPrinter(indent=1)

openai.api_key = os.getenv("OPENAI_API_KEY")

response = openai.Completion.create(
  model="text-curie-001",
  prompt="What are the key points from this text:\n\n\"\"\"\nPluto (minor planet designation: 134340 Pluto) is a dwarf planet in the Kuiper belt, a ring of bodies beyond the orbit of Neptune. It was the first and the largest Kuiper belt object to be discovered.\n\nPluto was discovered by Clyde Tombaugh in 1930 and declared to be the ninth planet from the Sun. After 1992, its status as a planet was questioned following the discovery of several objects of similar size in the Kuiper belt. In 2005, Eris, a dwarf planet in the scattered disc which is 27% more massive than Pluto, was discovered. This led the International Astronomical Union (IAU) to define the term \"planet\" formally in 2006, during their 26th General Assembly. That definition excluded Pluto and reclassified it as a dwarf planet.\n\nPluto is the ninth-largest and tenth-most-massive known object directly orbiting the Sun. It is the largest known trans-Neptunian object by volume but is less massive than Eris. Like other Kuiper belt objects, Pluto is primarily made of ice and rock and is relatively small—one-sixth the mass of the Moon and one-third its volume. It has a moderately eccentric and inclined orbit during which it ranges from 30 to 49 astronomical units or AU (4.4–7.4 billion km) from the Sun. This means that Pluto periodically comes closer to the Sun than Neptune, but a stable orbital resonance with Neptune prevents them from colliding. Light from the Sun takes 5.5 hours to reach Pluto at its average distance (39.5 AU).\n\nPluto has five known moons: Charon (the largest, with a diameter just over half that of Pluto), Styx, Nix, Kerberos, and Hydra. Pluto and Charon are sometimes considered a binary system because the barycenter of their orbits does not lie within either body.\n\"\"\"\n\nThe key points are:\n\n1.",
  temperature=0.5,
  max_tokens=233,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
  stop=["\"\"\""]
)
pp.pprint(response)
