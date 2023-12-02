import cohere
co = cohere.Client("o2KYh1CEVLYwS0ePRO4VmKsIWZuaSuz5cDS1MWjZ")

prompt = f"""Passage: Is Wordle getting tougher to solve? Players seem to be convinced that the game has gotten harder in recent weeks ever since The New York Times bought it from developer Josh Wardle in late January. The Times has come forward and shared that this likely isn't the case. That said, the NYT did mess with the back end code a bit, removing some offensive and sexual language, as well as some obscure words There is a viral thread claiming that a confirmation bias was at play. One Twitter user went so far as to claim the game has gone to "the dusty section of the dictionary" to find its latest words.

TLDR: Wordle has not gotten more difficult to solve.
--
Passage: ArtificialIvan, a seven-year-old, London-based payment and expense management software company, has raised $190 million in Series C funding led by ARG Global, with participation from D9 Capital Group and Boulder Capital. Earlier backers also joined the round, including Hilton Group, Roxanne Capital, Paved Roads Ventures, Brook Partners, and Plato Capital.

TLDR: ArtificialIvan has raised $190 million in Series C funding.
--
Passage: The National Weather Service announced Tuesday that a freeze warning is in effect for the Bay Area, with freezing temperatures expected in these areas overnight. Temperatures could fall into the mid-20s to low 30s in some areas. In anticipation of the hard freeze, the weather service warns people to take action now.

TLDR:"""

prompt_2 = "The goal of text summarization is to condense the original text into a shorter version that retains the most important information. In this example, we want to summarize a passage from a news article into its main point.--"

prompt_3 = f"""Passage: More visible signs of war are emerging in Russia, with air defences being placed on Moscow rooftops and Kremlin officials sharply decrying the widening array of weapons the West is providing to Ukraine. 

Reports emerged on social media last week that anti-aircraft missiles had been spotted on key buildings in central Moscow, including at a defence ministry command centre. Military drills also took place outside the capital, with Russia's defence ministry saying troops "conducted an exercise to repel a mock air attack."

There is nothing abnormal about countries having air defences around key military installations or major cities. But new military drills testing S-300 mobile surface-to-air batteries around the capital, coupled with social media reports that Pantsir S-1 anti-aircraft missiles had been mounted on buildings in central Moscow, suggest Russia may be reinforcing its air defences. 

These developments have occurred alongside growing tensions about tanks and other heavy weapons that Ukraine has been promised by Western allies, leaving Russia weighing their implications for the conflict.

In summary:

--"""

response = co.generate(
    model='xlarge',
    prompt = prompt_3,
    max_tokens=80,
    temperature=5,
    stop_sequences=["--"])

summary = response.generations[0].text
print(summary)

