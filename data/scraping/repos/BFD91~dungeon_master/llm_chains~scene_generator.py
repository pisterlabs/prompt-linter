from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferWindowMemory, ConversationSummaryMemory, CombinedMemory, ConversationEntityMemory

from llm_chains.helpers.llms import gpt_3_5


scene_generation_template_old = """Below are scenes from a DnD adventure followed by elaborated versions of the same scene. \
The elaborations should describe in detail any events that may happen as the players play out the scene, such as:  \
encounters with characters, fights agains enemies, traps that may be triggered, items that can be found, etc. Keep in \
mind that it is the decisions of the players that drive events forward, and state which player actions that lead to \
specific events.\n\n Scene summary: The adventure begins in a small farming village called Briarwood, nestled in the \
foothills of a rugged mountain range. The players arrive in town just in time for the village's annual harvest \
festival, and are welcomed by the locals with open arms. During the festivities, the players are approached by an \
old man who introduces himself as Eustace, the village's head elder. He has urgent news - a powerful and mysterious \
man called Lord Greed has been spotted in the nearby mountain range, and he has been kidnapping villagers in the \
night. Eustace is desperate for help and offers a hefty reward to anyone brave enough to investigate Lord Greed's \
whereabouts.\n\n Elaboration: The adventure begins in a small farming village called Briarwood, \
nestled in the foothills of a rugged mountain range. As the players approach the village they hear merry flute music \
and people singing and laughing. When reaching the village they see people gathering by a feast near the largest house \
in sight, beer already overflowing. If asking any of the feasting villagers what's being celebrated, the players will \
learn that it is the annual harvest festival and will be warmly invited to have a free beer. The players are free to \
explore the village during the scene, but will find most trade apart from food and drink sold to be halted due to the \
celebrations. The players may encounter potionmaker Cara the Witch in a very quaint hut in the outskirts of the village \
and can attempt to steal valuable potions and a wand of magic missiles from her. If buying anything with higher \
denomination that copper coins, local troublemaker urchin Trunnick will attempt to pick their pockets and may be \
caught with a passive perception check. If failed, the unlucky player will lose all the money they carry. Soon after \
arriving in the village, the players will be approached by Eustace, the village elder and owner of the house where the \
feast is held. He is friendly and curious about the players, asks them questions, and when he learns that they are \
capable adventurers, he tells them about a problem the village needs help with. Eustace explains that the powerful and \
mysterious lord Greed has been terrorizing the village by having his henchmen kidnap people during nightly raids. Most \
of the times the victims disappear, but on occasion they return insane. Lord Greed is thought to perform magical \
experiments on his victims, Eustace explains, and the players are implored to try to stop lord Greed who has been \
sighted in the nearby Karabas mountain range. If the players accept, Eustace will be very happy and offer to give \
the adventurers provisions for their travels.\n\n {scene}\n\n Elaboration:"""

scene_generation_template = """Below is a short scene description of a DnD scene that is part of an adventure, followed \
by an elaborated version of the same scene. The elaboration should describe in full detail any events that may happen as \
the players play out the scene, such as: encounters with characters, fights agains enemies, traps that may be triggered, \
items that can be found, etc. Be highly specific. For example, rather than writing that the players must "fight through waves of enemies",\
list the individual enemies and their relevant attributes. Keep in mind that it is the decisions of the players that drive events forward, and state \
which player actions that lead to specific events.\n\n Scene summary: {scene}\n\n Detailed elaboration:
"""

scene_generation_prompt = PromptTemplate(
    input_variables=["scene"],
    template=scene_generation_template
)
